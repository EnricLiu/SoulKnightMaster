pub mod event;
pub mod pool;
pub mod error;
pub mod action;
mod test;
mod health;
pub mod server;

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::Mutex;
use tokio::task::spawn_blocking;
use image::{ImageBuffer, Rgba};
use adb_client::{ADBDeviceExt, ADBServerDevice};

use error::Error;
use pool::DeviceConnPool;
use event::{Key, KeyValue};
use action::{InputAction, InputFactory};

use crate::utils::{get_id, log, perf_log, perf_timer, Position, START};

#[derive(Debug)]
pub enum ShellCommand<'a> {
    Action(InputAction<'a>),
    Raw(String)
}

impl ShellCommand<'_> {
    pub fn to_string(&self) -> String {
        match self {
            ShellCommand::Action(action) => action.cmd(),
            ShellCommand::Raw(cmd) => cmd.clone(),
        }
    }
    
    pub fn into_string(self) -> String {
        match self {
            ShellCommand::Action(action) => action.cmd(),
            ShellCommand::Raw(cmd) => cmd,
        }
    }
}

impl<'a> From<InputAction<'a>> for ShellCommand<'a> {
    fn from(action: InputAction<'a>) -> Self {
        ShellCommand::Action(action)
    }
}

impl From<String> for ShellCommand<'_> {
    fn from(str: String) -> Self {
        ShellCommand::Raw(str)
    }
}


pub struct Device<const POOL_SIZE: usize> {
    iden:   String,
    pending_cnt: AtomicUsize,
    pool:   Arc<Mutex<DeviceConnPool<POOL_SIZE>>>,

    action_man: InputFactory,
    // health:     NodeHealth
}

impl<const POOL_SIZE: usize> Default for Device<POOL_SIZE> {
    fn default() -> Self {
        let pool = DeviceConnPool::default();
        Device::new(pool, "/dev/input/event4")
    }
}

impl<const POOL_SIZE: usize> Device<POOL_SIZE> {
    pub fn new(pool: DeviceConnPool<POOL_SIZE>, ev_device: &str) -> Self {
        let ret = Device {
            iden:           "default".to_string(),
            pool:           Arc::new(Mutex::new(pool)),
            action_man:     InputFactory::new(ev_device),
            pending_cnt:    AtomicUsize::new(0),
        };
        ret
    }

    pub async fn pool_len(&self) -> usize {
        let pool = self.pool.lock().await;
        pool.len().await
    }

    pub fn get_pending_cnt(&self) -> usize {
        self.pending_cnt.load(Ordering::SeqCst)
    }

    pub async fn connect(&self) -> Result<(), Error> {
        let pool = self.pool.lock().await;
        pool.connect().await?;
        Ok(())
    }

    pub async fn get_conn_with_prior(&self, prior: usize) -> Result<ADBServerDevice, Error> {
        let pool = self.pool.lock().await;
        if let Some(device) = pool.try_get().await.ok() {
            return Ok(Arc::try_unwrap(device).or(Err(Error::ArcFailedUnwrap()))?);
        } else {
            drop(pool);
            self.pending_cnt.fetch_add(1, Ordering::SeqCst);
        }

        // concurrency limitation
        // default: waiting for 10*100ms, every pending req before contributes 1*100ms more.
        let poll_epoch = 10 + 1 * self.pending_cnt.load(Ordering::SeqCst);
        let poll_epoch = poll_epoch.min(prior);

        for _ in 0..poll_epoch {
            let pool = self.pool.lock().await;
            if let Some(device) = pool.try_get().await.ok() {
                self.pending_cnt.fetch_sub(1, Ordering::SeqCst);
                return Ok(Arc::try_unwrap(device).or(Err(Error::ArcFailedUnwrap()))?);
            };
            drop(pool);
            log("Waiting");
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        self.pending_cnt.fetch_sub(1, Ordering::SeqCst);
        Ok(Arc::try_unwrap(self.pool.lock().await.make_conn().await).or(Err(Error::ArcFailedUnwrap()))?)
    }

    pub async fn get_conn(&self) -> Result<ADBServerDevice, Error> {
        self.get_conn_with_prior(usize::MAX).await
    }

    pub async fn read_frame_buf(&self) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, Error> {
        let mut conn = self.get_conn().await?;

        let task_id = get_id();
        
        let start = perf_timer();
        log(&format!("[{task_id}] Reading FB!"));

        let task = spawn_blocking(move || {
            let res = conn.framebuffer_inner();
            perf_log(&format!("[{task_id}] FB Read!"), start);
            (res, conn)
        });

        let pool = self.pool.clone();
        let res = tokio::spawn(async move {
            let (res, conn) = task.await?;
            pool.lock().await.release(Arc::new(conn)).await;
            Ok(res?)
        }).await?;

        res
    }

    pub async fn read_frame_buf_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut conn = self.get_conn().await?;

        let task_id = get_id();
        perf_log(&format!("[{task_id}] Spawn!!"), Some(*START));

        let task = spawn_blocking(move || {
            let res = conn.framebuffer_bytes();
            perf_log(&format!("[{task_id}] Joined!!"), Some(*START));
            (res, conn)
        });

        let pool = self.pool.clone();
        let res = tokio::spawn(async move {
            let (res, conn) = task.await?;
            pool.lock().await.release(Arc::new(conn)).await;
            Ok(res?)
        }).await?;

        res
    }
}

impl<'a, const POOL_SIZE: usize> Device<POOL_SIZE> {
    pub(crate) async fn send_action(&self, action: &ShellCommand<'_>) -> Result<(), Error> {
        let mut conn = self.get_conn().await?;
        let command = action.to_string();
        let task = spawn_blocking(move || {
            let res = conn.shell_command(
                &[command.as_ref()],
                &mut ShellReturnHandler{});

            (res, conn)
        });
        let (_, conn) = task.await?;
        self.pool.lock().await.release(Arc::new(conn)).await;
        Ok(())
    }

    pub async fn key_down(&self, key: Key) -> Result<(), Error> {
        let action = self.action_man.get_key_down_action(key).await;
        let command = action.into();
        log(&format!("KeyDown {:?}", &command));
        self.send_action(&command).await?;
        Ok(())
    }
    pub async fn key_up(&self, key: Key) -> Result<(), Error> {
        let action = self.action_man.get_key_up_action(key).await;
        let command = action.into();
        log(&format!("KeyUp {:?}", &command));
        self.send_action(&command).await?;
        Ok(())
    }

    pub async fn touch_down(&self, iden: &'a str, pos: Position<u32>) -> Result<(), Error> {
        let action = self.action_man.get_touch_down_action(iden, pos).await;
        if action.is_none() { return Ok(()) };
        let command = action.unwrap().into();
        log(&format!("TouchDown['{iden}'] {:?}", &command));
        self.send_action(&command).await?;
        Ok(())
    }

    pub async fn touch_move(&self, iden: &'a str, pos: Position<u32>) -> Result<(), Error> {
        let action = self.action_man.get_touch_move_action(iden, pos).await;
        if action.is_none() { return Ok(()) };
        let command = action.unwrap().into();
        log(&format!("TouchMove['{iden}'] {:?}", &command));
        self.send_action(&command).await?;
        Ok(())
    }

    pub async fn touch_up(&self, iden: &'a str) -> Result<(), Error> {
        let action = self.action_man.get_touch_up_action(iden).await;
        if action.is_none() { return Ok(()) };
        let command = action.unwrap().into();
        log(&format!("TouchUp['{iden}'] {:?}", &command));
        self.send_action(&command).await?;
        Ok(())
    }
}

impl<'a , const POOL_SIZE: usize> Device<POOL_SIZE> {
    pub async fn tap(&self, pos: Position<u32>) -> Result<(), Error> {
        let command = ShellCommand::Raw(format!("input tap {} {}", pos.x, pos.y));
        self.send_action(&command).await?;
        Ok(())
    }

    pub async fn motion(&self, pos: Position<u32>, val: KeyValue) -> Result<(), Error> {
        let val = match val {
            KeyValue::Down  => "down",
            KeyValue::Up    => "up",
        };
        let command = ShellCommand::Raw(format!("input motionevent {val} {} {}", pos.x, pos.y));
        self.send_action(&command).await?;
        Ok(())
    }
}



pub struct EventParser {
    
}

impl Write for EventParser {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        println!("{}", String::from_utf8_lossy(buf));
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub struct ShellReturnHandler {
    
}
impl Write for ShellReturnHandler {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        println!("{}", String::from_utf8_lossy(buf));
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
