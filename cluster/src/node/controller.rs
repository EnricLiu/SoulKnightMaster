use std::io::{Write};
use std::net::SocketAddrV4;

use std::sync::{Arc, LazyLock};
use std::sync::atomic::{AtomicUsize, Ordering};

use adb_client::{ADBDeviceExt, ADBServerDevice};
use image::{ImageBuffer, Rgba};
use tokio::sync::{Mutex};
use tokio::task::spawn_blocking;
use crate::node::error::Error;
use crate::node::pool::DeviceConnPool;
use super::event::{NodeEvent, KeyValue, Key};
use super::action::{NodeAction, ActionFactory};

use chrono::Local;
use crate::utils::{get_id, log, Position};

pub enum AdbSignal {
    Command(Vec<NodeEvent>),
    Raw(String)
}

static START: LazyLock<i64> = LazyLock::new(|| {
    Local::now().timestamp_micros()
});


pub struct Node<'a, const POOL_SIZE: usize> {
    iden:   &'a str,
    pending_cnt: AtomicUsize,
    pool:   Arc<Mutex<DeviceConnPool<POOL_SIZE>>>,
    
    action_man: ActionFactory<'a>,
}

impl<const POOL_SIZE: usize> Default for Node<'_, POOL_SIZE> {
    fn default() -> Self {
        let pool = DeviceConnPool::default();
        Node::new(pool, "/dev/input/event4")
    }
}

impl<'a, const POOL_SIZE: usize> Node<'a, POOL_SIZE> {
    pub fn new(pool: DeviceConnPool<POOL_SIZE>, ev_device: &'a str) -> Self {
        let ret = Node {
            iden:           "default",
            pool:           Arc::new(Mutex::new(pool)),
            action_man:     ActionFactory::new(ev_device),
            pending_cnt:    AtomicUsize::new(0),
        };
        ret
    }
    
    pub async fn pool_len(&self) -> usize {
        let pool = self.pool.lock().await;
        pool.len().await
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
    
    async fn read_frame_buf(&self) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, Error> {
        let mut conn = self.get_conn().await?;

        let task_id = get_id();
        log(&format!("[{task_id}] Spawn!!  [{}]", Local::now().timestamp_micros() - *START));

        let task = spawn_blocking(move || {
            let res = conn.framebuffer_inner();
            log(&format!("[{task_id}] Joined!! [{}]", Local::now().timestamp_micros() - *START));
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
    
    
    async fn read_frame_buf_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut conn = self.get_conn().await?;
        
        let task_id = get_id();
        log(&format!("[{task_id}] Spawn!!  [{}]", Local::now().timestamp_micros() - *START));
        
        let task = spawn_blocking(move || {
            let res = conn.framebuffer_bytes();
            log(&format!("[{task_id}] Joined!! [{}]", Local::now().timestamp_micros() - *START));
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

impl<'a, const POOL_SIZE: usize> Node<'a, POOL_SIZE> {
    async fn send_action(&self, action: &NodeAction<'_>) -> Result<(), Error> {
        let mut conn = self.get_conn().await?;
        let command = action.cmd();
        log(&command);
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
        self.send_action(&action).await?;
        Ok(())
    }
    pub async fn key_up(&self, key: Key) -> Result<(), Error> {
        let action = self.action_man.get_key_up_action(key).await;
        self.send_action(&action).await?;
        Ok(())
    }
    
    pub async fn touch_down(&self, iden: &'a str, pos: Position<u32>) -> Result<(), Error> {
        let action = self.action_man.get_touch_down_action(iden, pos).await;
        self.send_action(&action).await?;
        Ok(())
    }
    
    pub async fn touch_move(&self, iden: &'a str, pos: Position<u32>) -> Result<(), Error> {
        let action = self.action_man.get_touch_move_action(iden, pos).await;
        self.send_action(&action).await?;
        Ok(())
    }
    
    pub async fn touch_up(&self, iden: &'a str) -> Result<(), Error> {
        let action = self.action_man.get_touch_up_action(iden).await;
        self.send_action(&action).await?;
        Ok(())
    }
}

#[tokio::test]
async fn test() -> Result<(), Error> {
    let ctrl: Node<32> = Node::default();
    let res = ctrl.read_frame_buf_bytes().await?;
    println!("{:?}", res.len());
    Ok(())
}

#[test]
fn serial_test() -> Result<(), Error> {
    let mut server = crate::ADB_SERVERS.get_mut(&*crate::ADB_SERVER_DEFAULT_IP).unwrap();
    let mut device = server.get_device_by_name("127.0.0.1:16384")?;
    for _ in 0..32 {
        let res = device.framebuffer_bytes()?;
        println!("{:?}", res.len());
    }

    Ok(())
}

#[tokio::test]
async fn concurrent_test() -> Result<(), Error> {
    let task_n = 64;
    
    let ctrl: Node<16> = Node::default();
    ctrl.connect().await?;
    let mut handles: Vec<tokio::task::JoinHandle<Result<(), Error>>> = Vec::with_capacity(task_n);
    let ctrl = Arc::new(ctrl);
    
    let start = Local::now().timestamp_micros();
    for _ in 0..task_n {
        let ctrl = ctrl.clone();
        handles.push(tokio::spawn(async move {
            let _ = ctrl.read_frame_buf_bytes().await?;
            Ok(())
        }));
    }
    for handle in handles {
       handle.await?;
    }
    
    println!("pool_len: {}", ctrl.pool_len().await);
    println!("pending_cnt: {}", ctrl.pending_cnt.load(Ordering::SeqCst));
    println!("[Perf] {}", Local::now().timestamp_micros() - start);

    Ok(())
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


#[tokio::test]
async fn shell_test() -> Result<(), Error> {
    let mut ctrl: Node<4> = Node::default();
    ctrl.connect().await?;
    
    let mut conn = ctrl.get_conn().await?;
    let task = spawn_blocking(move || {
        conn.shell_command(&["getevent", "/dev/input/event4"],&mut EventParser{})?;
        Ok::<(), Error>(())
    });
    
    let _ = tokio::time::timeout(std::time::Duration::from_secs(10), task).await;
    
    
    // conn.shell(&mut cursor, Box::new(EventParser{}))?;
    Ok(())
}

#[tokio::test]
async fn send_event_test() -> Result<(), Error> {
    let ctrl: Node<4> = Node::default();
    ctrl.connect().await?;
    for _ in 0..100 {
        ctrl.send_action(
            &NodeAction::new(
                vec![NodeEvent::Key(Key::Home, KeyValue::Down)],
                "/dev/input/event4"
            )
        ).await?;
        ctrl.send_action(
            &NodeAction::new(
                vec![NodeEvent::Key(Key::Home, KeyValue::Up)],
                "/dev/input/event4"
            )
        ).await?;
    }
    Ok(())
}

#[tokio::test]
async fn key() -> Result<(), Error> {
    use chrono::Local;
    let ctrl: Node<4> = Node::default();
    ctrl.connect().await?;
    const ROUND: i32 = 100;
    let start = Local::now();
    for _ in 0..ROUND {
        ctrl.key_down(Key::Home).await?;
        ctrl.key_up(Key::Home).await?;
    }
    let end = Local::now() - start;
    println!("{}", end.num_milliseconds() as f64 / ROUND as f64 / 2f64);
    Ok(())
}

#[tokio::test]
async fn touch() -> Result<(), Error> {
    let ctrl: Node<4> = Node::default();
    ctrl.connect().await?;
    let actions = vec![
        ("finger", Position::new(100, 100)),
        ("finger", Position::new(100, 200)),
        ("finger", Position::new(200, 200)),
        ("finger", Position::new(200, 100)),
        ("finger", Position::new(100, 100)),
        ("finger2", Position::new(300, 300)),
        ("finger", Position::new(100, 200)),
        ("finger2", Position::new(300, 400)),
        ("finger", Position::new(200, 200)),
        ("finger2", Position::new(400, 400)),
        ("finger", Position::new(200, 100)),
        ("finger2", Position::new(400, 300)),
        ("finger", Position::new(100, 100)),
        ("finger2", Position::new(300, 300)),
    ];

    for (iden, pos) in actions {
        ctrl.touch_down(iden, pos).await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    };
    
    println!("before: {:?}", ctrl.action_man.status());
    ctrl.touch_up("finger").await?;
    println!("middle: {:?}", ctrl.action_man.status());
    ctrl.touch_up("finger2").await?;
    println!("last: {:?}", ctrl.action_man.status());
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    Ok(())
}

#[tokio::test]
async fn multi_device() -> Result<(), Error> {
    use std::str::FromStr;
    let ctrl_1: Node<4> = Node::default();
    let ctrl_2: Node<4> = Node::new(
        DeviceConnPool::new("127.0.0.1:16416".to_string(), SocketAddrV4::from_str("127.0.0.1:16416").unwrap()),
        "/dev/input/event4"
    );
    ctrl_1.connect().await?;
    ctrl_2.connect().await?;
    
    let task1: tokio::task::JoinHandle<Result<(), Error>> = tokio::spawn(async move {
        for _ in 0..100 {
            ctrl_1.send_action(
                &NodeAction::new(
                    vec![NodeEvent::Key(Key::Home, KeyValue::Down)],
                    "/dev/input/event4"
                )
            ).await?;
            ctrl_1.send_action(
                &NodeAction::new(
                    vec![NodeEvent::Key(Key::Home, KeyValue::Up)],
                    "/dev/input/event4"
                )
            ).await?;
        }
        Ok(())
    });
    
    let task2: tokio::task::JoinHandle<Result<(), Error>> = tokio::spawn(async move {
        for _ in 0..100 {
            ctrl_2.send_action(
                &NodeAction::new(
                    vec![NodeEvent::Key(Key::Home, KeyValue::Down)],
                    "/dev/input/event4"
                )
            ).await?;
            ctrl_2.send_action(
                &NodeAction::new(
                    vec![NodeEvent::Key(Key::Home, KeyValue::Up)],
                    "/dev/input/event4"
                )
            ).await?;
        }
        Ok(())
    });
    
    let _ = tokio::join!(task1, task2);
    
    Ok(())
}
