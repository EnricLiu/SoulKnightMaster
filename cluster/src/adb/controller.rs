use std::sync::{Arc, LazyLock};
use std::sync::atomic::AtomicUsize;
use adb_client::{ADBDeviceExt, ADBServerDevice};
use image::{ImageBuffer, Rgba};
use tokio::sync::Mutex;
use tokio::task::spawn_blocking;
use crate::adb::error::Error;
use crate::adb::pool::ADBDevicePool;

use super::event::AdbEvent;

use chrono::Local;
use crate::utils::get_id;

pub enum AdbSignal {
    Command(Vec<AdbEvent>),
    Raw(String)
}

static START: LazyLock<i64> = LazyLock::new(|| {
    Local::now().timestamp_micros()
});


pub struct Controller<const POOL_SIZE: usize> {
    pool:   Arc<Mutex<ADBDevicePool<POOL_SIZE>>>,
    pub tx: tokio::sync::mpsc::Sender<AdbSignal>,
    pub rx: tokio::sync::mpsc::Receiver<AdbSignal>,
    pending_cnt: AtomicUsize,
}

impl<const POOL_SIZE: usize> Default for Controller<POOL_SIZE> {
    fn default() -> Self {
        // let iden = "127.0.0.1:16384".to_string();
        // let addr = SocketAddrV4::from_str("127.0.0.1:5037").unwrap();
        // let pool: ADBDevicePool<POOL_SIZE> = ADBDevicePool::new(iden, addr);
        let pool = ADBDevicePool::default();
        Controller::new(pool)
    }
}

impl<const POOL_SIZE: usize> Controller<POOL_SIZE> {
    pub fn new(pool: ADBDevicePool<POOL_SIZE>) -> Self {
        let (tx, inner_rx) = tokio::sync::mpsc::channel(64);
        let (inner_tx, rx) = tokio::sync::mpsc::channel(64);
        let ret = Controller {
            pool: Arc::new(Mutex::new(pool)),
            pending_cnt: AtomicUsize::new(0),
            tx, rx,
        };
        // ret.task(inner_rx);
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
            self.pending_cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        
        // concurrency limitation
        // default: waiting for 10*100ms, every pending req before contributes 1*100ms more.
        let poll_epoch = 10 + 1 * self.pending_cnt.load(std::sync::atomic::Ordering::SeqCst);
        let poll_epoch = poll_epoch.min(prior);
        
        for _ in 0..poll_epoch {
            let pool = self.pool.lock().await;
            if let Some(device) = pool.try_get().await.ok() {
                self.pending_cnt.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                return Ok(Arc::try_unwrap(device).or(Err(Error::ArcFailedUnwrap()))?);
            };
            drop(pool);
            println!("Waiting");
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        
        self.pending_cnt.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        Ok(Arc::try_unwrap(self.pool.lock().await.make_conn().await).or(Err(Error::ArcFailedUnwrap()))?)
    }

    pub async fn get_conn(&self) -> Result<ADBServerDevice, Error> {
        self.get_conn_with_prior(usize::MAX).await
    }
    
    
    async fn read_frame_buf(&self) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, Error> {
        let locked_pool = self.pool.lock().await;
        let conn = locked_pool.get().await?;
        drop(locked_pool);
        let mut conn = Arc::try_unwrap(conn).or(Err(Error::ArcFailedUnwrap()))?;
        let pool = self.pool.clone();

        let task = spawn_blocking(move || {
            let res = conn.framebuffer_inner();
            (res, conn)
        });

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
        println!("[{task_id}] Spawn!!  [{}]", Local::now().timestamp_micros() - *START);
        
        let task = spawn_blocking(move || {
            let res = conn.framebuffer_bytes();
            println!("[{task_id}] Joined!! [{}]", Local::now().timestamp_micros() - *START);
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

    // async fn ev_tx_task(&self) -> () {
    //     loop {
    //         let mut conn = self.conn.lock().await;
    //         let event = conn.event().await;
    //         match event {
    //             Ok(event) => {
    //                 let _ = self.tx.send(AdbSignal::Raw(event)).await;
    //             },
    //             Err(err) => {
    //                 println!("{}", err);
    //             }
    //         }
    //     }
    // }
    // 
    // async fn ev_rx_task(&self) -> () {
    //     let
    // }

    // async fn task(&self, mut rx: Receiver<AdbSignal>) -> Self {
    //     tokio::spawn(async move {
    //         while let Some(event) = rx.recv().await {
    //             let mut conn = self.conn.lock().await;
    //             match event {
    //                 AdbSignal::Command(cmd) => {
    //                     let cmd = cmd.into_iter()
    //                         .map(|x| x.to_command()).collect::<Vec<_>>()
    //                         .join(" ");
    //                     let _ = conn.shell_command(cmd).await;
    //                 },
    //                 AdbSignal::Raw(cmd) => {
    //                     let _ = conn.shell(cmd).await;
    //                 }
    //             }
    //         }
    //     })
    // }
}

#[tokio::test]
async fn test() -> Result<(), Error> {
    let ctrl: Controller<32> = Controller::default();
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
    let task_n = 32;
    
    let ctrl: Controller<16> = Controller::default();
    ctrl.connect().await?;
    let mut handles = Vec::with_capacity(task_n);
    let ctrl = Arc::new(ctrl);
    
    let start = Local::now().timestamp_micros();
    for _ in 0..task_n {
        let ctrl = ctrl.clone();
        handles.push(tokio::spawn(async move {
            let _ = ctrl.read_frame_buf_bytes().await.unwrap();
        }));
    }
    for handle in handles {
       handle.await?;
    }
    
    println!("pool_len: {}", ctrl.pool_len().await);
    println!("pending_cnt: {}", ctrl.pending_cnt.load(std::sync::atomic::Ordering::SeqCst));
    println!("[Perf] {}", Local::now().timestamp_micros() - start);

    Ok(())
}