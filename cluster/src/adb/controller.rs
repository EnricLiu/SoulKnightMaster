use std::net::SocketAddrV4;
use std::sync::Arc;
use adb_client::{ADBDeviceExt, ADBServerDevice};
use image::{ImageBuffer, Rgba};
use tokio::sync::mpsc::Receiver;
use tokio::sync::Mutex;
use tokio::task::spawn_blocking;
use crate::adb::error::Error;
use crate::adb::pool::ADBDevicePool;
use super::event::AdbEvent;

pub enum AdbSignal {
    Command(Vec<AdbEvent>),
    Raw(String)
}


pub struct Controller {
    pool:   Arc<Mutex<ADBDevicePool<16>>>,
    pub tx: tokio::sync::mpsc::Sender<AdbSignal>,
    pub rx: tokio::sync::mpsc::Receiver<AdbSignal>,
}

impl Controller {
    pub fn new(addr: SocketAddrV4) -> Self {
        let (tx, inner_rx) = tokio::sync::mpsc::channel(64);
        let (inner_tx, rx) = tokio::sync::mpsc::channel(64);
        let ret = Controller {
            pool: Arc::new(Mutex::new(ADBDevicePool::new(addr))),
            tx, rx
        };
        // ret.task(inner_rx);
        ret
    }

    async fn read_frame_buf(&self) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, Error> {
        let conn = self.pool.lock().await.get().await?;
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
async fn test() -> {
    
}