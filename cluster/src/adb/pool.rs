use std::collections::VecDeque;
use std::net::SocketAddrV4;
use std::str::FromStr;
use std::sync::Arc;
use adb_client::ADBServerDevice;
use log::trace;
use tokio::sync::{Mutex};

use crate::adb::error::Error;

pub struct DeviceConnPool<const MAX_POOL_SIZE: usize> {
    conns: Mutex<VecDeque<Arc<ADBServerDevice>>>,
    device_iden: &'static str,
    server_addr: SocketAddrV4,
    
    // health: PoolHealth,
}

impl<const MAX_POOL_SIZE: usize> Default for DeviceConnPool<MAX_POOL_SIZE> {
    fn default() -> Self {
        let device_iden = "127.0.0.1:16384";
        let server_addr = SocketAddrV4::from_str("127.0.0.1:5037").unwrap();
        DeviceConnPool::new(device_iden, server_addr)
    }
}

impl<const MAX_POOL_SIZE: usize> DeviceConnPool<MAX_POOL_SIZE> {
    pub fn new(device_iden: &'static str, server_addr: SocketAddrV4) -> Self {
        DeviceConnPool {
            conns: Mutex::new(VecDeque::with_capacity(MAX_POOL_SIZE)),
            device_iden,
            server_addr,
        }
    }
    
    pub async fn len(&self) -> usize {
        self.conns.lock().await.len()
    }
    
    pub async fn connect(&self) -> Result<(), Error> {
        let mut conns = self.conns.lock().await;
        if conns.len() >= MAX_POOL_SIZE { return Ok(()) }
        for _ in 0..MAX_POOL_SIZE - conns.len() {
            let conn = self.make_conn().await;
            conns.push_back(conn);
        }
        Ok(())
    }
    
    pub(crate) async fn make_conn(&self) -> Arc<ADBServerDevice> {
        let conns = ADBServerDevice::new(
            self.device_iden.to_string(), Some(self.server_addr));
        Arc::new(conns)
    }
    
    pub async fn try_get(&self) -> Result<Arc<ADBServerDevice>, Error> {
        let mut conns = self.conns.lock().await;
        conns.pop_front().ok_or(Error::PoolBusy(self.server_addr.clone()))
    }

    pub async fn get(&self) -> Result<Arc<ADBServerDevice>, Error> {
        for _ in 0..10 {
            if let Some(device) = self.try_get().await.ok() {
                return Ok(device);
            };
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        Ok(self.make_conn().await)
    }

    pub async fn release(&self, conn: Arc<ADBServerDevice>) {
        let mut conns = self.conns.lock().await;
        if conns.len() < MAX_POOL_SIZE {
            conns.push_back(conn);
            trace!("GIVEN BACK!!");
        }
    }
    
    pub async fn resurrect(&self) -> bool {
        let mut conns = self.conns.lock().await;
        if conns.len() >= MAX_POOL_SIZE { return false }
        
        let conn = self.make_conn().await;
        conns.push_back(conn);
        
        true
    }
}

#[tokio::test]
async fn test() -> Result<(), Error> {
    use adb_client::ADBDeviceExt;
    let pool: DeviceConnPool<32> = DeviceConnPool::default();
    let conn = pool.get().await?;
    let mut conn = Arc::into_inner(conn).unwrap();
    let res = conn.framebuffer_bytes()?;
    println!("{:?}", res.len());
    Ok(())
}
