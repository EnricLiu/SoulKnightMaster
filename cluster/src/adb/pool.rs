use std::collections::VecDeque;
use std::net::SocketAddrV4;
use std::str::FromStr;
use std::sync::Arc;
use adb_client::{ADBServer, ADBServerDevice};
use tokio::sync::{Mutex};

use tokio::sync::{Semaphore, SemaphorePermit};
use crate::adb::error::Error;

pub struct ADBDevicePool<const MAX_POOL_SIZE: usize> {
    conns: Mutex<VecDeque<Arc<ADBServerDevice>>>,
    addr: SocketAddrV4,
}

impl<const MAX_POOL_SIZE: usize> Default for ADBDevicePool<MAX_POOL_SIZE> {
    fn default() -> Self {
        let addr = SocketAddrV4::from_str("127.0.0.1:5037").unwrap();
        ADBDevicePool::new(addr)
    }
}

impl<const MAX_POOL_SIZE: usize> ADBDevicePool<MAX_POOL_SIZE> {
    pub fn new(addr: SocketAddrV4) -> Self {
        ADBDevicePool {
            conns: Mutex::new(VecDeque::with_capacity(MAX_POOL_SIZE)),
            addr,
        }
    }
    
    async fn make_conn(&self) -> Arc<ADBServerDevice> {
        let conns = ADBServerDevice::new(self.addr.to_string(), Some(self.addr));
        Arc::new(conns)
    }
    
    pub async fn try_get(&self) -> Result<Arc<ADBServerDevice>, Error> {
        let mut conns = self.conns.lock().await;
        conns.pop_front().ok_or(Error::PoolBusy(self.addr.clone()))
    }

    pub async fn get(&self) -> Result<Arc<ADBServerDevice>, Error> {
        for _ in 0..3 {
            if let Some(device) = self.try_get().await.ok() {
                return Ok(device);
            };
            tokio::time::sleep(std::time::Duration::from_micros(500)).await;
        }
        Ok(self.make_conn().await)
    }

    pub async fn release(&self, conn: Arc<ADBServerDevice>) {
        let mut conns = self.conns.lock().await;
        if conns.len() < MAX_POOL_SIZE {
            conns.push_back(conn);
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
