use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub enum NodeStatusCode {
    IDLE,
    DEAD,
    RUNNING,
    STOPPED,
    DISCONNECTED,
}

impl From<u8> for NodeStatusCode {
    fn from(code: u8) -> Self {
        match code {
            0 => NodeStatusCode::IDLE,
            1 => NodeStatusCode::DEAD,
            2 => NodeStatusCode::RUNNING,
            3 => NodeStatusCode::STOPPED,
            _ => NodeStatusCode::DISCONNECTED,
        }
    }
}

#[derive(Debug)]
pub struct AtomicNodeStatus {
    name:           &'static str,
    status:         AtomicU8,
    total_thread:   AtomicU64,
    used_thread:    AtomicU64,
}

impl AtomicNodeStatus {
    pub fn new(name: &'static str, total: usize) -> Self {
        AtomicNodeStatus {
            name,
            status:         AtomicU8::new(NodeStatusCode::DISCONNECTED as u8),
            used_thread:    AtomicU64::new(0),
            total_thread:   AtomicU64::new(total as u64),
        }
    }
    
    pub fn set_status(&self, status: NodeStatusCode) {
        self.status.store(status as u8, Ordering::SeqCst);
    }
    
    pub fn inc_thread(&self) {
        self.used_thread.fetch_add(1, Ordering::SeqCst);
    }
    
    pub fn dec_thread(&self) {
        self.used_thread.fetch_sub(1, Ordering::SeqCst);
    }
    
    pub fn snap(&self) -> NodeStatus {
        NodeStatus {
            name:           self.name,
            status:         self.status.load(Ordering::SeqCst).into(),
            used_thread:    self.used_thread.load(Ordering::SeqCst) as usize,
            total_thread:   self.total_thread.load(Ordering::SeqCst) as usize,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeStatus {
    name:           &'static str,
    status:         NodeStatusCode,
    used_thread:    usize,
    total_thread:   usize,
}

impl From<AtomicNodeStatus> for NodeStatus {
    fn from(atomic: AtomicNodeStatus) -> Self {
        NodeStatus {
            name:   atomic.name,
            status: atomic.status.load(Ordering::SeqCst).into(),
            used_thread:    atomic.used_thread.load(Ordering::SeqCst) as usize,
            total_thread:   atomic.total_thread.load(Ordering::SeqCst) as usize,
        }
    }
}
