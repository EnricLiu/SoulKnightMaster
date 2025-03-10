use std::sync::atomic::AtomicU8;
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub enum NodeStatusCode {
    IDLE,
    DEAD,
    RUNNING,
    STOPPED,
    DISCONNECTED,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeStatus {
    name:           &'static str,
    status:         NodeStatusCode,
    used_thread:    usize,
    total_thread:   usize,
}

impl NodeStatus {
    pub fn new(name: &'static str, total: usize) -> Self {
        NodeStatus {
            name,
            status:         NodeStatusCode::DISCONNECTED,
            used_thread:    0,
            total_thread:   total,
        }
    }
}
