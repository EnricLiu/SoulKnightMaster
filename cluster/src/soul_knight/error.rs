use crate::adb::error::Error as NodeErr;
use std::net::SocketAddrV4;
use thiserror::Error as ErrorTrait;
use tokio::sync::{mpsc, watch};
use crate::soul_knight::FrameBuffer;
use crate::soul_knight::model::Action;
use crate::soul_knight::model::NodeSignal;

#[derive(Debug, ErrorTrait)]
pub enum Error {
    #[error("Node Error: {}", .0)]
    NodeError(#[from] NodeError),
    
    #[error("adb {0} not found")]
    NodeNotFound(String),
    
    #[error("adb {0} already exists")]
    NodeAlreadyExist(String),
    
    #[error("ServerError: {0}")]
    ServerErr(#[from] ServerError),
    
    #[error("{0}")]
    Custom(String),
}

#[derive(Debug, ErrorTrait)]
pub enum NodeError {
    #[error("{0}")]
    NodeErr(#[from] NodeErr),
    #[error("Node[{name}] Already Scheduled")]
    NodeAlreadyScheduled{ name: &'static str },

    #[error("Node[{name}] Failed to act: {action}")]
    ActionFailed{ name: &'static str, action: &'static str },
    
    #[error("Node[{name}] Deschedule Timeout")]
    DescheduleTimeout{ name: &'static str },
    
    #[error("Fb sn provided({0}) elder than current({1})")]
    FbSnCorrupt(u64, u64),
    
    #[error("Timeout fetching fb from Node[{name}], sn: {sn}")]
    FbTimeout{ name: &'static str, sn: u64 },
    
    #[error("Failed to forward Action to Node[{name}]: {err}")]
    SendErrorAction { name: &'static str, err: mpsc::error::SendError<NodeSignal> },

    #[error("Failed to forward Action to Node[{name}]: {err}")]
    SendErrorFb { name: &'static str, err: watch::error::SendError<FrameBuffer> },
    
    
    #[error("{0}")]
    Custom(String),
}

#[derive(Debug, ErrorTrait)]
pub enum ServerError {
    #[error("server {0} not found")]
    ServerNotFound(String),
    #[error("device {0} not connected")]
    DeviceNotConnected(String),
    #[error("{0}")]
    AdbError(#[from] adb_client::RustADBError),
    #[error("{0}")]
    Custom(String),
}