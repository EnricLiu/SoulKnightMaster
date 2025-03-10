use crate::adb::error::Error as NodeErr;
use std::net::SocketAddrV4;
use thiserror::Error as ErrorTrait;
use tokio::sync::mpsc::error::SendError;
use crate::soul_knight::model::Action;
use crate::soul_knight::model::NodeTickerSignal;

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
    #[error("Node[{node_name}] Already Scheduled")]
    NodeAlreadyScheduled{ node_name: &'static str },
    #[error("Timeout fetching fb from Node[{node_name}], sn: {sn}")]
    ThreadErrorFbTimeout{ node_name: &'static str, sn: u64 },
    #[error("Failed to forward Tick to Node[{node_name}]: {err}")]
    ThreadErrorSend{ node_name: &'static str, err: SendError<NodeTickerSignal> },
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