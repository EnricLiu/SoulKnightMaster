use std::net::SocketAddrV4;
use thiserror::Error as ErrorTrait;

#[derive(Debug, ErrorTrait)]
pub enum Error {
    #[error("node {0} not found")]
    NodeNotFound(String),
    
    #[error("node {0} already exists")]
    NodeAlreadyExist(String),
    
    #[error("ServerError: {0}")]
    ServerErr(#[from] ServerError),
    
    #[error("{0}")]
    Custom(String),
}

#[derive(Debug, ErrorTrait)]
pub enum ServerError {
    #[error("server {0} not found")]
    ServerNotFound(String),
    #[error("device {0} not connected")]
    DeviceNotConnected(SocketAddrV4),
    #[error("{0}")]
    AdbError(#[from] adb_client::RustADBError),
    #[error("{0}")]
    Custom(String),
}