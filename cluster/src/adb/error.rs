use std::net::SocketAddrV4;
use adb_client::RustADBError;
use thiserror::Error;
use tokio::task::JoinError;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    AdbError(#[from]RustADBError),
    
    #[error("Failed to join task: {0}")]
    JoinTask(#[from]JoinError),
    
    #[error("PoolBusy: no service available: {0}")]
    PoolBusy(SocketAddrV4),

    #[error("Failed to unwrap Arc")]
    ArcFailedUnwrap(),

    #[error("{0}")]
    Custom(String)
}