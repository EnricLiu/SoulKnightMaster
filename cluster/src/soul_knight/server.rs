use std::net::SocketAddrV4;
use adb_client::{ADBServer, DeviceShort};
use super::config::ServerConfig;
use super::error::{Error, ServerError};

pub struct Server {
    name: String,
    addr: SocketAddrV4,
    server: ADBServer,
}

impl From<ServerConfig> for Server {
    fn from(config: ServerConfig) -> Self {
        Server {
            name: config.name().to_string(),
            addr: config.addr(),
            server: ADBServer::new(config.addr()),
        }
    }
}

impl Server {
    pub fn new(name: &str, addr: SocketAddrV4) -> Self {
        Server {
            addr,
            name: name.to_string(),
            server: ADBServer::new(addr),
        }
    }
    
    pub fn check_node_by_iden(&mut self, iden: &SocketAddrV4) -> Result<(), ServerError> {
        let nb_devices = self.server
            .devices()?
            .into_iter()
            .filter(|d| d.identifier.as_str() == &iden.to_string())
            .collect::<Vec<DeviceShort>>()
            .len();
        if nb_devices == 0 {
            return Err(ServerError::DeviceNotConnected(*iden));
        }
        Ok(())
    }
    
    pub fn addr(&self) -> SocketAddrV4 {
        self.addr
    }
}