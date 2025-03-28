use std::net::SocketAddrV4;
use adb_client::{ADBServer, DeviceShort, RustADBError};
use crate::cluster::ServerConfig;
use crate::cluster::{Error, ServerError};

pub struct Server {
    name: &'static str,
    addr: SocketAddrV4,
    server: ADBServer,
}

impl From<ServerConfig> for Server {
    fn from(config: ServerConfig) -> Self {
        Server {
            name: config.name(),
            addr: config.addr(),
            server: ADBServer::new(config.addr()),
        }
    }
}

impl Server {
    pub fn new(name: &'static str, addr: SocketAddrV4) -> Self {
        Server {
            addr,
            name,
            server: ADBServer::new(addr),
        }
    }
    
    pub fn connect_node_by_addr(&mut self, iden: &str) -> Result<(), ServerError> {
        println!("Server[{}]: connect node by iden[{}]", self.name, iden);
        if let Err(e) = self.check_node_by_iden(iden) {
            if let ServerError::DeviceNotConnected(_) = e {
                self.server.connect_device(iden.parse().unwrap())?;

                if let Err(e) = self.server.connect_device(iden.parse().unwrap()) {
                    return if let RustADBError::ADBRequestFailed(err) = e {
                        if err == format!("already connected to {}", iden) {
                            Ok(())
                        } else {
                            Err(RustADBError::ADBRequestFailed(err).into())
                        }
                    } else {
                        Err(e.into())
                    }
                }
            }
        }
        Ok(())
    }
    
    pub fn check_node_by_iden(&mut self, iden: &str) -> Result<(), ServerError> {
        let nb_devices = self.server
            .devices()?
            .into_iter()
            .filter(|d| d.identifier.as_str() == iden)
            .collect::<Vec<DeviceShort>>()
            .len();
        if nb_devices == 0 {
            return Err(ServerError::DeviceNotConnected(iden.to_string()));
        }
        Ok(())
    }
    
    pub fn addr(&self) -> SocketAddrV4 {
        self.addr
    }
}