use adb_client::ADBServer;
use dashmap::DashMap;
use super::server::Server;
use super::config::{NodeConfig, ServerConfig};
use super::error::{Error, ServerError};
use super::node::Node;

pub struct Cluster<'a> {
    // name -> Node
    nodes: DashMap<String, Node<'a, 16>>,
    // name -> ADBServer
    servers: DashMap<String, Server>,
}

impl<'a> Cluster<'a> {
    fn new(server_configs: Vec<ServerConfig>) -> Self {
        let ret = Cluster {
            nodes: DashMap::new(),
            servers: DashMap::new(),
        };
        
        for config in server_configs {
            let name = config.name().to_string();
            let server = Server::from(config);
            ret.servers.insert(name, server);
        };
        
        ret
    }
    
    fn new_node(&self, config: NodeConfig<'a>) -> Result<String, Error> {
        let iden = config.iden();
        let server_name = config.server();
        let server_addr
            = if let Some(mut server) = self.servers.get_mut(server_name) {
                let _ = server.check_node_by_iden(&iden)?;
                Ok(server.addr())
            } else {
                Err(ServerError::ServerNotFound(server_name.to_string()))
            }?;
        
        let name = config.name().to_string();
        if self.nodes.get(&name).is_some() {
            return Err(Error::NodeAlreadyExist(name));
        }
        
        self.nodes.insert(name.clone(), Node::new(config, server_addr));
        
        Ok(name)
    }
    
    fn drop_node(&self, name: &str) -> Result<(), Error> {
        if self.nodes.get(name).is_none() {
            return Err(Error::NodeNotFound(name.to_string()));
        };
        self.nodes.remove(name);
        Ok(())
    }
    
    fn node_len(&self) -> usize {
        self.nodes.len()
    }
    
}

impl Cluster<'_> {
    async fn joystick(&self, direction: Option<f64>) -> Result<(), crate::node::error::Error> {
        todo!()
        // let tasks = Vec::with_capacity(self.node_len());
        // for node in self.nodes.iter() {
        //     tasks.push(tokio::spawn(async move {
        //         node.joystick(direction).await
        //     }));
        // }
    }

    async fn fire(&self) -> Result<bool, crate::node::error::Error> {
        todo!()
    }

    async fn cease(&self) -> Result<bool, crate::node::error::Error> {
        todo!()
    }

    async fn skill(&self) -> Result<(), crate::node::error::Error> {
        todo!()
    }

    async fn weapon(&self) -> Result<(), crate::node::error::Error> {
        todo!()
    }
}