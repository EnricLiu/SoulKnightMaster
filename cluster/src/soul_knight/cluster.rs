use std::collections::VecDeque;
use std::fmt::Debug;
use dashmap::DashMap;
use crate::soul_knight::model::NodeSignal;
use super::server::Server;
use super::config::{NodeConfig, ServerConfig};
use super::error::{Error, ServerError};
use crate::node::{Node, NodeStatus};
use crate::soul_knight::{Action, FrameBuffer, NodeError};

pub struct Cluster {
    // name -> Node
    nodes: DashMap<String, Node<12>>,
    // name -> ADBServer
    servers: DashMap<String, Server>,
}

impl Debug for Cluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cluster")
            .field("adb num", &self.nodes.len())
            .field("server num", &self.servers.len())
            .finish()
    }
}

impl Cluster {
    pub fn new(server_configs: Vec<ServerConfig>) -> Self {
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
    
    pub fn all_devices(&self) -> Vec<NodeStatus> {
        let mut ret = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.iter() {
            ret.push(node.get_status());
        }
        ret
    }
    
    pub async fn act_by_name(&self, name: &str, action: Action) -> Result<(), Error> {
        if let Some(node) = self.nodes.get(name) {
            match node.value().act(NodeSignal::Action(action)).await {
                Ok(_) => Ok(()),
                Err(e) => Err(Error::NodeError(e))
            }
        } else {
            Err(Error::NodeNotFound(name.to_string()))
        }
    }
    
    pub async fn new_node(&self, config: NodeConfig) -> Result<String, Error> {
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
        
        let node = Node::new(config, server_addr);
        self.nodes.insert(name.clone(), node);
        
        Ok(name)
    }
    
    pub async fn schedule_node(&self, name: &str) -> Result<(), Error> {
        if let Some(node) = self.nodes.get(name) {
            node.value().schedule().await?;
            Ok(())
        } else {
            Err(Error::NodeNotFound(name.to_string()))
        }
    }

    pub async fn deschedule_node(&self, name: &str) -> Result<(), Error> {
        if let Some(node) = self.nodes.get(name) {
            node.value().deschedule().await?;
            Ok(())
        } else {
            Err(Error::NodeNotFound(name.to_string()))
        }
    }
    
    pub async fn drop_node(&self, name: &str) -> Result<(), Error> {
        if let Some((name, node)) = self.nodes.remove(name) {
            match node.release().await {
                Ok(_) => {},
                Err(NodeError::NodeNotScheduled { ..}) => {},
                Err(e) => {
                    self.nodes.insert(name, node);
                    return Err(Error::NodeError(e))
                }
            }
            Ok(())
        } else {
            Err(Error::NodeNotFound(name.to_string()))
        }
    }
    
    pub async fn get_fb_by_name(&self, name: &str) -> Result<FrameBuffer, Error> {
        if let Some(node) = self.nodes.get(name) {
            let node = node.value();
            Ok(node.get_fb().await)
        } else {
            Err(Error::NodeNotFound(name.to_string()))
        }
    }
    
    pub fn node_len(&self) -> usize {
        self.nodes.len()
    }
    
    
}


#[tokio::test]
async fn test() -> Result<(), Error> {
    use serde_json::from_str;
    let server_configs = from_str(include_str!("../../configs/server.json"))
        .expect("parse server configs error");
    let cluster = Cluster::new(server_configs);
    let mut node_config: VecDeque<NodeConfig> = from_str(include_str!("../../configs/node.json"))
        .expect("parse adb config error");
    
    cluster.new_node(node_config.pop_front().unwrap()).await?;
    
    Ok(())
}