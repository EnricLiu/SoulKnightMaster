use std::collections::VecDeque;
use std::fmt::Debug;
use axum::http::Response;
use dashmap::DashMap;
use log::{debug, error, info};
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use crate::cluster::model::NodeSignal;
use crate::adb::server::Server;
use super::config::{NodeConfig, ServerConfig};
use super::error::{Error, ServerError};
use crate::node::{Node, NodeStatus};
use crate::cluster::{SoulKnightAction, FrameBuffer, NodeError, NodeWatcherSignal};
use crate::utils::Position;

pub struct Cluster {
    // name -> Node
    nodes: DashMap<String, Node<12>>,
    // name -> ADBServer
    servers: DashMap<String, Server>,
    loggers: DashMap<String, JoinHandle<Result<(), Error>>>
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
            loggers: DashMap::new()
        };
        
        for config in server_configs {
            let name = config.name().to_string();
            let server = Server::from(config);
            ret.servers.insert(name, server);
        };
        
        ret
    }
    
    pub fn get_status_by_name(&self, name: &str) -> Result<NodeStatus, Error> {
        self.nodes.get(name)
            .map(|node| node.value().get_status())
            .ok_or(Error::NodeNotFound(name.to_string()))
    }
    
    pub fn all_devices(&self) -> Vec<NodeStatus> {
        let mut ret = Vec::with_capacity(self.nodes.len());
        for node in self.nodes.iter() {
            ret.push(node.get_status());
        }
        ret
    }
    
    pub async fn act_by_name(&self, name: &str, action: NodeSignal) -> Result<(), Error> {
        if let Some(node) = self.nodes.get(name) {
            debug!("Node[{name}]: {:?}", &action);
            match node.value().act(action).await {
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
                let _ = server.connect_node_by_addr(&iden)?;
                Ok(server.addr())
            } else {
                Err(ServerError::ServerNotFound(server_name.to_string()))
            }?;
        
        let name = config.name();
        let name_string = name.to_string();
        if self.nodes.get(name).is_some() {
            return Err(Error::NodeAlreadyExist(name_string));
        }
        
        let node = Node::new(config, server_addr);
        self.loggers.insert(name_string.clone(), self.make_logger(name, node.watch()));


        self.nodes.insert(name_string.clone(), node);
        Ok(name_string)
    }
    
    fn make_logger(
        &self, name: &'static str, mut logger: broadcast::Receiver<NodeWatcherSignal>
    ) -> JoinHandle<Result<(), Error>> {
        let handle = tokio::spawn(async move {
            while let Ok(signal) = logger.recv().await {
                match signal {
                    NodeWatcherSignal::Error {node_name, err } => {
                        error!("Node[{node_name}]: {err}");
                    }
                    NodeWatcherSignal::Ready {node_name} => {
                        info!("Node[{node_name}] Ready.")
                    }
                    _ => {}
                }
            }
            error!("Node[{name}] logger down.");
            Err(Error::NodeError(NodeError::NodeLoggerDown{ name }))
        });
        handle
    }
    
    pub async fn schedule_node(&self, name: &str) -> Result<(), Error> {
        if let Some(node) = self.nodes.get(name) {
            let node = node.value();
            if let Some(mut server) = self.servers.get_mut(node.get_server_name()) {
                server.value_mut().connect_node_by_addr(node.get_iden())?;
                node.schedule().await?;
                info!("Node[{name}] scheduled.");
                Ok(())
            } else {
                Err(Error::ServerNotFound(node.get_server_name().to_string()))
            }
            
        } else {
            Err(Error::NodeNotFound(name.to_string()))
        }
    }

    pub async fn deschedule_node(&self, name: &str) -> Result<(), Error> {
        if let Some(node) = self.nodes.get(name) {
            node.value().deschedule().await?;
            info!("Node[{name}] descheduled.");
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