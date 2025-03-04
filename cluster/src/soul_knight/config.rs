use std::net::SocketAddrV4;
use std::ops::Deref;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::utils::Position;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    name: String,
    addr: SocketAddrV4
}

impl ServerConfig {
    pub fn new(name: &str, addr: SocketAddrV4) -> Self {
        ServerConfig {
            name: name.to_string(),
            addr
        }
    }
    pub fn name(&self) -> &str {
        self.name.as_str()
    }
    pub fn addr(&self) -> SocketAddrV4 {
        self.addr
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeConfigKeyMap {
    joystick:   Position<u32>,
    attack:     Position<u32>,
    skill:      Position<u32>,
    weapon:     Position<u32>,
}

impl NodeConfigKeyMap {
    pub fn get(&self, key: &str) -> Option<Arc<Position<u32>>> {
        match key {
            "joystick"  => Some(Arc::new(self.joystick)),
            "attack"    => Some(Arc::new(self.attack)),
            "skill"     => Some(Arc::new(self.skill)),
            "weapon"    => Some(Arc::new(self.weapon)),
            _ => None
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeConfig<'a> {
    name: String, 
    iden: SocketAddrV4,
    server:     &'a str,
    ev_device:  &'a str,
    resolution: Position<u32>,
    keymap:     NodeConfigKeyMap,
}

impl<'a> Default for NodeConfig<'a> {
    fn default() -> Self {
        NodeConfig {
            name:   "default".to_string(),
            server: "default",
            keymap: NodeConfigKeyMap {
                joystick:   Position::new(100, 100),
                attack:     Position::new(200, 200),
                skill:      Position::new(300, 300),
                weapon:     Position::new(400, 400),
            },
            ev_device:  "/dev/input/event4",
            iden:       SocketAddrV4::new("127.0.0.1".parse().unwrap(), 16384),
            resolution: Position::new(1280, 720),
        }
    }
}
impl<'a> NodeConfig<'a> {
    pub fn name(&self) -> &str {
        self.name.as_str()
    }
    
    pub fn iden(&self) -> SocketAddrV4 {
        self.iden
    }
    
    pub fn resolution(&self) -> Position<u32> {
        self.resolution
    }
    
    pub fn server(&self) -> &'a str {
        self.server
    }
    
    pub fn ev_device(&self) -> &'a str {
        self.ev_device
    }
    
    pub fn keymap_get(&self, key: &str) -> Option<Arc<Position<u32>>> {
        self.keymap.get(key)
    }
}

#[test]
fn test() -> Result<(),()> {
    let str = include_str!("../../configs/server.json");
    let config: Vec<ServerConfig> = serde_json::from_str(str).unwrap();
    println!("{:?}", config);
    
    let str = include_str!("../../configs/node.json");
    let config: Vec<NodeConfig> = serde_json::from_str(str).unwrap();
    println!("{:?}", config);
    Ok(())
}