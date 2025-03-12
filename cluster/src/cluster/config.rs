use std::net::SocketAddrV4;
use serde::{Deserialize, Serialize};
use crate::utils::Position;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    name: &'static str,
    addr: SocketAddrV4
}

impl ServerConfig {
    pub fn new(name: &'static str, addr: SocketAddrV4) -> Self {
        ServerConfig {
            name,
            addr
        }
    }
    pub fn name(&self) -> &'static str {
        self.name
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
    pub fn keys() -> Vec<&'static str> {
        vec!["joystick", "attack", "skill", "weapon"]
    }
    pub fn get(&self, key: &str) -> Option<Position<u32>> {
        match key {
            "joystick"  => Some(self.joystick),
            "attack"    => Some(self.attack),
            "skill"     => Some(self.skill),
            "weapon"    => Some(self.weapon),
            _ => None
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeConfig {
    name: &'static str,
    iden: &'static str,
    server:     &'static str,
    ev_device:  &'static str,
    resolution: Position<u32>,
    keymap:     NodeConfigKeyMap,
}

impl Default for NodeConfig {
    fn default() -> Self {
        NodeConfig {
            name:   "default",
            server: "default",
            keymap: NodeConfigKeyMap {
                joystick:   Position::new(100, 100),
                attack:     Position::new(200, 200),
                skill:      Position::new(300, 300),
                weapon:     Position::new(400, 400),
            },
            ev_device:  "/dev/input/event4",
            iden:       "127.0.0.1:16384",
            resolution: Position::new(1280, 720),
        }
    }
}
impl NodeConfig {
    pub fn name(&self) -> &'static str {
        self.name
    }
    
    pub fn iden(&self) -> &'static str {
        self.iden
    }
    
    pub fn resolution(&self) -> Position<u32> {
        self.resolution
    }
    
    pub fn server(&self) -> &'static str {
        self.server
    }
    
    pub fn ev_device(&self) -> &'static str {
        self.ev_device
    }
    
    pub fn keymap_get(&self, key: &str) -> Position<u32> {
        self.keymap.get(key).expect(&format!("KeyMap: Key[{key}] not found, Please check config."))
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