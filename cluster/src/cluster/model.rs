use serde::{Deserialize, Serialize};
use crate::adb::event::{Key, KeyValue};
use crate::utils::Position;

#[derive(Debug, Clone)]
pub enum NodeWatcherSignal {
    Created { node_name: &'static str },
    Ready { node_name: &'static str },
    Halt { node_name: &'static str },
    Dead { node_name: &'static str },

    Error { node_name: &'static str, err: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NodeSignal {
    Action(SoulKnightAction),
    RawAction(RawAction),
    Close,
    Kill
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RawAction {
    Click { pos: Position<u32> },
    Key { key: Key, val: KeyValue },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoulKnightAction {
    sn: u64,
    direction: Option<f64>,
    attack: bool,
    skill:  bool,
    weapon: bool,
}

impl SoulKnightAction {
    pub fn new(sn: u64, dir: Option<f64>, attack: bool, skill: bool, weapon: bool) -> Self {
        SoulKnightAction {
            sn,
            skill,
            attack,
            weapon,
            direction: dir,
        }
    }
    pub fn sn(&self) -> u64 {
        self.sn
    }
    pub fn direction(&self) -> Option<f64> {
        self.direction
    }
    pub fn attack(&self) -> bool {
        self.attack
    }
    pub fn skill(&self) -> bool {
        self.skill
    }
    pub fn weapon(&self) -> bool {
        self.weapon
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameBuffer {
    sn:     u64,
    node:   String,
    fb:     Vec<u8>,
}

impl FrameBuffer {
    pub fn new(node: &str, sn: u64, fb: Vec<u8>) -> Self {
        FrameBuffer {
            node: node.to_string(),
            sn,
            fb,
        }
    }

    pub fn unwrap(self) -> (String, u64, Vec<u8>) {
        (self.node, self.sn, self.fb)
    }
}