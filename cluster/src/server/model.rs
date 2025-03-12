use serde::Deserialize;
use crate::cluster::{NodeConfig, NodeSignal};
use crate::utils::Position;

#[derive(Debug, Deserialize)]
pub struct GetFbParams {
    node: String,
}
impl GetFbParams {
    pub fn node(&self) -> &str {
        &self.node
    }
}

// # [derive(Debug, Deserialize)]
// pub struct NewNodeParam {
//     config: NodeConfig
// }
// impl NewNodeParam {
//     pub fn config(self) -> NodeConfig {
//         self.config
//     }
// }

// # [derive(Debug, Deserialize)]
// pub struct DropNodeParam {
//     node: String
// }
// impl DropNodeParam {
//     pub fn node(&self) -> &str {
//         &self.node
//     }
// }

# [derive(Debug, Deserialize)]
pub struct ScheduleParams {
    node: String,
}
impl ScheduleParams {
    pub fn node(&self) -> &str {
        &self.node
    }
}

#[derive(Debug, Deserialize)]
pub struct PostActionParam {
    node:   String,
    action: NodeSignal,
}

impl PostActionParam {
    pub fn node(&self) -> &str {
        &self.node
    }
    pub fn action(&self) -> NodeSignal {
        self.action.clone()
    }
}