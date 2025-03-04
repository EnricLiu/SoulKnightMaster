use std::net::SocketAddrV4;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use image::{ImageBuffer, Rgba};
use crate::node::error::Error;
use crate::node::Node as InnerNode;
use crate::node::pool::DeviceConnPool;
use crate::soul_knight::action::Action;
use crate::soul_knight::config::NodeConfig;
use crate::utils::Position;

pub struct NodeState {
    direction:  AtomicU64,
    movement:   AtomicBool,
    attack:     AtomicBool,
}

impl NodeState {
    pub fn new() -> Self {
        NodeState {
            direction:  AtomicU64::new(0),
            movement:   AtomicBool::new(false),
            attack:     AtomicBool::new(false),
        }
    }

    pub fn is_moving(&self) -> bool {
        self.movement.load(Ordering::SeqCst)
    }

    pub fn is_attacking(&self) -> bool {
        self.attack.load(Ordering::SeqCst)
    }
    
    pub fn angle(&self) -> Option<f64> {
        if !self.is_moving() {
            return None
        }
        let u = self.direction.load(Ordering::SeqCst);
        Some(f64::from_bits(u))
    }
    
    pub fn set_move(&self, angle: Option<f64>) -> () {
        match angle {
            None => {
                self.movement.store(false, Ordering::SeqCst);
                // self.direction.store(0, Ordering::SeqCst);
            },
            Some(angle) => {
                self.direction.store(angle.to_bits(), Ordering::SeqCst);
            }
        }
    }
    
    pub fn set_attack(&self, attack: bool) -> () {
        self.attack.store(attack, Ordering::SeqCst);
    }
}


pub struct Node<'a, const POOL_SIZE: usize> {
    node:   Arc<InnerNode<'a, POOL_SIZE>>,
    states: Arc<NodeState>,
    config: Arc<NodeConfig<'a>>,
}


impl<'a, const POOL_SIZE: usize> Node<'a, POOL_SIZE> {
    pub fn new(config: NodeConfig<'a>, server_addr: SocketAddrV4) -> Self {
        let iden = config.iden();
        let ev_device = config.ev_device();
        Node {
            node: Arc::new(InnerNode::new(DeviceConnPool::new(iden.to_string(), server_addr), ev_device)),
            states: Arc::new(NodeState::new()),
            config: Arc::new(config),
        }
    }
    
    async fn get_fb(&self) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, Error> {
        Ok(self.node.read_frame_buf().await?)
    }
}

impl<'a, const POOL_SIZE: usize> Node<'a, POOL_SIZE> {
    // async fn act(&'a self, action: Action) -> Result<(), Error> {
    //     let joystick_task = self.joystick(action.direction());
    //     let attack_task = self.attack(action.attack());
    //     let skill_down_task = self.skill(true);
    //     let weapon_down_task = self.weapon(true);
    //     
    //     let skill_task: tokio::task::JoinHandle<Result<(), Error>> = tokio::spawn(async move {
    //         if !action.skill() {
    //             return Ok(())
    //         }
    //         skill_down_task.await?;
    //         weapon_down_task.await?;
    //         
    //         Ok(())
    //     });
    //     
    //     Ok(())
    // }
    
    // time independent action
    // async fn joystick(
    //     node: Arc<InnerNode<'a, POOL_SIZE>>,
    //     config: Arc<NodeConfig<'a>>,
    //     direction: Option<f64>
    // ) -> Result<(), Error> {
    //     match direction {
    //         Some(direction) => {
    //             let center
    //                 = config.keymap_get("joystick")
    //                 .unwrap_or(Position::default());
    //             let distance = 200f64;
    //             let target = Position::new(
    //                 center.x + (distance / direction.cos()) as u32,
    //                 center.y + (distance / direction.sin()) as u32,
    //             );
    //             node.touch_down("joystick", target).await?
    //         },
    //         None => {
    //             node.touch_up("joystick").await?
    //         }
    //     };
    //     Ok(())
    // }
    // 
    // async fn attack(&self, attack: bool) -> Result<bool, Error> {
    //     let states = self.states.clone();
    //     let config = self.config.clone();
    //     let pos = config.keymap_get("attack")
    //         .ok_or(Error::Custom("attack keymap not found".to_string()))?;
    //     let node = self.node.clone();
    //     
    //     tokio::spawn(async move {
    //         if !(states.is_attacking() ^ attack) {
    //             return Ok(false)
    //         }
    //         match attack {
    //             true => {
    //                 node.touch_down("attack", *pos).await?;
    //             },
    //             false => {
    //                 node.touch_up("attack").await?;
    //             }
    //         }
    //         states.set_attack(attack);
    //         Ok(true)
    //     }).await?
    // }
    // 
    // async fn skill(&self, skill: bool) -> Result<(), Error> {
    //     let pos = self.config.keymap_get("skill")
    //         .ok_or(Error::Custom("skill keymap not found".to_string()))?;
    //     match skill {
    //         true => {
    //             self.node.touch_down("skill", pos).await?;
    //         },
    //         false => {
    //             self.node.touch_up("skill").await?;
    //         }
    //     }
    //     Ok(())
    // }
    // 
    // async fn weapon(&self, weapon: bool) -> Result<(), Error> {
    //     let pos = self.config.keymap_get("weapon")
    //         .ok_or(Error::Custom("weapon keymap not found".to_string()))?;
    //     match weapon {
    //         true => {
    //             self.node.touch_down("weapon", pos).await?;
    //         },
    //         false => {
    //             self.node.touch_up("weapon").await?;
    //         }
    //     }
    //     Ok(())
    // }
}

// #[tokio::test]
// async fn test() -> Result<(), Error> {
//     const FPS: f64 = 10.0;
//     let sleep_duration = std::time::Duration::from_millis((1000.0 / FPS) as u64);
//     let mut interval = tokio::time::interval(sleep_duration);
//
//     let is_busy = AtomicBool::new(false);
//     loop {
//         if is_busy.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
//             tokio::spawn(async move {
//                 perform_frame_task().await;
//                 is_busy.store(false, Ordering::Relaxed);
//             });
//         }
//         interval.tick().await;
//     }
// }