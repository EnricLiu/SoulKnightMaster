use std::net::SocketAddrV4;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use image::{ImageBuffer, Rgba};
use tokio::task::JoinHandle;
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
    
    pub fn set_move(&self, angle: Option<f64>) {
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
    
    pub fn set_attack(&self, attack: bool) {
        self.attack.store(attack, Ordering::SeqCst);
    }
}


pub struct Node<'a, const POOL_SIZE: usize> {
    node:   Arc<InnerNode<POOL_SIZE>>,
    states: Arc<NodeState>,
    config: Arc<NodeConfig<'a>>,
    ticker_rx: tokio::sync::mpsc::Receiver<Action>,
    ticker_tx: tokio::sync::mpsc::Sender<Action>,
}


impl<'a, const POOL_SIZE: usize> Node<'a, POOL_SIZE> {
    pub fn new(config: NodeConfig<'a>, server_addr: SocketAddrV4) -> Self {
        let iden = config.iden();
        let ev_device = config.ev_device();
        let (ticker_tx, ticker_rx) = tokio::sync::mpsc::channel(1);
        Node {
            node: Arc::new(InnerNode::new(DeviceConnPool::new(iden.to_string(), server_addr), ev_device)),
            states: Arc::new(NodeState::new()),
            config: Arc::new(config),
            ticker_rx, ticker_tx,
        }
    }
    
    async fn get_fb(&self) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, Error> {
        self.node.read_frame_buf().await
    }
    
    pub async fn schedule(&self) -> Result<(), Error> {
        loop {
            todo!()
        }
    }
}

type NodeResultHandle<T> = Result<JoinHandle<Result<T, Error>>, Error>;

impl<const POOL_SIZE: usize> Node<'_, POOL_SIZE> {
    fn tick(&self, action: Action) -> Result<(), Error> {
        todo!()
    }
    
    async fn act(&self, action: Action) -> Result<(), Error> {
        let mut tasks = Vec::with_capacity(8);
        // movement
        tasks.push(self.joystick(action.direction())?);
        // attack
        tasks.push(self.attack(action.attack())?);
        // skill
        if action.skill() {
            tasks.push(self.click("skill")?);
        }
        if action.weapon() {
            tasks.push(self.click("weapon")?);
       }
        
        for task in tasks {
            let _ = task.await?;
        }
        
        Ok(())
        
    }
    
    fn click(&self, key: &str) -> NodeResultHandle<()> {
        let key = key.to_string();
        let node = self.node.clone();
        let pos = self.config.keymap_get(&key)
            .ok_or(Error::Custom(format!("keymap {key} not found")))?;
        
        let handle = tokio::spawn(async move {
            node.touch_down(&key, pos).await?;
            node.touch_up(&key).await?;
            Ok(())
        });
        
       Ok(handle)
    }
    
    
    fn joystick(&self, direction: Option<f64>) -> NodeResultHandle<()> {
        let node = self.node.clone();
        
        let handle = match direction {
            Some(direction) => {
                let center
                    = self.config.keymap_get("joystick")
                    .unwrap_or(Position::default());
                let distance = 200f64;
                let target = Position::new(
                    center.x + (distance / direction.cos()) as u32,
                    center.y + (distance / direction.sin()) as u32,
                );
                tokio::spawn(async move {
                    node.touch_down("joystick", target).await?;
                    Ok(())
                })
            },
            None => {
                tokio::spawn(async move {
                    node.touch_up("joystick").await?;
                    Ok(())
                })
            }
        };
        
        Ok(handle)
    }
    
    fn attack(&self, attack: bool) -> NodeResultHandle<()> {
        let states = self.states.clone();
        let node = self.node.clone();
        let pos = self.config.keymap_get("attack")
            .ok_or(Error::Custom("attack keymap not found".to_string()))?;
        
        let handle = tokio::spawn(async move {
            if !(states.is_attacking() ^ attack) {
                return Ok(())
            }
            match attack {
                true => {
                    node.touch_down("attack", pos).await?;
                },
                false => {
                    node.touch_up("attack").await?;
                }
            }

            states.set_attack(attack);
            Ok(())
        });
        
        Ok(handle)
    }

    fn skill(&self, skill: bool) -> NodeResultHandle<()> {
        let node = self.node.clone();
        let pos = self.config.keymap_get("skill")
            .ok_or(Error::Custom("skill keymap not found".to_string()))?;
        
        let handle = tokio::spawn(async move {
            match skill {
                true => {
                    node.touch_down("skill", pos).await?;
                },
                false => {
                    node.touch_up("skill").await?;
                }
            }
            Ok(())
        });
        Ok(handle)
    }
    
    fn weapon(&self, weapon: bool) -> NodeResultHandle<()> {
        let node = self.node.clone();
        let pos = self.config.keymap_get("weapon")
            .ok_or(Error::Custom("weapon keymap not found".to_string()))?;

        let handle = tokio::spawn(async move {
            match weapon {
                true => {
                    node.touch_down("weapon", pos).await?;
                },
                false => {
                    node.touch_up("weapon").await?;
                }
            }
            Ok(())
        });
        Ok(handle)
    }
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