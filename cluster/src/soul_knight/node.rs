use std::net::SocketAddrV4;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use image::{ImageBuffer, Rgba};
use tokio::sync::{RwLock, mpsc, watch};
use tokio::task::JoinHandle;
use crate::node::Node as InnerNode;
use crate::node::pool::DeviceConnPool;
use super::Action;
use super::NodeError;
use super::NodeConfig;
use super::{NodeTickerSignal, NodeWatcherSignal};
use crate::utils::Position;

use log::debug;

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


pub struct Node<const POOL_SIZE: usize> {
    name:   &'static str,
    node:   Arc<InnerNode<POOL_SIZE>>,
    states: Arc<NodeState>,
    config: Arc<NodeConfig>,
    frame_buffer: Arc<RwLock<(u64, ImageBuffer<Rgba<u8>, Vec<u8>>)>>,

    ticker_rx: Arc<RwLock<mpsc::Receiver<NodeTickerSignal>>>,
    ticker_tx: mpsc::Sender<NodeTickerSignal>,
    watcher_rx: watch::Receiver<NodeWatcherSignal>,
    watcher_tx: watch::Sender<NodeWatcherSignal>,
}

impl<const POOL_SIZE: usize> Node<POOL_SIZE> {
    pub fn new(config: NodeConfig, server_addr: SocketAddrV4) -> Self {
        let iden = config.iden();
        let ev_device = config.ev_device();
        let (watcher_tx, watcher_rx)
            = watch::channel(NodeWatcherSignal::Created { node_name: config.name() });
        let (ticker_tx, ticker_rx) = mpsc::channel(8);
        Node {
            name: config.name(),
            node: Arc::new(InnerNode::new(DeviceConnPool::new(iden, server_addr), ev_device)),
            states: Arc::new(NodeState::new()),
            config: Arc::new(config),
            frame_buffer: Arc::new(Default::default()),
            ticker_tx,
            ticker_rx: Arc::new(RwLock::new(ticker_rx)),
            watcher_rx, watcher_tx,
        }
    }
    
    pub async fn schedule(&self) -> Result<JoinHandle<Result<(), NodeError>>, NodeError> {
        let fb = self.frame_buffer.clone();
        let node = self.node.clone();
        let config = self.config.clone();
        let states = self.states.clone();
        let ticker_rx = self.ticker_rx.clone();
        let watcher_tx = self.watcher_tx.clone();
        self.node.connect().await?;

        let handle = tokio::spawn(async move {
            // no need, drop frame
            // let fb_res = AtomicBool::new(true);
            
            let joystick_res = AtomicBool::new(true);
            let attack_res = AtomicBool::new(true);
            
            let skill_res = AtomicBool::new(true);
            let weapon_res = AtomicBool::new(true);

            let mut ticker_rx = ticker_rx.write().await;
            while let Some(action) = ticker_rx.recv().await {
                match action {
                    NodeTickerSignal::Tick(action) => {
                        let node_name = config.name();

                        // FrameBufferTask
                        let fb_sn = action.sn();
                        let _fb = fb.clone();
                        let _node = node.clone();
                        let _watcher_tx = watcher_tx.clone();
                        let fb_task = tokio::spawn(async move {
                            let res = {
                                let fb_new = _node.read_frame_buf().await;
                                match fb_new {
                                    Ok(fb_new) => {
                                        let mut _fb = _fb.write().await;
                                        *_fb = (fb_sn, fb_new);
                                        println!("[Node] FrameBufferTask: Node[{node_name}] FrameBuffer Updated.");
                                        Ok(())
                                    },
                                    Err(e) => {
                                        Err(NodeError::NodeErr(e))
                                    }
                                }

                            };

                            if let Err(err) = res {
                                let err = err.to_string();
                                _watcher_tx
                                    .send(NodeWatcherSignal::Error { node_name, err })
                                    .expect("[Fatal] Watcher Sender Failed to send.");
                            };
                        });

                        // JoystickTask
                        let _node = node.clone();
                        let _watcher_tx = watcher_tx.clone();
                        let touch_pos = config.keymap_get("joystick");
                        let direction = action.direction();
                        let joystick_task = tokio::spawn(async move {
                            let res = {
                                match direction {
                                    Some(direction) => {
                                        let distance = 128f64;
                                        let target = Position::new(
                                            touch_pos.x + (distance * direction.cos()) as u32,
                                            touch_pos.y + (distance * direction.sin()) as u32,
                                        );
                                        _node.touch_down("joystick", target).await
                                    },
                                    None => {
                                        _node.touch_up("joystick").await
                                    }
                                }.map_err(|e| NodeError::NodeErr(e))
                            };

                            if let Err(err) = res {
                                let err = err.to_string();
                                _watcher_tx
                                    .send(NodeWatcherSignal::Error { node_name, err })
                                    .expect("[Fatal] Watcher Sender Failed to send.");
                            };

                        });

                        // AttackTask

                        let _states = states.clone();
                        let attack = action.attack();
                        if !(attack ^ _states.is_attacking()) {
                            let _node = node.clone();
                            let _watcher_tx = watcher_tx.clone();
                            let touch_pos = config.keymap_get("attack");

                            let attack_task = tokio::spawn(async move {
                                let res: Result<(), NodeError> = {
                                    match attack {
                                        true  => _node.touch_down("attack", touch_pos).await,
                                        false => _node.touch_up("attack").await,
                                    }.map_err(|e| NodeError::NodeErr(e))
                                };
                                
                                _states.set_attack(attack);

                                if let Err(err) = res {
                                    let err = err.to_string();
                                    _watcher_tx
                                        .send(NodeWatcherSignal::Error {  node_name, err })
                                        .expect("[Fatal] Watcher Sender Failed to send.");
                                };
                            });
                        }

                        // SkillTask
                        if action.skill() {
                            let _node = node.clone();
                            let key = "skill";
                            let pos = config.keymap_get(key);
                            let skill_task
                                = Self::click(_node, config.name(), watcher_tx.clone(), pos, key);
                        }

                        // WeaponTask
                        if action.weapon() {
                            let _node = node.clone();
                            let key = "weapon";
                            let pos = config.keymap_get(key);
                            let weapon_task
                                = Self::click(_node, config.name(), watcher_tx.clone(), pos, key);
                        }
                    },
                    _ => {
                        todo!();
                    }
                }
            };

            Ok(())
        });
        Ok(handle)
    }
}

impl<const POOL_SIZE: usize> Node<POOL_SIZE> {
    pub async fn tick(&self,
                      signal: NodeTickerSignal
    ) -> Result<Arc<RwLock<(u64, ImageBuffer<Rgba<u8>, Vec<u8>>)>>, NodeError> {
        if let Err(err) = self.ticker_tx.send(signal).await {
            return Err(NodeError::ThreadErrorSend { node_name: self.name, err });
        }
        Ok(self.frame_buffer.clone())
    }
    
    pub fn watch(&self) -> watch::Receiver<NodeWatcherSignal> {
        self.watcher_rx.clone()
    }
    
    fn click(node: Arc<InnerNode<POOL_SIZE>>, node_name: &'static str,
             watcher: watch::Sender<NodeWatcherSignal>,
             pos: Position<u32>, key: &'static str) -> JoinHandle<Result<(), NodeError>> {
        tokio::spawn(async move {
            // let start = chrono::Local::now();
            let res: Result<(), NodeError> = {
                node.touch_down(key, pos).await?;
                tokio::time::sleep(std::time::Duration::from_millis(80)).await;
                node.touch_up(key).await?;
                Ok(())
            };

            if let Err(err) = res {
                let err = err.to_string();
                watcher
                    .send(NodeWatcherSignal::Error { node_name, err })
                    .expect("[Fatal] Watcher Sender Failed to send.");
            }
            Ok(())
        })
    }
    
    fn trigger(node: Arc<InnerNode<POOL_SIZE>>, node_name: &'static str,
               watcher: watch::Sender<NodeWatcherSignal>,
               pos: Position<u32>, key: &'static str) -> JoinHandle<Result<(), NodeError>> {
        tokio::spawn(async move {
            // let start = chrono::Local::now();
            let res: Result<(), NodeError> = {
                node.touch_down(key, pos).await?;
                node.touch_up(key).await?;
                Ok(())
            };

            if let Err(err) = res {
                let err = err.to_string();
                watcher
                    .send(NodeWatcherSignal::Error { node_name, err })
                    .expect("[Fatal] Watcher Sender Failed to send.");
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn test() -> Result<(), NodeError> {
    const FPS: f64 = 10.0;
    let sleep_duration = std::time::Duration::from_millis((1000.0 / FPS) as u64);
    let mut interval = tokio::time::interval(sleep_duration);
    
    use serde_json;
    let configs: Vec<NodeConfig> = serde_json::from_str(include_str!("../../configs/node.json")).unwrap();
    let config = configs.get(0).unwrap();
    
    let node: Node<16> = Node::new(config.clone(), "127.0.0.1:5037".parse().unwrap());
    let _handle = node.schedule().await?;
    
    for i in 0..100 {
        // let action = Action::new(i, Some(0.0), true, true, true);
        let action = Action::new(i, None, false, false, true);
        node.tick(NodeTickerSignal::Tick(action)).await.expect("我超");
        interval.tick().await;
        println!("tick!!!!!")
    };
    
    Ok(())
}