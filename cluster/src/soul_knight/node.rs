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
            let node_name = config.name();
            let send_if_err =
                |res: Result<(), NodeError>,
                 watcher: &watch::Sender<NodeWatcherSignal>| {
                    if res.is_ok() { return }
                    let err = res.unwrap_err().to_string();
                    watcher
                        .send(NodeWatcherSignal::Error { node_name, err })
                        .expect("[Fatal] Watcher Sender Failed to send.");
                };

            let mut ticker_rx = ticker_rx.write().await;
            while let Some(action) = ticker_rx.recv().await {
                match action {
                    NodeTickerSignal::Tick(action) => {
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
                        
                        let _node = node.clone();
                        let _config = config.clone();
                        let _watcher_tx = watcher_tx.clone();
                        let _states = states.clone();
                        let event_task = tokio::spawn(async move {
                            // ------------- skill_down ------------
                            if action.skill() {
                                let res = _node
                                    .touch_down("skill", _config.keymap_get("skill")).await
                                    .map_err(|e| NodeError::NodeErr(e));
                                send_if_err(res, &_watcher_tx);
                            }
                            // ------------ action_down ------------
                            if action.weapon() {
                                let res = _node
                                    .touch_down("weapon", _config.keymap_get("weapon")).await
                                    .map_err(|e| NodeError::NodeErr(e));
                                send_if_err(res, &_watcher_tx);
                            }
                            // ------------ joystick ------------ 
                            let res = {
                                match action.direction() {
                                    Some(direction) => {
                                        let distance = 128f64;
                                        let touch_pos = _config.keymap_get("joystick");
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
                            send_if_err(res, &_watcher_tx);

                            // ------------- attack ------------- 
                            let is_attack = action.attack();
                            if !(is_attack ^ _states.is_attacking()) {
                                let touch_pos = _config.keymap_get("attack");

                                let res: Result<(), NodeError> = {
                                    match is_attack {
                                        true => _node.touch_down("attack", touch_pos).await,
                                        false => _node.touch_up("attack").await,
                                    }.map_err(|e| NodeError::NodeErr(e))
                                };

                                _states.set_attack(is_attack);
                                send_if_err(res, &_watcher_tx);

                            }

                            // ------------- skill_up -------------
                            if action.skill() {
                                let res = _node
                                    .touch_up("skill").await
                                    .map_err(|e| NodeError::NodeErr(e));
                                send_if_err(res, &_watcher_tx);
                            }
                            // ------------ action_up -------------
                            if action.weapon() {
                                let res = _node
                                    .touch_up("weapon").await
                                    .map_err(|e| NodeError::NodeErr(e));
                                send_if_err(res, &_watcher_tx);
                            }
                            
                        });
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