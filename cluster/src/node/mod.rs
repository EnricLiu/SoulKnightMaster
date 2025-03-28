mod status;

use std::collections::VecDeque;
pub(crate) use status::NodeStatus;

use std::net::SocketAddrV4;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use log::{error, trace};
use tokio::sync::{mpsc, watch, broadcast, Mutex, RwLock};
use tokio::task::JoinHandle;
use crate::adb::Device;
use crate::adb::pool::DeviceConnPool;
use crate::cluster::{RawAction, SoulKnightAction};
use crate::cluster::{FrameBuffer, NodeError};
use crate::cluster::NodeConfig;
use crate::utils::{perf_log, perf_timer, Position};

use crate::adb::event::KeyValue;
use crate::node::status::{AtomicNodeStatus, NodeStatusCode};
use crate::cluster::{NodeSignal, NodeWatcherSignal};

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
    config: Arc<NodeConfig>,
    device: Arc<Device<POOL_SIZE>>,

    status: Arc<AtomicNodeStatus>,
    task:   Arc<Mutex<Option<JoinHandle<Result<(), NodeError>>>>>,
    
    fb_sn:  Arc<AtomicU64>,
    fb_tx:  watch::Sender<FrameBuffer>,
    fb_rx:  watch::Receiver<FrameBuffer>,
    
    act_sn: Arc<AtomicU64>,
    act_tx: mpsc::Sender<NodeSignal>,
    act_rx: Arc<Mutex<mpsc::Receiver<NodeSignal>>>,
    
    res_rx: broadcast::Receiver<NodeWatcherSignal>,
    res_tx: broadcast::Sender<NodeWatcherSignal>,
}

impl<const POOL_SIZE: usize> Node<POOL_SIZE> {
    const FB_INTERVAL_MS: u64 = 50; //ms
    const STATUS_INTERVAL_MS: u64 = 500; //ms
    const FB_GUARD_INTERVAL_MS: u64 = 15; //s
    
    pub fn new(config: NodeConfig, server_addr: SocketAddrV4) -> Self {
        let name = config.name();
        let iden = config.iden();
        let ev_device = config.ev_device();
        let (res_tx, res_rx)
            = broadcast::channel(16);

        let (fb_tx, fb_rx)
            = watch::channel(FrameBuffer::new(name, 0, vec![]));
        
        let (act_tx, act_rx) = mpsc::channel(16);
        
        Node {
            name,
            device:   Arc::new(Device::new(DeviceConnPool::new(iden, server_addr), ev_device)),
            config: Arc::new(config),

            status: Arc::new(AtomicNodeStatus::new(name, POOL_SIZE)),
            task:   Default::default(),
            
            fb_sn: Arc::new(AtomicU64::new(0)),
            fb_tx, fb_rx,
            
            act_sn: Arc::new(AtomicU64::new(0)),
            act_tx, act_rx: Arc::new(Mutex::new(act_rx)),
            
            res_rx, res_tx,
        }
    }

    pub fn get_status(&self) -> NodeStatus {
        self.status.snap()
    }
    
    pub fn get_server_name(&self) -> &'static str {
        self.config.server()
    }
    
    pub fn get_iden(&self) -> &'static str {
        self.config.iden()
    }
    
    pub fn watch(&self) -> broadcast::Receiver<NodeWatcherSignal> {
        self.res_tx.subscribe()
    }
    
    pub async fn get_fb(&self) -> FrameBuffer {
        self.fb_rx.borrow().clone()
    }
    
    pub async fn act(&self, signal: NodeSignal) -> Result<(), NodeError> {
        if !self.status.snap().is_ready() {
            return Err(NodeError::NodeNotScheduled { name: self.name })
        }
        self.act_tx.send(signal).await
            .map_err(|err| NodeError::SendErrorAction { name: self.name, err })
    }
    
    pub async fn schedule(&self) -> Result<(), NodeError> {
        if self.task.lock().await.is_some() {
            return Err(NodeError::NodeAlreadyScheduled { name: self.name })
        }
        
        let name = self.name;
        let node = self.device.clone();
        let config = self.config.clone();
        node.connect().await?;
        
        let status = self.status.clone();
        self.status.set_status(NodeStatusCode::IDLE);
        
        let act_sn = self.act_sn.clone();
        let act_rx = self.act_rx.clone();
        
        let fb_tx = self.fb_tx.clone();
        let fb_sn = self.fb_sn.clone();

        let res_tx = self.res_tx.clone();
        // loosely guarantee
        if let Err(_) = act_rx.try_lock() {
            return Err(NodeError::NodeAlreadyScheduled { name })
        }

        let send_if_err = |res: Result<(), NodeError>,
             watcher: &broadcast::Sender<NodeWatcherSignal>| {
                if res.is_ok() { return }
                let err = res.unwrap_err().to_string();
                watcher
                    .send(NodeWatcherSignal::Error { node_name: name, err })
                    .expect("[Fatal] Result Sender Failed to send.");
            };

        res_tx.send(NodeWatcherSignal::Ready { node_name: name })
            .expect("[Fatal] Result Sender Failed to send.");

        self.status.set_status(NodeStatusCode::RUNNING);
        let schedule = tokio::spawn(async move {
            
            let action
                = Arc::new(RwLock::new(SoulKnightAction::new(0, None, false, false, false)));
            let stop_flag = Arc::new(AtomicBool::new(false));
            
            let _device = node.clone();
            let _action = action.clone();
            let _status = status.clone();
            let _config = config.clone();
            let _act_sn = act_sn.clone();
            let _res_tx = res_tx.clone();
            let _stop_flag = stop_flag.clone();
            let _act_task_: JoinHandle<Result<(), NodeError>> = tokio::spawn(async move {
                let attack_flag = Arc::new(AtomicBool::new(false));
                loop {
                    if _stop_flag.load(Ordering::SeqCst) { break }
                    _status.task_start();
                    let action = _action.read().await;
                    let curr_sn = _act_sn.load(Ordering::SeqCst);
                    if curr_sn < action.sn() {
                        if action.skill() {
                            let pos = _config.keymap_get("skill");
                            let res = _device.tap(pos).await
                                .map_err(|err| NodeError::ActionFailed { name, action: "skill", err });
                            send_if_err(res, &_res_tx);
                        }
                        if action.weapon() {
                            let pos = _config.keymap_get("weapon");
                            let res = _device.tap(pos).await
                                .map_err(|err| NodeError::ActionFailed { name, action: "weapon", err });
                            send_if_err(res, &_res_tx);
                        }
                        
                        if action.attack() {
                            let pos = _config.keymap_get("attack");
                            let res = _device.tap(pos).await
                                .map_err(|err| NodeError::ActionFailed { name, action: "attack", err });
                            send_if_err(res, &_res_tx);
                        }

                        _act_sn.store(action.sn(), Ordering::SeqCst);
                    }

                    let start = perf_timer();


                    if let Some(dir) = action.direction() {
                        attack_flag.store(true, Ordering::SeqCst);
                        let distance = 128f64;
                        let touch_pos = _config.keymap_get("joystick");
                        let target = Position::new(
                            (touch_pos.x as i32 + (distance * dir.cos()) as i32) as u32,
                            (touch_pos.y as i32 + (distance * dir.sin()) as i32) as u32,
                        );
                        let res = _device.motion(target, KeyValue::Down).await
                            .map_err(|err| NodeError::ActionFailed { name, action: "joystick", err });
                        send_if_err(res, &_res_tx);
                    } else {
                        if attack_flag.load(Ordering::SeqCst) {
                            let pos = _config.keymap_get("joystick");
                            let res = _device.motion(pos, KeyValue::Up).await
                                .map_err(|err| NodeError::ActionFailed { name, action: "joystick", err });
                            send_if_err(res, &_res_tx);
                            attack_flag.store(false, Ordering::SeqCst);
                        }
                    }
                    _status.task_end();
                    perf_log(&format!("[{name}] Action finished"), start);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                };
                trace!("[{name}] Action task finished.");
                Ok(())
            });

            let _device = node.clone();
            let _action = action.clone();
            let _status = status.clone();
            let _res_tx = res_tx.clone();
            let _stop_flag = stop_flag.clone();
            let _recv_task_ = tokio::spawn(async move {
                let mut _act_rx = act_rx.try_lock();
                if let Err(_) = _act_rx {
                    return Err(NodeError::NodeAlreadyScheduled { name })
                }
                _status.task_start();
                let mut _act_rx = _act_rx.unwrap();
                while let Some(signal) = _act_rx.recv().await {
                    match signal {
                        NodeSignal::Action(action) => {
                            let mut __action = _action.write().await;
                            *__action = action;
                        },
                        NodeSignal::RawAction(action) => {
                            let __device = _device.clone();
                            let __res_tx = _res_tx.clone();
                            tokio::spawn(async move {
                                match action {
                                    RawAction::Click { pos} => {
                                        let res = __device.tap(pos).await
                                            .map_err(|err| NodeError::ActionFailed { name, action: "click", err });
                                        send_if_err(res, &__res_tx);
                                    },
                                    RawAction::Key { key, val } => {
                                        todo!();
                                    }
                                }
                            });
                        }
                        NodeSignal::Close => {
                            // _stop_flag.store(true, Ordering::SeqCst);
                            break;
                        }
                        _ => {}
                    }
                };
                _stop_flag.store(true, Ordering::SeqCst);
                trace!("[{name}] Recv task finished.");
                _status.task_end();
                Ok(())
            });
            
            
            let _device = node.clone();
            let _fb_tx = fb_tx.clone();
            let _fb_sn = fb_sn.clone();
            let _res_tx = res_tx.clone();
            let _status = status.clone();
            let _stop_flag = stop_flag.clone();
            let _fb_task_: JoinHandle<Result<(), NodeError>> = tokio::spawn(async move {
                let __device = _device.clone();
                let mut interval = tokio::time::interval(Duration::from_millis(Self::FB_INTERVAL_MS));
                loop {
                    if _stop_flag.load(Ordering::SeqCst) { break }
                    _status.task_start();
                    interval.tick().await;
                    let res = tokio::time::timeout(
                        Duration::from_millis(500),
                        async {
                            let fb_new = __device.read_frame_buf().await;
                            if let Ok(fb_new) = fb_new {
                                let res = _fb_tx.send(FrameBuffer::new(name, 0, fb_new.into_vec()))
                                    .map_err(|err| NodeError::SendErrorFb { name, err });
                                send_if_err(res, &_res_tx);
                                _fb_sn.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    ).await.map_err(|_| NodeError::FbTimeout { name, sn: _fb_sn.load(Ordering::SeqCst) });
                    if res.is_err() {
                        let res = _device.connect().await;
                        send_if_err(res.map_err(|err| NodeError::AdbError(err)), &_res_tx);
                    }
                    send_if_err(res, &_res_tx);
                    _status.task_end();
                }


                // let mut sn = 0;
                // let mut tasks = VecDeque::new();
                // let mut interval = tokio::time::interval(Duration::from_millis(Self::FB_INTERVAL_MS));
                //
                // loop {
                //     if _stop_flag.load(Ordering::SeqCst) { break }
                //     _status.task_start();
                //     sn += 1;
                //     interval.tick().await;
                //     let _old_sn = _fb_sn.load(Ordering::SeqCst);
                //     if _old_sn >= sn {
                //         return Err(NodeError::FbSnCorrupt(sn, _old_sn))
                //     }
                //
                //     let __device = _device.clone();
                //     let __status = _status.clone();
                //     let __fb_sn = _fb_sn.clone();
                //     let __fb_tx = _fb_tx.clone();
                //     let __res_tx = _res_tx.clone();
                //     let task = tokio::spawn(async move {
                //         __status.task_start();
                //         let res = tokio::time::timeout(
                //             Duration::from_millis(1000),
                //             async {
                //                 let fb_new = __device.read_frame_buf().await;
                //                 if let Ok(fb_new) = fb_new {
                //                     if __fb_sn.load(Ordering::SeqCst) >= sn { return }
                //                     let res = __fb_tx.send(FrameBuffer::new(name, 0, fb_new.into_vec()))
                //                         .map_err(|err| NodeError::SendErrorFb { name, err });
                //                     send_if_err(res, &__res_tx);
                //                     __fb_sn.store(sn, Ordering::SeqCst);
                //                 }
                //             }
                //         ).await.map_err(|_| NodeError::FbTimeout { name, sn });
                //         send_if_err(res, &__res_tx);
                //         __status.task_end();
                //     });
                //
                //     tasks.push_back((sn, task));
                //     let latest_sn = _fb_sn.load(Ordering::SeqCst);
                //     loop {
                //         if let Some((sn, task)) = tasks.pop_front() {
                //             if sn <= latest_sn {
                //                 let _ = task.await;
                //                 continue;
                //             } else {
                //                 tasks.push_front((sn, task));
                //                 break;
                //             }
                //         } else {
                //             break;
                //         }
                //     }
                //     _status.task_end();
                // }
                //
                // for (_, task) in tasks {
                //     let _ = task.await;
                // }
                Ok(())
            });

            let _fb_sn = fb_sn.clone();
            let _status = status.clone();
            let _stop_flag = stop_flag.clone();
            let _fb_guard_task_: JoinHandle<Result<(), NodeError >> = tokio::spawn(async move {
                let mut last_fb_sn = 0u64;

                let mut time_cnt = 0u64;
                let mut interval = tokio::time::interval(Duration::from_millis(1000));
                loop {
                    if _stop_flag.load(Ordering::SeqCst) { break }
                    interval.tick().await;
                    time_cnt += 1;
                    if time_cnt < Self::FB_GUARD_INTERVAL_MS { continue }
                    time_cnt = 0;

                    let fb_sn = _fb_sn.load(Ordering::SeqCst);
                    if fb_sn <= last_fb_sn {
                        _status.set_status(NodeStatusCode::DEAD);
                        _stop_flag.store(true, Ordering::SeqCst);
                        error!("Node[{name}] Dead due to Timeout Fetching FrameBuf.");
                        continue;
                    }
                    last_fb_sn = fb_sn;
                }

                Ok(())
            });
            let _device = node.clone();
            let _status = status.clone();
            let _fb_sn = fb_sn.clone();
            let _stop_flag = stop_flag.clone();
            let _status_task_ = tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(Self::STATUS_INTERVAL_MS));
                let mut fb_cnt = _fb_sn.load(Ordering::SeqCst);
                loop {
                    if _stop_flag.load(Ordering::SeqCst) {
                        _status.set_fps(0.0);
                        break;
                    }
                    interval.tick().await;
                    _status.set_thread(_device.pool_len().await);
                    let new_fn_sn = _fb_sn.load(Ordering::SeqCst);
                    _status.set_fps((new_fn_sn - fb_cnt) as f64
                                    * 1000.0 / Self::STATUS_INTERVAL_MS as f64);
                    fb_cnt = new_fn_sn;
                }

                trace!("[{name}] Status task finished.");
            });
            
            let _res = tokio::join! {
                _act_task_,
                _recv_task_,
                _fb_task_,
                _status_task_
            };

            status.set_status(NodeStatusCode::STOPPED);
            Ok(())
        });
        
        let mut task = self.task.lock().await;
        *task = Some(schedule);
        
        Ok(())
    }
    
    pub async fn deschedule(&self) -> Result<(), NodeError> {
        let schedule = self.task.lock().await.take();
        if schedule.is_none() {
            return Err(NodeError::NodeNotScheduled { name: self.name })
        }
        let schedule: JoinHandle<Result<(), NodeError>> = schedule.unwrap();
        self.act_tx.send(NodeSignal::Close).await
            .map_err(|err| NodeError::SendErrorAction { name: self.name, err })?;
        
        let wait_close_task = async {
            schedule.await
                .map_err(|_join_err| NodeError::NodeTaskCanNotJoin { name: self.name })?
        };
        
        tokio::time::timeout(Duration::from_secs(5), wait_close_task).await
            .map_err(|_elapsed| NodeError::DescheduleTimeout { name: self.name })?
    }
    
    pub async fn release(&self) -> Result<(), NodeError> {
        self.deschedule().await?;
        Ok(())
    }
}


#[tokio::test]
async fn test() -> Result<(), NodeError> {
    const FPS: f64 = 10.0;
    let sleep_duration = Duration::from_millis((1000.0 / FPS) as u64);
    let mut interval = tokio::time::interval(sleep_duration);
    
    use serde_json;
    let configs: Vec<NodeConfig> = serde_json::from_str(include_str!("../../configs/node.json")).unwrap();
    let config = configs.get(0).unwrap();
    
    let node: Node<16> = Node::new(config.clone(), "127.0.0.1:5037".parse().unwrap());
    let _handle = node.schedule().await?;
    
    for i in 0..100 {
        // let action = Action::new(i, Some(0.0), true, true, true);
        let action = SoulKnightAction::new(i, None, false, false, true);
        node.act(NodeSignal::Action(action)).await.expect("我超");
        interval.tick().await;
        println!("act!!!!!")
    };
    
    Ok(())
}