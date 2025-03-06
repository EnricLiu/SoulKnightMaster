use std::fmt;
use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::{RwLock};
use crate::node::event::{Key, KeyValue, NodeEvent};
use crate::utils::Position;

pub struct NodeAction<'a> {
    pub ev_device: &'a str,
    payload: Vec<NodeEvent>,
}
impl<'a> NodeAction<'a> {
    pub fn new(payload: Vec<NodeEvent>, ev_device: &'a str) -> Self {
        Self {
            ev_device,
            payload,
        }
    }
    pub fn cmd(&self) -> String {
        let mut ret = vec![];
        ret.push(NodeEvent::SynReport(0).to_command());
        for ev in &self.payload {
            ret.push(ev.to_command());
        }
        let ret = ret.iter()
            .map(|ev| format!("sendevent {} {}", self.ev_device, ev));
        ret.collect::<Vec<String>>().join(" && ")
    }
}

#[derive(Debug)]
pub struct NodeActionStatus {
    key: Arc<DashMap<u32, KeyValue>>,
    touch: Arc<DashMap<String, (u32, Position<u32>)>>,
    last_touch: Arc<RwLock<String>>,
}
impl fmt::Display for NodeActionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "key: {:?}, touch: {:?}, last_touch: {}",
               self.key, self.touch, self.last_touch.blocking_read())
    }
}


pub struct ActionFactory {
    ev_device: String,
    status: NodeActionStatus,
}

impl ActionFactory {
    const TOUCH_UP_TRACK_ID: u32 = 0xc352;
    
    pub fn new(ev_device: &str) -> Self {
        Self {
            ev_device: ev_device.to_string(),
            status: NodeActionStatus {
                key: Arc::new(DashMap::new()),
                touch: Arc::new(DashMap::new()),
                last_touch: Arc::new(RwLock::new(String::new())),
            },
        }
    }
    
    pub fn status(&self) -> &NodeActionStatus {
        &self.status
    }

    pub(crate) async fn get_key_down_action(&self, key: Key) -> NodeAction {
        self.status.key.insert(key as u32, KeyValue::Down);
        NodeAction::new(vec![NodeEvent::Key(key, KeyValue::Down)], &self.ev_device)
    }

    pub(crate) async fn get_key_up_action(&self, key: Key) -> NodeAction {
        self.status.key.insert(key as u32, KeyValue::Up);
        NodeAction::new(vec![NodeEvent::Key(key, KeyValue::Up)], &self.ev_device)
    }

    pub(crate) async fn get_touch_down_action(&self, iden: &str, pos: Position<u32>) -> NodeAction {
        let mut is_move = false;
        let mut slot_id = 1;
        for item in self.status.touch.iter() {
            if *item.key() == iden { is_move = true; break }
            if item.value().0 == slot_id {
                slot_id += 1;
            }
        }
        if is_move { return self.get_touch_move_action(iden, pos).await }
        
        let ret = vec![
            NodeEvent::AbsMtSlot { slot_id },
            NodeEvent::AbsMtTrackingId { slot_id },
            NodeEvent::BtnTouch(KeyValue::Down),
            NodeEvent::AbsMtPositionX(pos.x),
            NodeEvent::AbsMtPositionY(pos.y),
        ];
        
        let mut last_touch = self.status.last_touch.write().await;
        *last_touch = iden.to_string();
        drop(last_touch);
        self.status.touch.insert(iden.to_string(), (slot_id, pos));
        NodeAction::new(ret, &self.ev_device)
    }

    pub(crate) async fn get_touch_move_action(&self, iden: &str, pos: Position<u32>) -> NodeAction {
        if !self.status.touch.contains_key(iden) {
            panic!("touch not found");
        }
        let slot_id = self.status.touch.get(iden).unwrap().0;

        let mut ret = Vec::with_capacity(4);
        ret.push(NodeEvent::BtnTouch(KeyValue::Down));
        let mut last_touch = self.status.last_touch.write().await;
        if *last_touch != iden {
            ret.push(NodeEvent::AbsMtSlot { slot_id });
            *last_touch = iden.to_string();
        }
        drop(last_touch);
        ret.push(NodeEvent::AbsMtPositionX(pos.x));
        ret.push(NodeEvent::AbsMtPositionY(pos.y));

        self.status.touch.insert(iden.to_string(), (slot_id, pos));
        NodeAction::new(ret, &self.ev_device)
    }

    pub(crate) async fn get_touch_up_action(&self, iden: &str) -> NodeAction {
        if !self.status.touch.contains_key(iden) {
            panic!("touch not found");
        }

        let mut ret = Vec::with_capacity(3);
        let mut last_touch = self.status.last_touch.write().await;
        if *last_touch != iden {
            let slot_id = self.status.touch.get(iden).unwrap().0;
            ret.push(NodeEvent::AbsMtSlot { slot_id });
        }
        *last_touch = String::new();
        drop(last_touch);
        ret.push(NodeEvent::AbsMtTrackingId { slot_id: Self::TOUCH_UP_TRACK_ID });
        ret.push(NodeEvent::BtnTouch(KeyValue::Up));
        
        self.status.touch.remove(iden);
        NodeAction::new(ret, &self.ev_device)
    }
}

#[tokio::test]
async fn action() -> Result<(), Box<dyn std::error::Error>> {
    let action = ActionFactory::new("/dev/input/event4");
    let action = action.get_touch_down_action("joystick", Position::new(100, 100)).await;
    println!("{:?}", action.cmd());
    Ok(())
} 