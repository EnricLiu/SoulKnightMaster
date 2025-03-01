use std::fmt;
use std::sync::Arc;
use dashmap::DashMap;
use crate::node::event::{Key, KeyValue, NodeEvent};
use crate::utils::Position;

pub struct NodeAction {
    payload: Vec<NodeEvent>,
    formatted: Option<Vec<[String; 5]>>
}
impl NodeAction {
    pub fn new(payload: Vec<NodeEvent>) -> Self {
        Self {
            payload,
            formatted: None,
        }
    }

    pub fn into_cmd(self, ev_device: &str) -> Vec<[String; 5]> {
        let mut ret = vec![];
        ret.push(NodeEvent::SynReport(0).to_command(ev_device));
        for ev in self.payload {
            ret.push(ev.to_command(ev_device));
        }
        ret
    }
}

#[derive(Debug)]
pub struct NodeActionStatus<'a> {
    key: Arc<DashMap<u32, KeyValue>>,
    touch: Arc<DashMap<&'a str, (u32, Position<u32>)>>,
    last_touch: &'a str,
}
impl fmt::Display for NodeActionStatus<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "key: {:?}, touch: {:?}, last_touch: {}", self.key, self.touch, self.last_touch)
    }
}


pub struct NodeActionController<'a> {
    ev_device: &'static str,
    status: NodeActionStatus<'a>,
}

impl<'a> NodeActionController<'a> {
    pub fn new() -> Self {
        Self {
            ev_device: "/dev/input/event4",
            status: NodeActionStatus {
                key: Arc::new(DashMap::new()),
                touch: Arc::new(DashMap::new()),
                last_touch: "",
            },
        }
    }

    pub fn key_down(&self, key: Key) -> NodeAction {
        self.status.key.insert(key as u32, KeyValue::Down);
        NodeAction::new(vec![NodeEvent::Key(key, KeyValue::Down)])
    }

    pub fn key_up(&self, key: Key) -> NodeAction {
        self.status.key.insert(key as u32, KeyValue::Up);
        NodeAction::new(vec![NodeEvent::Key(key, KeyValue::Up)])
    }

    pub fn touch_down(&mut self, iden: &'a str, pos: Position<u32>) -> NodeAction {
        let mut is_move = false;
        let mut is_multi_touch = false;
        let mut slot_id = 1;
        for item in self.status.touch.iter() {
            is_multi_touch = true;
            if *item.key() == iden { is_move = true; break }

            if item.value().0 == slot_id {
                slot_id += 1;
            }
        }

        if is_move { return self.touch_move(iden, pos) }

        let mut ret = Vec::with_capacity(5);
        if is_multi_touch {
            ret.push(NodeEvent::AbsMtSlot { slot_id });
        }
        ret.push(NodeEvent::AbsMtTrackingId { slot_id });
        ret.push(NodeEvent::BtnTouch(KeyValue::Down));
        ret.push(NodeEvent::AbsMtPositionX(pos.x));
        ret.push(NodeEvent::AbsMtPositionY(pos.y));

        self.status.last_touch = iden;
        self.status.touch.insert(iden, (slot_id, pos));
        NodeAction::new(ret)
    }

    pub fn touch_move(&mut self, iden: &'a str, pos: Position<u32>) -> NodeAction {
        if !self.status.touch.contains_key(&iden) {
            panic!("touch not found");
        }
        let slot_id = self.status.touch.get(iden).unwrap().0;

        let mut ret = Vec::with_capacity(4);
        ret.push(NodeEvent::BtnTouch(KeyValue::Down));
        if self.status.last_touch != iden {
            ret.push(NodeEvent::AbsMtSlot { slot_id });
        };
        ret.push(NodeEvent::AbsMtPositionX(pos.x));
        ret.push(NodeEvent::AbsMtPositionY(pos.y));

        self.status.last_touch = iden;
        self.status.touch.insert(iden, (slot_id, pos));
        NodeAction::new(ret)
    }

    pub fn touch_up(&mut self, iden: &'a str) -> NodeAction {
        if !self.status.touch.contains_key(&iden) {
            panic!("touch not found");
        }

        let mut ret = Vec::with_capacity(3);
        if self.status.last_touch != iden {
            let slot_id = self.status.touch.get(iden).unwrap().0;
            ret.push(NodeEvent::AbsMtSlot { slot_id });
        }
        ret.push(NodeEvent::AbsMtTrackingId { slot_id: 0xc350 });
        ret.push(NodeEvent::BtnTouch(KeyValue::Up));

        self.status.last_touch = iden;
        self.status.touch.remove(iden);
        NodeAction::new(ret)
    }
}

#[test]
fn action() -> Result<(), Box<dyn std::error::Error>> {
    let mut action = NodeActionController::new();
    let action = action.touch_down("joystick", Position::new(100, 100));
    println!("{:?}", action.into_cmd("/dev/input/event4"));
    Ok(())
}