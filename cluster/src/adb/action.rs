use std::fmt;
use std::ops::Add;
use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::{RwLock};
use crate::adb::event::{Key, KeyValue, AdbEvent};
use crate::utils::{log, Position};

#[derive(Debug)]
pub struct InputAction<'a> {
    pub ev_device: &'a str,
    payload: Vec<AdbEvent>,
}
impl<'a> InputAction<'a> {
    pub fn new(payload: Vec<AdbEvent>, ev_device: &'a str) -> Self {
        Self {
            ev_device,
            payload,
        }
    }
    pub fn cmd(&self) -> String {
        let mut ret = vec![];
        ret.push(AdbEvent::SynReport(0).to_command());
        for ev in &self.payload {
            ret.push(ev.to_command());
        }
        let ret = ret.iter()
            .map(|ev| format!("sendevent {} {}", self.ev_device, ev));
        ret.collect::<Vec<String>>().join(" && ").add("\r\n")
    }
}

#[derive(Debug)]
pub struct InputStatus {
    key: Arc<DashMap<u32, KeyValue>>,
    touch: Arc<DashMap<String, (u32, Position<u32>)>>,
    last_touch: Arc<RwLock<String>>,
}
impl fmt::Display for InputStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "key: {:?}, touch: {:?}, last_touch: {}",
               self.key, self.touch, self.last_touch.blocking_read())
    }
}


pub struct InputFactory {
    ev_device: String,
    status: InputStatus,
}

impl InputFactory {
    const SCREEN_SIZE: (u32, u32) = (1280, 720);
    const TOUCH_UP_TRACK_ID: u32 = 0xc352;
    
    pub fn new(ev_device: &str) -> Self {
        Self {
            ev_device: ev_device.to_string(),
            status: InputStatus {
                key: Arc::new(DashMap::new()),
                touch: Arc::new(DashMap::new()),
                last_touch: Arc::new(RwLock::new(String::new())),
            },
        }
    }
    
    pub fn status(&self) -> &InputStatus {
        &self.status
    }

    pub(crate) async fn get_key_down_action(&self, key: Key) -> InputAction {
        self.status.key.insert(key as u32, KeyValue::Down);
        InputAction::new(vec![AdbEvent::Key(key, KeyValue::Down)], &self.ev_device)
    }

    pub(crate) async fn get_key_up_action(&self, key: Key) -> InputAction {
        self.status.key.insert(key as u32, KeyValue::Up);
        InputAction::new(vec![AdbEvent::Key(key, KeyValue::Up)], &self.ev_device)
    }

    pub(crate) async fn get_touch_down_action(&self, iden: &str, pos: Position<u32>) -> Option<InputAction> {
        let mut is_move = false;
        let mut slot_id = 2;
        for item in self.status.touch.iter() {
            if *item.key() == iden { is_move = true; break }
            if item.value().0 == slot_id {
                slot_id += 1;
            }
        }
        if is_move { return self.get_touch_move_action(iden, pos).await }

        self.status.touch.insert(iden.to_string(), (slot_id, pos));
        let mut last_touch = self.status.last_touch.write().await;
        *last_touch = iden.to_string();
        drop(last_touch);

        // let entry = self.status.touch.entry(iden.to_string());
        // if let dashmap::Entry::Occupied(_) = entry {
        //     return self.get_touch_move_action(iden, pos).await;
        // }
        // let slot_id = ...;
        // entry.or_insert((slot_id, pos));

        let ret = vec![
            AdbEvent::AbsMtSlot { slot_id },
            AdbEvent::AbsMtTrackingId { slot_id },
            AdbEvent::BtnTouch(KeyValue::Down),
            AdbEvent::AbsMtPositionX(pos.x),
            AdbEvent::AbsMtPositionY(pos.y),
        ];

        Some(InputAction::new(ret, &self.ev_device))
    }

    pub(crate) async fn get_touch_move_action(&self, iden: &str, pos: Position<u32>) -> Option<InputAction> {
        let mut slot_id: Option<u32> = None;
        
        self.status.touch.entry(iden.to_string())
            .and_modify(|v| {
                let (inner_id, inner_pos) = v;
                slot_id = Some(*inner_id);
                *inner_pos = pos;
            });
        let slot_id = slot_id.expect("touch not found");
        let mut last_touch = self.status.last_touch.write().await;

        let mut ret = Vec::with_capacity(4);
        ret.push(AdbEvent::BtnTouch(KeyValue::Down));
        if *last_touch != iden {
            ret.push(AdbEvent::AbsMtSlot { slot_id });
            *last_touch = iden.to_string();
        }
        drop(last_touch);
        ret.push(AdbEvent::AbsMtPositionX(pos.x));
        ret.push(AdbEvent::AbsMtPositionY(pos.y));

        Some(InputAction::new(ret, &self.ev_device))
    }

    pub(crate) async fn get_touch_up_action(&self, iden: &str) -> Option<InputAction> {
        if let Some((_, (slot_id, _))) = self.status.touch.remove(iden) {
            let mut last_touch = self.status.last_touch.write().await;
            let mut ret = Vec::with_capacity(3);
            if *last_touch != iden {
                ret.push(AdbEvent::AbsMtSlot { slot_id });
            }
            *last_touch = String::new();
            drop(last_touch);
            ret.push(AdbEvent::AbsMtTrackingId { slot_id: Self::TOUCH_UP_TRACK_ID });
            ret.push(AdbEvent::BtnTouch(KeyValue::Up));
            
            return Some(InputAction::new(ret, &self.ev_device))
        }

        log(&format!("touch['{iden}'] not found"));
        None
    }
}

#[tokio::test]
async fn action() -> Result<(), Box<dyn std::error::Error>> {
    let action = InputFactory::new("/dev/input/event4");
    let action = action.get_touch_down_action("joystick", Position::new(100, 100)).await;
    println!("{:?}", action.unwrap().cmd());
    Ok(())
} 