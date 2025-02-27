use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum AdbEvent {
    AbsMtTrackingId { slot_id: u32 } = 0x0039,
    AbsMtSlot { slot_id: u32 } = 0x002f,
    
    AbsMtPositionX(u32) = 0x0035,
    AbsMtPositionY(u32) = 0x0036,
    
    KeyW(AdbEventKey)       = 0x0011,
    KeyA(AdbEventKey)       = 0x001e,
    KeyS(AdbEventKey)       = 0x001f,
    KeyD(AdbEventKey)       = 0x0020,
    BtnTouch(AdbEventKey)   = 0x014A,
    
    SynReport(u32)   = 0x0000,
}

impl AdbEvent {
    pub(crate) fn value(&self) -> u32 {
        unsafe { 
            let res = std::mem::transmute::<Self, u64>(*self);
            res as u32
        }
    }

    pub(crate)fn key_value(&self) -> u32 {
        match self {
            AdbEvent::AbsMtTrackingId { slot_id } => *slot_id,
            AdbEvent::AbsMtSlot { slot_id } => *slot_id,
            
            AdbEvent::AbsMtPositionX(x) => *x,
            AdbEvent::AbsMtPositionY(y) => *y,
            
            AdbEvent::KeyW(v) => *v as u32,
            AdbEvent::KeyA(v) => *v as u32,
            AdbEvent::KeyS(v) => *v as u32,
            AdbEvent::KeyD(v) => *v as u32,
            AdbEvent::BtnTouch(v) => *v as u32,
            
            AdbEvent::SynReport(v) => *v,
        }   
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum AdbEventKey {
    Up = 0,
    Down = 1,
}

impl TryFrom<i32> for AdbEventKey {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Up),
            1 => Ok(Self::Down),
            _ => Err(format!("Invalid AdbEventValue value: {}", value)),
        }
    }
}

impl FromStr for AdbEventKey {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            _ => Err(format!("Invalid AdbEventValue string: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum AdbEventType {
    EvSyn = 0,
    EvKey = 1,
    EvAbs = 3,
}

impl AdbEvent {
    fn ev_type(&self) -> AdbEventType {
        match self {
            Self::SynReport(_)
                => AdbEventType::EvSyn,
            Self::KeyW(_) | Self::KeyA(_) | Self::KeyS(_) | Self::KeyD(_) | Self::BtnTouch(_) 
                => AdbEventType::EvKey,
            Self::AbsMtTrackingId { .. } | Self::AbsMtSlot { .. }
                | Self::AbsMtPositionX(_) | Self::AbsMtPositionY(_) 
                => AdbEventType::EvAbs,
        }
    }

    pub fn to_command(&self) -> String {
        format!(
            "{} {} {}",
            self.ev_type() as u32,
            self.value(),
            self.key_value(),
        )
    }
}

#[test]
fn test() -> Result<(), Box<dyn std::error::Error>> {
    let event = AdbEvent::KeyW(AdbEventKey::Down);
    println!("{}", event.to_command());

    // let mut event_command = EventCommand::new(String::from("/dev/input/event0"));
    // event_command.append(event);
    // event_command.append(AdbEvent::SynReport(0));
    // println!("{}", event_command.to_command()?);
    Ok(())
}