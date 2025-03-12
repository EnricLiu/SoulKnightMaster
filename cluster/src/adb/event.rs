use std::hash::Hash;
use std::str::FromStr;
use serde::{Deserialize, Serialize};

// from /system/usr/keylayout/qwerty.kl
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum Key {
    Back = 1,
    Key1 = 2,
    Key2 = 3,
    Key3 = 4,
    Key4 = 5,
    Key5 = 6,
    Key6 = 7,
    Key7 = 8,
    Key8 = 9,
    Key9 = 10,
    Key0 = 11,
    Minus   = 12,
    Equal   = 13,
    Del     = 14,
    Tab     = 15,
    Q = 16,
    W = 17,
    E = 18,
    R = 19,
    T = 20,
    Y = 21,
    U = 22,
    I = 23,
    O = 24,
    P = 25,
    LeftBracket     = 26,
    RightBracket    = 27,
    Enter           = 28,
    A = 30,
    S = 31,
    D = 32,
    F = 33,
    G = 34,
    H = 35,
    J = 36,
    K = 37,
    L = 38,
    Semicolon   = 39,
    Apostrophe  = 40,
    Backslash   = 43,
    Z = 44,
    X = 45,
    C = 46,
    V = 47,
    B = 48,
    N = 49,
    M = 50,
    Comma   = 51,
    Period  = 52,
    Slash   = 53,

    Menu        = 59,
    SoftLeft    = 60,
    EndCall     = 62,
    Call        = 61,
    ShiftLeft   = 42,
    ShiftRight  = 54,
    AltLeft     = 56,
    Space       = 57,
    // Menu        = 68,
    AltRight    = 100,
    Home        = 102,
    DPadUp      = 103,
    DPadLeft    = 105,
    DPadRight   = 106,
    // EndCall     = 107,
    DPadDown    = 108,
    VolumeDown  = 114,
    VolumeUp    = 115,
    Power   = 116,
    Search  = 127,
    // Menu    = 139,
    // Back    = 158,
    Camera  = 212,
    // Search  = 217,
    Star    = 227,
    Pound   = 228,
    // Menu    = 229,
    SoftRight   = 230,
    // Call     = 231,
    DPadCenter  = 232,


    Sleep       = 142,
    Explorer    = 150,
    Envelope    = 155,
    MediaClose  = 160,
    MediaEject  = 161,
    MediaNext   = 163,
    MediaPlayPause  = 164,
    MediaPrevious   = 165,
    MediaStop   = 166,
    MediaRecord = 167,
    MediaRewind = 168,
    At          = 215,
    Grave       = 399,
    AppSwitch   = 580,
    StemPrimary = 581,
    Stem1 = 582,
    Stem2 = 583,
    Stem3 = 584,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum KeyValue {
    Up = 0,
    Down = 1,
}

impl TryFrom<i32> for KeyValue {
    type Error = String;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Up),
            1 => Ok(Self::Down),
            _ => Err(format!("Invalid AdbEventValue value: {}", value)),
        }
    }
}
impl FromStr for KeyValue {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum AdbEvent {
    AbsMtTrackingId { slot_id: u32 } = 0x0039,
    AbsMtSlot { slot_id: u32 } = 0x002f,

    AbsMtPositionX(u32) = 0x0035,
    AbsMtPositionY(u32) = 0x0036,

    Key(Key, KeyValue),
    BtnTouch(KeyValue)  = 0x014A,

    SynReport(u32)   = 0x0000,
}

impl AdbEvent {
    pub(crate) fn value(&self) -> u32 {
        match self {
            Self::Key(k, _) => *k as u32,
            _ => unsafe { std::mem::transmute::<Self, [u32;3]>(*self)[0] }
        }
    }

    pub(crate) fn key_value(&self) -> u32 {
        match self {
            AdbEvent::AbsMtTrackingId { slot_id } => *slot_id,
            AdbEvent::AbsMtSlot { slot_id } => *slot_id,
            
            AdbEvent::AbsMtPositionX(x) => *x,
            AdbEvent::AbsMtPositionY(y) => *y,
            
            AdbEvent::Key(_, v) => *v as u32,
            AdbEvent::BtnTouch(v) => *v as u32,
            
            AdbEvent::SynReport(v) => *v,
        }   
    }

    pub(crate) fn ev_type(&self) -> AdbEventType {
        match self {
            Self::SynReport(_) => AdbEventType::EvSyn,
            Self::Key(_, _)
            | Self::BtnTouch(_) => AdbEventType::EvKey,
            Self::AbsMtTrackingId { .. }
            | Self::AbsMtSlot { .. }
            | Self::AbsMtPositionX(_)
            | Self::AbsMtPositionY(_) => AdbEventType::EvAbs,
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
fn event() -> Result<(), Box<dyn std::error::Error>> {
    // let event = AdbEvent::Key(Key::W, KeyValue::Down);
    let event = AdbEvent::AbsMtPositionY(123);
    println!("{:?}", event.to_command());

    // let mut event_command = EventCommand::new(String::from("/dev/input/event0"));
    // event_command.append(event);
    // event_command.append(AdbEvent::SynReport(0));
    // println!("{}", event_command.to_command()?);
    Ok(())
}