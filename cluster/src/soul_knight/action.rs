use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    sn: u64,
    direction: Option<f64>,
    attack: bool,
    skill:  bool,
    weapon: bool,
}

impl Action {
    pub fn new(sn: u64, dir: Option<f64>, attack: bool, skill: bool, weapon: bool) -> Self {
        Action {
            sn,
            direction: dir,
            attack,
            skill,
            weapon,
        }
    }
    pub fn sn(&self) -> u64 {
        self.sn
    }
    pub fn direction(&self) -> Option<f64> {
        self.direction
    }
    pub fn attack(&self) -> bool {
        self.attack
    }
    pub fn skill(&self) -> bool {
        self.skill
    }
    pub fn weapon(&self) -> bool {
        self.weapon
    }
}