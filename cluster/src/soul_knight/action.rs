pub struct Action {
    direction: Option<f64>,
    attack: bool,
    skill:  bool,
    weapon: bool,
}

impl Action {
    pub fn new(dir: Option<f64>, attack: bool, skill: bool, weapon: bool) -> Self {
        Action {
            direction: dir,
            attack,
            skill,
            weapon,
        }
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