import numpy as np
import json
import math
from math import pi as PI


class Action:
    SN = 0
    
    def __init__(self, angle: float|None, attack: bool, skill: bool, weapon:bool):
        Action.SN += 1
        
        self._sn     = Action.SN
        self._angle  = (angle + PI) % (2 * PI) - PI if angle is not None else None
        self._attack = attack
        self._skill  = skill
        self._weapon = weapon
        
    def from_onehot(raw_action: np.ndarray) -> 'Action':
        raw_action = raw_action.reshape((4,))
        if raw_action.shape != (4,):
            raise ValueError("raw_action must be a 1-D array with shape (4,)")
        raw_action = raw_action.tolist()
        return Action(
            angle  = (math.pi * (raw_action[1] - 128) / 128) if raw_action[0] == 1 else None,
            attack = raw_action[2] == 1,
            skill  = raw_action[3] == 1,
            weapon = False
        )
        
    def from_raw(raw_action: np.ndarray) -> 'Action':
        raw_action = raw_action.reshape((4,))
        if raw_action.shape != (4,):
            raise ValueError("raw_action must be a 1-D array with shape (4,)")
        raw_action = raw_action.tolist()
        return Action(
            angle  = raw_action[1] * math.pi if raw_action[0] > 0.5 else None,
            attack = raw_action[2] > 0.5,
            skill  = raw_action[3] > 0.5,
            weapon = False
        )
    
    def __dict__(self) -> dict[str, any]:
        return {
            "sn":       self._sn,
            "weapon":   self._weapon,
            "skill":    self._skill,
            "attack":   self._attack,
            "direction": self._angle,
        }
    
    def json(self) -> str:
        ret = json.dumps(self.__dict__())
        return ret
    
    def __str__(self):
        return self.json()


if __name__ == "__main__":
    arr = np.asarray([[[[[[[[1,2,3,4]]]]]]]], dtype=np.float32)
    action = Action.from_raw(arr)
    print(action.json())