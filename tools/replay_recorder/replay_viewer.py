import cv2
import time
import json
from tqdm import tqdm
from pathlib import Path

class SoulKnightAction:
    _ACTION_MAP = {
        "move":     0,
        "attack":   1,
        "skill":    2,
        "weapon":   3
    }
    def __init__(self, action: list):
        self._states, self._move_angle = SoulKnightAction._parse_action(action)
    
    @staticmethod
    def _parse_action(maygbe_action: list):
        try:
            results  = {}
            movement = maygbe_action["movement"]
            move_angle = movement["direction"]
            
            results["move"]     = movement["is_moving"]
            results["attack"]   = maygbe_action["attack"]
            results["skill"]    = maygbe_action["skill"]
            results["weapon"]   = maygbe_action["weapon_switching"]
            
        except Exception as e:
            raise ValueError(f"SoulKnightAction: Invalid action: {maygbe_action}")
        
        states = 0b00000000
        for k, v in results.items():
            shift_num = SoulKnightAction._ACTION_MAP[k]
            if shift_num is None: continue
            if v: states = states | (0x01 << shift_num)
            
        return states, move_angle
        
    def is_move(self) -> bool:
        return self._get_state_by_name("move")
    def is_attack(self) -> bool:
        return self._get_state_by_name("attack")
    def is_skill(self) -> bool:
        return self._get_state_by_name("skill")
    def is_weapon(self) -> bool:
        return self._get_state_by_name("weapon")
    def get_move_angle(self) -> bool:
        return self._move_angle if self.is_move() else None
        
    def _get_state_by_name(self, action_name) -> bool:
        bit_mask = self._ACTION_MAP[action_name]
        ret = (self._states & (0x01 << bit_mask)) != 0
        return ret
    
    def __str__(self) -> str:
        d = {
            "move": self.is_move(),
            "angle": self.get_move_angle(),
            "attack": self.is_attack(),
            "skill": self.is_skill(),
            "weapon": self.is_weapon()
        }
        return json.dumps(d)


class SoulKnightReplayViewer:
    def __init__(self, replay_path: Path):
        self._actions = {} # {time_us: action}
        self._replay_path = replay_path
        self._action_path, self._screen_path = self._extract_paths(replay_path)
        
    def parse_actions(self):
        with open(self._action_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "": continue
                
                line = line.split(" | ")
                if len(line) != 2:
                    raise ValueError(f"ReplayViewer: Broken Actions: Invalid line {line}")
                
                time_us, action_str = line
                maybe_action = json.loads(action_str)
                action = SoulKnightAction(maybe_action)
                
                self._actions[time_us] = action
        
    def _extract_paths(self, replay_path: Path):
        if not replay_path.is_dir():
            raise ValueError(f"ReplayViewer: {replay_path} not a exist dir.")
        
        action_path = replay_path / "action.txt"
        screen_path = replay_path / "screen.mp4"
        
        if not action_path.is_file():
            raise ValueError(f"ReplayViewer: {action_path} not exist.")
        if not screen_path.is_file():
            raise ValueError(f"ReplayViewer: {screen_path} not exist.")
        
        return action_path, screen_path

if __name__ == "__main__":
    viewer = SoulKnightReplayViewer(Path("./out/record_20241223-18_48_58-_out"))
    viewer.parse_actions()
    k, v = list(viewer._actions.items())[1]
    print(v)
    