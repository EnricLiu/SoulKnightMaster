import cv2
import json
import numpy as np
import polars as pl
from PIL import Image
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
    
    def is_move(self) -> bool:
        return self._get_state_by_name("move")
    def is_attack(self) -> bool:
        return self._get_state_by_name("attack")
    def is_skill(self) -> bool:
        return self._get_state_by_name("skill")
    def is_weapon(self) -> bool:
        return self._get_state_by_name("weapon")
    def get_move_angle(self) -> float:
        if not self.is_move(): return None
        if not self._move_angle: return 0.0
        return float(self._move_angle)
    
    def to_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "move":     [self.is_move()],
            "angle":    [self.get_move_angle()],
            "attack":   [self.is_attack()],
            "skill":    [self.is_skill()],
            "weapon":   [self.is_weapon()],
        })
    
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


class SoulKnightReplay:
    def __init__(self, replay_path: Path):
        self._actions: dict[int, SoulKnightAction] = {} # {time_us: action}
        self._time_us_list: list[int] = []
        self._start_time: int = None
        self._end_time  : int = None
        
        self._replay_path = replay_path
        self._action_path, self._screen_path = self._extract_paths(replay_path)
        
        self._video = None
        self._video_width  = None
        self._video_height = None
        self._frame_duration_us = None
        
        self._is_loaded = False

    def load(self):
        with open(self._action_path, "r", encoding="utf-8") as f:
            last_time_us = -1
            for line in f.readlines():
                line = line.strip()
                if line == "": continue
                
                line = line.split(" | ")
                if len(line) != 2:
                    raise ValueError(f"ReplayViewer: Broken Actions: Invalid line {line}")
                
                time_us, action_str = line
                time_us = int(time_us)
                if time_us < last_time_us: continue
                last_time_us = time_us
                
                maybe_action = json.loads(action_str)
                action = SoulKnightAction(maybe_action)
                
                self._actions[time_us] = action
                
        self._time_us_list  = list(self._actions.keys())
        
        self._start_time    = self._time_us_list[0]
        self._end_time      = self._time_us_list[-1]
        
        #####################################################################
        
        video = cv2.VideoCapture(str(replay.screen_path()))
        self._frame_duration_us = int(1000000 / video.get(cv2.CAP_PROP_FPS))
        self._frame_num     = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_width   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._video = video
        
        self._is_loaded = True
        return self
    
    def get_frame_num(self) -> int:
        return self._frame_num

    def gen_dataset(self):
        cnt = -1
        if not self._is_loaded:
            raise ValueError("ReplayViewer: Replay not loaded.")
        while True:
            ret, frame = self._video.read()
            if not ret: break
            cnt += 1
            curr_time_us = self._frame_duration_us * cnt
            
            if curr_time_us > self._end_time: break
            if curr_time_us > self._start_time:
                action = self.get_action_by_time(self._frame_duration_us * cnt)
            else:
                action = self.get_action_by_time(self._start_time)
                
            yield cnt, frame, action
            
    def get_action_by_time(self, time_us: int):
        if self._is_loaded == False:
            raise ValueError("ReplayViewer: Replay not loaded.")
        
        head = 0
        tail = len(self._actions) - 1
        
        while head < tail:
            mid = (head + tail) // 2
            mid_val = self._time_us_list[mid]
            # print(head, tail, mid)
            if mid_val == time_us: 
                return self._actions[mid_val]
            
            if time_us < mid_val:
                tail = mid
            else:
                head = mid
                
            if head +1 == tail:
                return self._actions[self._time_us_list[head]]

        return None
        
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def screen_path(self) -> Path:
        return self._screen_path
    
    def action_path(self) -> Path:
        return self._action_path
    
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
    replay = SoulKnightReplay(Path("./out/record_20241223-18_48_58-_out"))
    replay.load()

    # print(replay.get_action_by_time(94741134))
    out_path = Path("./datasets/record_20241223-18_48_58-_out")
    out_path.mkdir(parents=True, exist_ok=True)
    
    df = pl.DataFrame(schema = {
        "img_dir": pl.Utf8,
        "move": pl.Boolean,
        "angle": pl.Float64,
        "attack": pl.Boolean,
        "skill": pl.Boolean,
        "weapon": pl.Boolean
    })
    for frame_idx, img, action in tqdm(replay.gen_dataset(), total=replay.get_frame_num()):
        img_dir = f"{frame_idx+1}.npy"
        img_dir_df = pl.DataFrame({"img_dir": [str(img_dir)]})
        df = df.vstack(img_dir_df.hstack(action.to_df()))
        img_dir = out_path / img_dir
        img_result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_result = np.asarray(img_result)
        np.save(img_dir, img_result)
        
    df.write_csv(out_path / "dataset.csv")

