import time
from pathlib import Path

import cv2
from tqdm import tqdm

from utils import get_region, CircleRegion
from replay import SoulKnightReplay, SoulKnightAction

def view_replay(replay_path: Path, overlap_out: Path, config: list) -> None:
    def draw_action(target_frame, action: SoulKnightAction, color: tuple=(0, 255, 0)):
        def _draw_attack():
            cv2.circle(target_frame, btn_atk_region.center().round_to_tuple(), 2, color, 2)
        def _draw_skill():
            cv2.circle(target_frame, btn_skill_region.center().round_to_tuple(), 2, color, 2)
        def _draw_weapon():
            cv2.circle(target_frame, btn_weapon_region.center().round_to_tuple(), 2, color, 2)
        def _draw_movement(angle: float):
            draw_center = joystick_region.center().offset_polar(joystick_region.radius(), angle)
            cv2.circle(target_frame, draw_center.round_to_tuple(), 2, color, 2)
            
        if action.is_attack():
            _draw_attack()
        elif action.is_skill():
            _draw_skill()
        elif action.is_weapon():
            _draw_weapon()
        elif action.is_move():
            _draw_movement(action.get_move_angle())
            
        return target_frame
    
    
    regions = config["ActionListenerConfig"]["regions"]
    joystick_region   = get_region(**regions[  "joystick"])
    btn_atk_region    = get_region(**regions[   "btn_atk"])
    btn_skill_region  = get_region(**regions[ "btn_skill"])
    btn_weapon_region = get_region(**regions["btn_weapon"])
    
    replay = SoulKnightReplay(replay_path).load()
    
    # cv2 extract all the frames
    video = cv2.VideoCapture(str(replay.screen_path()))
    fps = video.get(cv2.CAP_PROP_FPS)
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"fps: {fps}, width: {width}, height: {height}")
    frame_duration_us = int(1000000 / fps)
    
    frames = []
    while True:
        ret, frame = video.read()
        if not ret: break
        frames.append(frame)
        
    result_frames = []
    for id, frame in tqdm(enumerate(frames)):
        action = replay.get_action_by_time(id * frame_duration_us)
        if action is None:
            overlapped_frame = frame
        else: 
            overlapped_frame = draw_action(frame, action)
            cv2.imshow("frame", overlapped_frame)
            time.sleep(10086)
            return
        result_frames.append(overlapped_frame)
        
    # reform into mp4
    cv2.VideoWriter(str(overlap_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)).write(cv2.hconcat(result_frames))    

if __name__ == "__main__":
    import json
    config = json.load(open("config.json"))
    view_replay(Path("./out/record_20241223-18_48_58-_out"), Path("./out/record_20241223-18_48_58-_out"), config)
    