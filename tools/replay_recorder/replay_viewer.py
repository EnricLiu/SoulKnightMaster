from os import cpu_count
import time
from pathlib import Path

import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils import get_region, CircleRegion, Position
from replay import SoulKnightReplay, SoulKnightAction

def view_replay(replay_path: Path, out_path: Path, config: list, buf_size: int=64, max_workers: int=cpu_count()) -> None:
    print("[INFO]  ReplayViewer: Loading replay...")
    regions_config = config["ActionListenerConfig"]["regions"]
    
    regions = {k: get_region(**regions_config[k]) for k in ["joystick", "btn_atk", "btn_skill", "btn_weapon"]}
    regions_center = {k: v.center().swap().set_y(lambda r: 720 - r.y) for k, v in regions.items()}

    def draw_action(target_frame, action: SoulKnightAction, color: tuple=(255, 0, 0)):
        def _draw_attack(p: Position, size: int = 10):
            cv2.circle(target_frame, p.round_to_tuple(), size, color, size)
        def _draw_skill(p: Position, size: int = 10):
            cv2.circle(target_frame, p.round_to_tuple(), size, color, size)
        def _draw_weapon(p: Position, size: int = 10):
            cv2.circle(target_frame, p.round_to_tuple(), size, color, size)
        def _draw_movement(p: Position, radius: float, angle: float, size: int = 10):
            draw_center = p.offset_polar(radius, angle)
            cv2.circle(target_frame, draw_center.round_to_tuple(), size, color, size)
            
        if action.is_attack():
            _draw_attack(regions_center["btn_atk"])
        if action.is_skill():
            _draw_skill(regions_center["btn_skill"])
        if action.is_weapon():
            _draw_weapon(regions_center["btn_weapon"])
        if action.is_move():
            _draw_movement(regions_center["joystick"], 60, action.get_move_angle())
            
        return target_frame

    replay = SoulKnightReplay(replay_path).load()
    
    # cv2 extract all the frames
    video = cv2.VideoCapture(str(replay.screen_path()))
    if not video.isOpened():
        raise Exception("Cannot open replay")
    
    fps  = video.get(cv2.CAP_PROP_FPS)
    fnum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\t fps: {fps}, width: {width}, height: {height}")
    frame_duration_us = int(1000000 / fps)

    print("[INFO]  ReplayViewer: replay loaded, extracting frames...")
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / "overlapped.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    def draw_worker(frame_chunk) -> None:
        frame_idx, frame = frame_chunk
        action = replay.get_action_by_time(frame_idx * frame_duration_us)
        frame = draw_action(frame, action)
        return {"idx": frame_idx, "frame": frame}
    
    with tqdm(total=fnum) as pbar:
        f_cnt = 0
        if max_workers == 0:
            ret, frame = video.read()
            action = replay.get_action_by_time(f_cnt * frame_duration_us)
            frame = draw_action(frame, action)
            out.write(frame)
            pbar.update(1)

        else:
            while f_cnt < fnum:
                frame_chunks = []
                for _ in range(buf_size):
                    ret, frame = video.read()
                    if not ret: 
                        if f_cnt < fnum: raise Exception("Replay file is corrupted")
                        break
                    frame_chunks.append((f_cnt, frame))
                    f_cnt += 1

                with ThreadPoolExecutor(max_workers) as executor:
                    results = executor.map(draw_worker, frame_chunks)
                results = sorted(results, key=lambda x: x["idx"])
                for result in results:
                    out.write(result["frame"])
                    pbar.update(1)

    video.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[INFO]  ReplayViewer: Done!\n\tOutput Path: {out_path.absolute()}")

if __name__ == "__main__":

    import json
    config = json.load(open("config.json"))
    view_replay(Path("./out/test/record_20241231-17_18_24-_out"), Path("./out/test/record_20241231-17_18_24-_out"), config, 1024, 8)
    