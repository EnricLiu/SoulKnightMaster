import argparse
from pathlib import Path

import cv2
import ptlflow
import torch
from ptlflow.utils import flow_utils
from infer import infer
import numpy as np
from pathlib import Path

from tqdm import tqdm
import time

def make_dataset(model, video_path: Path, out_path: Path) -> tuple[Path, int]:
    print(str(video_path.absolute()))
    video = cv2.VideoCapture(str(video_path.absolute()))
    if not video.isOpened():
        raise Exception("Cannot open video")
    out_path.mkdir(parents=True, exist_ok=True)

    frames = [None, None]
    frame_counter = 1
    with tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while True:
            ret, frame_nt = video.read()
            if not ret: break
            pbar.update(1)
            frames[0] = frames[1]
            frames[1] = frame_nt
            if frames[0] is None: continue
            flow = infer(model, frames)
            flow_cpu = flow.cpu()
            flow_detached = flow_cpu.detach()  # Detach the tensor from the computation graph
            flow_squeezed = np.squeeze(flow_detached.numpy(),axis=(0, 1))  # Convert to NumPy array and remove dimensions of size 1
            flow_transposed = np.transpose(flow_squeezed, (1, 2, 0))
            np.save(out_path / f'{frame_counter}.npy', flow_transposed)
            frame_counter += 1
    return out_path, frame_counter
    

# if __name__ == '__main__':
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ptlflow.get_model('liteflownet2', ckpt_path='sintel').to(DEVICE)
    
#     make_dataset(model, Path('screen.mp4'), Path('./output'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make dataset from replay file")
    parser.add_argument("--replay", type=str, required=True, nargs='+', help="Path(s) of the replay file(s)")
    parser.add_argument("--out", type=str, required=True, default="./datasets", help="Path of the output folder")
    
    args = parser.parse_args()
    
    try:
        replays = map(lambda x: Path(x), args.replay)
        out_path = Path(args.out)
    except Exception as e:
        print(f"Invalid path! {e}")
    
    import torch
    import ptlflow
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ptlflow.get_model('liteflownet2', ckpt_path='sintel').to(DEVICE)
    
    for replay in replays:
        if not replay.is_dir():
            print(f"Skip replay {replay}: file does not exist!")
            continue
        out = out_path / replay.name
        try:
            _ = make_dataset(model, replay / "screen.mp4", out)
        except Exception as e:
            print(f"Failed to make dataset from {replay}: {e}")
            
    print("Done!")