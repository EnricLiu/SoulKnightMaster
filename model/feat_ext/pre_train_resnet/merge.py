import cv2
import ptlflow
import torch
from ptlflow.utils import flow_utils
from infer import infer
import numpy as np
from pathlib import Path

from tqdm import tqdm
import time

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ptlflow.get_model('liteflownet2', ckpt_path='sintel').to(DEVICE)
    
    video = cv2.VideoCapture('screen.mp4')
    if not video.isOpened():
        print("Error: 无法打开视频文件")
        exit()
    save_dir = Path('./output')
    save_dir.mkdir(parents=True, exist_ok=True)

    frames = [None, None]
    frame_counter = 1
    with tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while True:
            ret, frame_nt = video.read()
            if not ret: break
            
            pbar.update(2)

            frames[0] = frames[1]
            frames[1] = frame_nt
            if frames[0] is None: continue
            flow = infer(model, frames)
            flow_cpu = flow.cpu()
            flow_detached = flow_cpu.detach()  # Detach the tensor from the computation graph
            flow_squeezed = np.squeeze(flow_detached.numpy(),axis=(0, 1))  # Convert to NumPy array and remove dimensions of size 1
            flow_transposed = np.transpose(flow_squeezed, (1, 2, 0))
            np.save(save_dir / f'{frame_counter}.npy', flow_transposed)
            frame_counter += 1

