from typing import Tuple

import cv2 as cv
import ptlflow
from ptlflow import BaseModel
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get an optical flow model. As as example, we will use RAFT Small
# with the weights pretrained on the FlyingThings3D dataset



def infer(model: BaseModel, images, resize: tuple = None):
    # print(images)
    # A helper to manage inputs and outputs of the model
    io_adapter = IOAdapter(model, images[0].shape[:2], target_size = resize, cuda=True)

    # inputs is a dict {'images': torch.Tensor}
    # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
    # (1, 2, 3, H, W)
    inputs = io_adapter.prepare_inputs(images)
    # inputs = { k: v.to(DEVICE) for k, v in inputs.items()}

    # Forward the inputs through the model
    predictions = model(inputs)

    # The output is a dict with possibly several keys,
    # but it should always store the optical flow prediction in a key called 'flows'.
    flows = predictions['flows']
    return flows

    # flows will be a 5D tensor BNCHW.
    # This example should print a shape (1, 1, 2, H, W).
    print(flows.shape)

    # Create an RGB representation of the flow to show it on the screen
    flow_rgb = flow_utils.flow_to_rgb(flows)
    # Make it a numpy array with HWC shape
    flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
    flow_rgb_npy = flow_rgb.detach().cpu().numpy()
    # OpenCV uses BGR format
    flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)
    print(flow_bgr_npy.shape)
    # Show on the screen
    cv.imshow('image1', images[0])
    cv.imshow('image2', images[1])
    cv.imshow('flow', flow_bgr_npy)
    cv.waitKey()
    
    exit()