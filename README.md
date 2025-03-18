# SoulKnightMaster

SoulKnightMaster is a project that involves training and evaluating machine learning models for the game Soul Knight. This project includes various scripts and tools for data preprocessing, model training, and evaluation.

## Project Structure

The project is organized into the following directories:

- `model/feat_ext/`: Contains feature extraction models and related scripts.
  - `pre_train_resnet/`: Scripts for pre-training ResNet models.
  - `efficient_GRU/`: Scripts for training and evaluating Efficient GRU models.
- `train/`: Contains training scripts and utilities.
  - `httb/`: HTTP client-related scripts.
  - `utils/`: Utility scripts for training.
- `tools/replay_recorder/`: Tools for recording and processing game replays.
- `cluster/`: Contains scripts for managing and monitoring a cluster of nodes.

## Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch
- torchvision
- polars
- tqdm
- wandb
- OpenCV
- PIL
- numpy
- scikit-learn
- gym
- stable-baselines3

