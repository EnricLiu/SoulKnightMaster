# SoulKnightMaster

## Overview
SoulKnightMaster is a pure vision-based reinforcement learning project specializing in the game *Soul Knight*, with Proximal Policy Optimization (PPO) as its core algorithmic foundation. The system leverages PPO for game agent training while integrating Rust-based cluster management and Python machine learning pipelines to enable efficient distributed training and game data processing.

## Directory Structure
```
├── cluster/       Rust-based cluster management system
├── interface/     Python API interfaces and controllers
├── model/         Machine learning models (ResNet, GRU, etc.)
├── tools/         Utility tools for data processing and analysis
├── train/         Training frameworks and utilities
└── LICENSE        MIT license file
```

## Key Components
### Cluster Module (`cluster/`)
- **ADB Client**: Android Debug Bridge interface for device communication
- **Node Management**: Real-time status monitoring and health checks
- **API Server**: RESTful service built with Axum framework
- **Config**: JSON-based configuration system (`configs/*.json`)

### Machine Learning Models (`model/`)
- **Feature Extraction**: 
  - Efficient GRU for sequence modeling
  - Pre-trained ResNet for image feature extraction
- **Training Frameworks**: 
  - PPO reinforcement learning implementation
  - Full-frame training modules

### Tools Suite (`tools/`)
- **Replay Recorder**: Game session capture and analysis
- **PSD Parser**: Photoshop file processing for game assets
- **Recognition Tools**: Image/video analysis utilities

## License
MIT License - see [LICENSE](LICENSE) file for details