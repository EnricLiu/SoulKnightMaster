# SoulKnightMaster - 游戏AI训练框架

## 项目概述
SoulKnightMaster是一个用于训练游戏AI的深度学习框架，专注于《元气骑士》游戏的自动化操作。项目使用PyTorch框架，基于ResNet架构实现游戏画面到操作指令的端到端学习。

## 主要功能
- 游戏画面采集与预处理
- 操作指令录制与标注
- ResNet模型训练
- 实时推理与操作预测
- 游戏回放录制与解析

## 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (推荐)
- 其他依赖见requirement.txt

## 安装指南
1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/SoulKnightMaster.git
   cd SoulKnightMaster
   ```

2. 安装依赖：
   ```bash
   pip install -r requirement.txt
   ```

3. 配置环境变量：
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

## 使用说明

### 数据采集
1. 启动游戏回放录制：
   ```bash
   python tools/replay_recorder/replay_recorder.py
   ```

2. 进行游戏操作，系统会自动记录画面和操作指令

### 模型训练
1. 准备训练数据：
   - 将采集的数据放入`tools/replay_recorder/datasets/`目录

2. 启动训练：
   ```bash
   python model/train.py
   ```

3. 训练参数配置：
   - 修改`model/train.py`中的超参数
   - 检查点保存在`model/ckpt/`目录

### 模型推理
1. 加载训练好的模型：
   ```bash
   python model/infer.py
   ```

2. 实时推理：
   - 系统会自动加载最新检查点进行推理

## 目录结构
```
SoulKnightMaster/
├── model/                  # 模型相关代码
│   ├── ckpt/               # 模型检查点
│   ├── dataset_test.py     # 数据集处理
│   ├── infer.py            # 推理脚本
│   ├── resnet.py           # ResNet实现
│   ├── test.py             # 训练脚本
│   └── results/            # 训练结果图表
├── tools/                  # 工具集
│   ├── psd_parser/         # PSD文件解析
│   ├── recognition_tools/  # 识别工具
│   └── replay_recorder/    # 游戏回放录制
├── requirement.txt         # 依赖列表
└── README.md               # 项目文档
```

## 贡献指南
1. Fork本项目
2. 创建新分支 (`git checkout -b feature/YourFeature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送分支 (`git push origin feature/YourFeature`)
5. 创建Pull Request

## 许可证
本项目采用GPL-3.0许可证，详情见LICENSE文件。

## 模型架构
### ResNet-101 配置
- 输入: 1280x720 RGB图像
- 主干网络: ResNet-101
  - 初始卷积层: 7x7, stride=2
  - 最大池化: 3x3, stride=2
  - 残差块: [3, 4, 23, 3]
  - 输出维度: 512
- 全连接层: 512 -> 5 (对应5个输出任务)
- 激活函数: ReLU
- 归一化: BatchNorm

### 输出任务
1. 移动方向
2. 视角角度
3. 攻击状态
4. 技能使用
5. 武器选择
