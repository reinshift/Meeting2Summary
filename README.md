# Meeting2Conv

## 项目简介

Meeting2Conv 是一个会议音频（中文）转对话格式文本的混合模型系统。该系统能够接收 flac 格式的会议音频，然后自动将其转换为 `[时间戳]-[说话人id]-[说话内容]` 格式的文本。

系统综合了以下技术：
- 语音激活检测 (VAD)：使用 Silero VAD 检测语音片段
- 说话人识别：使用 ECAPA-TDNN 提取说话人声纹特征并聚类
- 语音识别 (ASR)：使用 Whisper 将语音转换为文本

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- torchaudio
- numpy
- scikit-learn
- textgrid
- whisper
- tqdm

## 安装依赖

```bash
pip install torch torchaudio numpy scikit-learn textgrid openai-whisper tqdm
```

## 数据集

本项目使用 AISHELL-4 数据集进行训练和测试。数据集应放置在 `data` 目录下，结构如下：

```
data/
├── wav/
│   └── *.flac            # 会议音频文件
└── TextGrid/
    ├── *.rttm
    └── *.TextGrid
```

## 使用方法

### 训练模型

```bash
python algorithm/main.py --mode train --config algorithm/config.json --output_dir algorithm/output
```

从检查点恢复训练：

```bash
python algorithm/main.py --mode train --config algorithm/config.json --checkpoint algorithm/output/checkpoint_epoch_10.pt
```

### 评估模型

```bash
python algorithm/main.py --mode test --config algorithm/config.json --checkpoint algorithm/output/best_model.pt
```

### 处理单个音频文件

```bash
python algorithm/main.py --mode process --config algorithm/config.json --checkpoint algorithm/output/best_model.pt --audio data/wav/example.flac
```

## 配置文件

系统的主要配置位于 `algorithm/config.json`，包括以下部分：

- **data**: 数据集相关配置
- **model**: 模型相关配置（VAD, 说话人嵌入, ASR）
- **training**: 训练相关配置（批次大小、学习率等）

## 输出示例

处理音频后将得到如下格式的文本输出：

```
[00:33]-SPK-001-零零二我是指导老师
[00:48]-SPK-001-行，好的，今天是这样的，我来听一下你们这个咱们这个
[01:20]-SPK-002-老师好，我是学生李明，我们这次的项目是...
```

## 项目结构

```
Meeting2Conv/
├── algorithm/                 
│   ├── models/                # 用于存放模型
│   │   └── Meeting2Conv.py    # 混合模型文件
│   ├── output/                # 用于保存模型以及测试生成文本
│   ├── config.json            # 配置文件
│   ├── data_utils.py          # 用于解析数据集
│   ├── trainer.py             # 用于封装训练主循环、解析数据、解析配置
│   └── main.py                # 主程序
└── data/
    ├── wav/
    │   └── *.flac             # 会议音频文件
    └── TextGrid/
        ├── *.rttm
        └── *.TextGrid
``` 