## 项目简介

Meeting2Summary 是一个将会议音频（中文）转换为会议纪要的混合模型系统。该系统能够接收 flac 格式的会议音频，然后自动将其转换为 `[时间戳]-[说话人id]-[说话内容]` 格式的文本，并据此最终生成会议纪要。

系统综合了包括但不限于以下技术：
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