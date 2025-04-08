# Meeting2Conv

## 项目简介

本项目旨在开发一个会议音频（中文）转对话格式文本的混合模型，包含声纹识别区分说话人、语音识别文本等技术，输入一个flac格式的会议音频，将转为\[时间戳\]-\[说话人id\]-\[说话内容\]格式的文本。

## 数据集介绍

采用AISHELL-4数据集，构成为flac音频文件以及rttm、TextGrid格式文件。其中，rttm格式部分片段如下：

```bash
SPEAKER 20200706_L_R001S01C01 1 33.6354 2.3550 <NA> <NA> 001-M <NA> <NA>
SPEAKER 20200706_L_R001S01C01 1 48.1404 8.6550 <NA> <NA> 001-M <NA> <NA>
SPEAKER 20200706_L_R001S01C01 1 58.5454 0.4895 <NA> <NA> 001-M <NA> <NA>
```

这里每一行信息具体意义为：说话人、音频文件名、通道数（单通道为1）、说话时间（单位秒，下同）、持续时间、\<NA\>占位符、说话人ID以及性别（M为男）。而TextGrid格式文件部分片段如下：

```bash
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 2011.088          # 音频总时长（秒）
tiers? <exists> 
size = 7                 # 总共有7个标注层（tiers）
item []: 
    item [1]:
        class = "IntervalTier"   # 区间层类型
        name = "001-M"           # 说话人ID（对应RTTM中的说话人）
        xmin = 0 
        xmax = 2011.088          # 该层覆盖的时间范围
        intervals: size = 313    # 该层有313个时间区间
        intervals [1]:
            xmin = 0 
            xmax = 33.63536 
            text = ""            # 空文本可能表示静音或非说话段
        intervals [2]:
            xmin = 33.63536 
            xmax = 35.99036 
            text = "零零二我是指导老师"  # 标注的文本内容
        intervals [3]:
            xmin = 35.99036 
            xmax = 48.14036 
            text = ""
        intervals [4]:
            xmin = 48.14036 
            xmax = 56.79536 
            text = "行，好的，今天<sil>是这样的，我来听一下你们嗯这个<sil>咱们这个"
```

## 开发思路

首先创建项目路径如下：

```bash
ATTTN-XDU/                     # 项目文件夹
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

对于Meeting2Conv.py，思路是用VAD检测语音激活片段，ECAPA-TDNN进行声纹嵌入说话人识别，ASR进行语音转中文文本，主要训练ECAPA-TDNN模块，VAD和ASR用本地缓存模型；

训练ECAPA-TDNN时需要注意，AISHELL-4的数据是多通道，而ECAPA-TDNN的输入是mel频谱图，因此需要将多通道数据转换为单通道数据，具体做法是取每个通道的均值，然后作为输入；

data_utils.py用于解析数据集，将数据集要用的信息（要和模型正确交互）提取出来；

trainer.py用于封装数据解析、config解析以及训练主循环的代码，这里需要有logger以进度条的形式打印当前训练的轮数、loss、以及预计剩余时间（不需要保存日志，只要打印就行）；

main.py为主程序，里面用一个run()方法来运行整个模型，并且支持切换训练/测试生成文本模式。


