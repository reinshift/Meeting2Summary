# 开发流程

## Meeting2Conv

首先开发一个会议音频（中文）转对话格式文本的混合模型，包含声纹识别区分说话人、语音识别文本等技术，输入一个flac格式的会议音频，将转为`[时间戳]-[说话人id]-[说话内容]`格式的文本。

## 数据集介绍
采用[AISHELL-4](https://aishelltech.com/aishell_4)数据集，构成为flac音频文件以及rttm、TextGrid格式文件。其中，rttm格式部分片段如下：

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