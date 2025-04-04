import torch
import torchaudio
import os
from models.Meeting2Conv import build_model
import json

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

print("正在加载模型...")
# 构建模型
model = build_model(config)
device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
print(f"使用设备: {device}")

# 加载模型权重
checkpoint = torch.load(config['model_path'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"模型加载成功: {config['model_path']}")

model = model.to(device)
model.eval()

# 选择具有多个说话人的音频文件进行测试
test_wav = config['test_wav']
print(f"加载测试音频: {test_wav}")

try:
    # 加载更长时间的音频以捕获多个说话人
    waveform, sample_rate = torchaudio.load(test_wav, num_frames=320000)  # 20秒 * 16000采样率
    print(f"音频形状: {waveform.shape}, 采样率: {sample_rate}")
    
    # 保留立体声以便更好区分说话人
    if waveform.size(0) > 1:
        print(f"检测到多声道音频 ({waveform.size(0)}个声道)，这有助于区分不同说话人")
        # 使用所有可用声道而不是混合为单声道
        channels_data = []
        for i in range(waveform.size(0)):
            channels_data.append(waveform[i:i+1]) 
        
        # 处理每个声道
        all_conversations = []
        for i, channel_data in enumerate(channels_data):
            print(f"处理声道 {i+1}...")
            with torch.no_grad():
                # 使用较小的段长度，以捕获更多的说话人变化
                segment_length = 32000  # 2秒
                channel_conv = model.process_meeting(channel_data.squeeze(0).to(device), segment_length=segment_length)
                all_conversations.extend(channel_conv)
                
        # 按时间戳排序
        all_conversations.sort(key=lambda x: float(x.split(']')[0][1:]))
        conversation = all_conversations
    else:
        print("单声道音频，直接处理")
        # 处理音频
        with torch.no_grad():
            # 使用较小的段长度，以捕获更多的说话人变化
            segment_length = 32000  # 2秒
            conversation = model.process_meeting(waveform.squeeze(0).to(device), segment_length=segment_length)
    
    # 输出结果
    output_file = config['output_file']
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in conversation:
            f.write(line + '\n')
            print(line)
    
    print(f"生成了 {len(conversation)} 条对话记录")
    print(f"结果已保存到: {os.path.abspath(output_file)}")
    
    # 分析说话人数量
    speakers = set()
    for line in conversation:
        speaker = line.split(']')[1].split('：')[0]
        speakers.add(speaker)
    print(f"识别到 {len(speakers)} 位说话人: {', '.join(speakers)}")
    
except Exception as e:
    print(f"处理过程中出错: {str(e)}") 