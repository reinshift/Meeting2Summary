import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import json
import whisper
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    """音频片段数据类"""
    start: float  # 开始时间 (秒)
    end: float    # 结束时间 (秒)
    audio: torch.Tensor  # 音频数据
    sr: int       # 采样率
    speaker_id: Optional[str] = None  # 说话人ID
    text: Optional[str] = None  # 文本内容

class VAD(nn.Module):
    """语音活动检测模块，使用预训练模型"""
    
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"正在初始化VAD模块，使用设备: {self.device}")
        
        # 使用silero-vad (小型轻量级VAD模型)
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # 导入工具函数
        self.get_speech_timestamps = utils[0]
        self.save_audio = utils[1]
        self.read_audio = utils[2]
        self.vad_collector = utils[3]
        
        logger.info("VAD模块初始化完成")
    
    def forward(self, audio_path: str, min_speech_duration_ms: int = 250, threshold: float = 0.5) -> List[AudioSegment]:
        """检测音频中的语音段"""
        logger.info(f"正在处理音频文件: {audio_path}")
        
        # 读取音频文件
        audio_tensor, sr = torchaudio.load(audio_path)
        
        # 如果是立体声，转换为单声道
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        
        # 如果采样率不是16kHz，进行重采样
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
            sr = 16000
        
        # 将音频移动到设备上
        audio_tensor = audio_tensor.to(self.device)
        
        # 获取语音时间戳
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor[0],
            self.model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            return_seconds=False
        )
        
        # 创建AudioSegment对象列表
        segments = []
        for ts in speech_timestamps:
            start_sample, end_sample = ts['start'], ts['end']
            start_time = start_sample / sr
            end_time = end_sample / sr
            
            # 提取该段音频
            segment_audio = audio_tensor[:, start_sample:end_sample]
            
            segments.append(AudioSegment(
                start=start_time,
                end=end_time,
                audio=segment_audio,
                sr=sr
            ))
        
        logger.info(f"检测到 {len(segments)} 个语音段")
        return segments


class ECAPA_TDNN(nn.Module):
    """基于ECAPA-TDNN架构的说话人识别模型"""
    
    def __init__(self, input_size=80, channels=512, emb_dim=192):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_size, channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        
        # SE-Res2Block 1
        self.res1 = SERes2Block(channels, channels, kernel_size=3, stride=1, padding=1)
        
        # SE-Res2Block 2
        self.res2 = SERes2Block(channels, channels, kernel_size=3, stride=1, padding=1)
        
        # SE-Res2Block 3
        self.res3 = SERes2Block(channels, channels, kernel_size=3, stride=1, padding=1)
        
        # Attentive Statistics Pooling
        self.asp = AttentiveStatsPool(channels, attention_channels=128)
        
        # Final embedding layer
        self.fc = nn.Linear(channels * 2, emb_dim)
        self.bn2 = nn.BatchNorm1d(emb_dim)
        
    def forward(self, x):
        # 输入x可能的形状:
        # 1. [batch, time, freq]
        # 2. [batch, time, channel, freq]
        
        # 检查输入维度，如果是4D，则合并通道维度（取平均值）
        if x.dim() == 4:
            # x形状为[batch, time, channel, freq]
            # 在通道维度上取平均值
            x = torch.mean(x, dim=2)  # 结果形状为[batch, time, freq]
        
        # 转换为 [batch, freq, time] 以适配卷积层
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = self.asp(x)
        x = self.fc(x)
        x = self.bn2(x)
        
        # L2 归一化，添加数值稳定性保护
        eps = 1e-8
        norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=eps)
        x = x / norm
        
        return x


class SERes2Block(nn.Module):
    """具有Squeeze-Excitation的Res2Net块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale=8, se_ratio=16):
        super().__init__()
        
        self.scale = scale
        width = in_channels // scale
        
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride, padding=padding))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.se = SE_Block(out_channels, ratio=se_ratio)
        
        self.relu = nn.ReLU()
        self.width = width
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](sp)
            sp = self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        out = torch.cat((out, spx[self.scale-1]), 1)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.se(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class SE_Block(nn.Module):
    """压缩激励(Squeeze-Excitation)模块"""
    
    def __init__(self, channel, ratio=16):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class AttentiveStatsPool(nn.Module):
    """注意力统计池化层"""
    
    def __init__(self, in_dim, attention_channels=128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):
        # x: [batch, channels, time]
        eps = 1e-5  # 添加一个小的常数以确保数值稳定性
        
        attention_weights = self.attention(x)
        
        # 应用注意力权重
        weighted_x = x * attention_weights
        
        # 计算统计量
        mean = torch.sum(weighted_x, dim=2, keepdim=True)
        
        # 使用更稳定的方法计算标准差
        # 原始: std = torch.sqrt(torch.sum(weighted_x ** 2, dim=2, keepdim=True) - mean ** 2)
        # 问题: 当 mean 很大时，mean^2 可能会导致数值不稳定
        
        # 更稳定的计算方式:
        var = torch.sum(weighted_x ** 2, dim=2, keepdim=True) - mean ** 2
        # 确保方差不为负（由于数值精度问题可能出现）
        var = torch.clamp(var, min=eps)
        std = torch.sqrt(var)
        
        # 拼接均值和标准差
        pooled = torch.cat([mean, std], dim=1)
        pooled = pooled.view(pooled.shape[0], -1)
        
        return pooled


class SpeakerRecognition(nn.Module):
    """说话人识别模块"""
    
    def __init__(self, device=None, threshold=0.5):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"正在初始化说话人识别模块，使用设备: {self.device}")
        
        # 特征提取
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            n_mels=80
        )
        
        # ECAPA-TDNN模型
        self.model = ECAPA_TDNN(input_size=80, channels=512, emb_dim=192)
        self.model.to(self.device)
        
        # 说话人嵌入数据库
        self.speaker_embeddings = {}
        self.threshold = threshold
        
        logger.info("说话人识别模块初始化完成")
    
    def _extract_features(self, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """从音频中提取梅尔频谱特征"""
        # 确保音频是单通道的
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # 重采样到16kHz (如果需要)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)
        
        # 提取梅尔频谱图
        with torch.no_grad():
            mel_spec = self.feature_extractor(audio)
            
        # 对数变换
        mel_spec = torch.log(mel_spec + 1e-6)
        
        # 标准化
        mean = torch.mean(mel_spec, dim=2, keepdim=True)
        std = torch.std(mel_spec, dim=2, keepdim=True)
        mel_spec = (mel_spec - mean) / (std + 1e-6)
        
        # 转置为 [time, freq]，去掉批次和通道维度
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)  # [time, freq]
        
        return mel_spec
    
    def forward(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """为语音片段分配说话人ID"""
        segments_with_speakers = []
        
        for segment in segments:
            # 提取特征
            features = self._extract_features(segment.audio, segment.sr).to(self.device)
            
            # 提取说话人嵌入
            with torch.no_grad():
                embedding = self.model(features).cpu().numpy()
            
            # 使用余弦相似度查找最匹配的说话人
            speaker_id = self._find_best_match(embedding)
            
            # 如果是新说话人，则注册
            if speaker_id is None:
                speaker_id = f"SPK_{len(self.speaker_embeddings) + 1:03d}"
                self.speaker_embeddings[speaker_id] = embedding
                logger.info(f"注册新说话人: {speaker_id}")
            
            # 更新片段的说话人ID
            segment.speaker_id = speaker_id
            segments_with_speakers.append(segment)
        
        return segments_with_speakers
    
    def _find_best_match(self, embedding: np.ndarray) -> Optional[str]:
        """找到最匹配的说话人ID"""
        if not self.speaker_embeddings:
            return None
        
        max_similarity = -1
        best_speaker = None
        
        for speaker_id, stored_embedding in self.speaker_embeddings.items():
            # 计算余弦相似度
            similarity = np.sum(embedding * stored_embedding) / (
                np.sqrt(np.sum(embedding ** 2)) * np.sqrt(np.sum(stored_embedding ** 2))
            )
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_speaker = speaker_id
        
        # 如果相似度高于阈值，返回最佳匹配的说话人
        if max_similarity > self.threshold:
            return best_speaker
        else:
            return None
    
    def save_embeddings(self, path: str):
        """保存说话人嵌入到文件"""
        embeddings_dict = {spk: emb.tolist() for spk, emb in self.speaker_embeddings.items()}
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"说话人嵌入已保存到: {path}")
    
    def load_embeddings(self, path: str):
        """从文件加载说话人嵌入"""
        if not os.path.exists(path):
            logger.warning(f"嵌入文件不存在: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            embeddings_dict = json.load(f)
        
        self.speaker_embeddings = {spk: np.array(emb) for spk, emb in embeddings_dict.items()}
        logger.info(f"已加载 {len(self.speaker_embeddings)} 个说话人嵌入")


class ASR(nn.Module):
    """语音识别模块，使用Whisper模型"""
    
    def __init__(self, model_size="tiny", device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"正在初始化ASR模块，使用设备: {self.device}，模型大小: {model_size}")
        
        # 加载Whisper模型
        self.model = whisper.load_model(model_size, device=self.device)
        
        logger.info("ASR模块初始化完成")
    
    def forward(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """为语音片段识别文本"""
        segments_with_text = []
        
        for segment in segments:
            # 将音频转为numpy数组
            audio_np = segment.audio.cpu().numpy().squeeze()
            
            # 使用Whisper进行识别
            result = self.model.transcribe(
                audio_np, 
                language="zh",  # 指定中文
                task="transcribe",
                fp16=False
            )
            
            # 更新片段的文本内容
            segment.text = result["text"].strip()
            segments_with_text.append(segment)
        
        return segments_with_text


class Meeting2Conv(nn.Module):
    """会议转换为对话格式的混合模型"""
    
    def __init__(self, device=None, vad_threshold=0.5, speaker_threshold=0.75, 
                 asr_model_size="tiny", model_path=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"正在初始化Meeting2Conv模型，使用设备: {self.device}")
        
        # 初始化子模块
        self.vad = VAD(device=self.device)
        self.speaker_recognition = SpeakerRecognition(device=self.device, threshold=speaker_threshold)
        self.asr = ASR(model_size=asr_model_size, device=self.device)
        
        # 如果提供了模型路径，加载预训练的模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"已加载预训练模型: {model_path}")
        
        logger.info("Meeting2Conv模型初始化完成")
    
    def forward(self, audio_path: str, output_path: str = None) -> List[Dict]:
        """处理会议音频并生成对话格式文本"""
        logger.info(f"开始处理会议音频: {audio_path}")
        
        # 1. 使用VAD检测语音片段
        segments = self.vad(audio_path)
        
        if not segments:
            logger.warning("未检测到语音片段")
            return []
        
        # 2. 使用说话人识别为每个片段分配说话人ID
        segments = self.speaker_recognition(segments)
        
        # 3. 使用ASR识别每个片段的文本
        segments = self.asr(segments)
        
        # 4. 将结果整理为对话格式
        conversation = []
        for segment in segments:
            if segment.text:  # 只保留有文本内容的片段
                dialogue = {
                    "start_time": f"{int(segment.start // 60):02d}:{int(segment.start % 60):02d}",
                    "end_time": f"{int(segment.end // 60):02d}:{int(segment.end % 60):02d}",
                    "speaker_id": segment.speaker_id,
                    "text": segment.text
                }
                conversation.append(dialogue)
        
        # 5. 如果指定了输出路径，保存结果
        if output_path:
            self._save_conversation(conversation, output_path)
        
        logger.info(f"会议处理完成，共生成 {len(conversation)} 条对话")
        return conversation
    
    def _save_conversation(self, conversation: List[Dict], output_path: str):
        """保存对话到文件"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为文本格式
        with open(output_path, 'w', encoding='utf-8') as f:
            for utterance in conversation:
                start_time = utterance["start_time"]
                speaker_id = utterance["speaker_id"]
                text = utterance["text"]
                
                f.write(f"[{start_time}]-[{speaker_id}]-{text}\n")
        
        logger.info(f"对话已保存到: {output_path}")
    
    def save_model(self, path: str):
        """保存模型参数"""
        # 只保存说话人识别模型的参数
        torch.save(self.speaker_recognition.model.state_dict(), path)
        logger.info(f"模型已保存到: {path}")
        
        # 保存说话人嵌入
        embeddings_path = Path(path).with_suffix('.json')
        self.speaker_recognition.save_embeddings(str(embeddings_path))
    
    def load_model(self, path: str):
        """加载模型参数"""
        # 加载说话人识别模型的参数
        self.speaker_recognition.model.load_state_dict(torch.load(path, map_location=self.device))
        
        # 加载说话人嵌入
        embeddings_path = Path(path).with_suffix('.json')
        if os.path.exists(embeddings_path):
            self.speaker_recognition.load_embeddings(str(embeddings_path))

# 用于训练说话人识别模型的损失函数
class ArcFaceLoss(nn.Module):
    """ArcFace损失函数"""
    
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5):
        super().__init__()
        self.in_features = in_features # 嵌入维度
        self.out_features = out_features # 说话人数量
        self.scale = scale # 缩放因子
        self.margin = margin # 角度边界
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features)) # 权重
        nn.init.xavier_uniform_(self.weight) # 行维度为out_features，列维度为in_features，是因为是左乘
        
    def forward(self, embeddings, label):
        # 归一化权重和嵌入，确保数值稳定性
        eps = 1e-8
        
        # 使用内置的normalize函数，添加eps参数
        weight_norm = F.normalize(self.weight, p=2, dim=1, eps=eps)
        
        # 计算余弦相似度
        cos_theta = F.linear(embeddings, weight_norm)
        # F.linear的作用是将embeddings和weight_norm进行矩阵乘法，
        # 结果是一个形状为(batch_size, out_features)的矩阵，矩阵的每个元素是embeddings和weight_norm的点积
        
        # 限制余弦值在有效范围内，留出一点空间避免边界问题
        cos_theta = cos_theta.clamp(-1 + eps, 1 - eps)
        
        # 添加角度边界
        theta = torch.acos(cos_theta)
        
        # 对应标签的角度加上margin
        target_mask = torch.zeros_like(cos_theta)
        target_mask.scatter_(1, label.view(-1, 1), 1.0)
        
        theta = theta + self.margin * target_mask
        cos_theta_m = torch.cos(theta)
        
        # 应用缩放因子
        logits = self.scale * cos_theta_m
        
        return F.cross_entropy(logits, label)
