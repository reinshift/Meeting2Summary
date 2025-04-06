import os
import re
import torch
import torchaudio
import numpy as np
import textgrid
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    """说话人片段数据类"""
    speaker_id: str  # 说话人ID
    start_time: float  # 开始时间 (秒)
    end_time: float  # 结束时间 (秒)
    text: Optional[str] = None  # 文本内容


def parse_rttm_file(rttm_path: str) -> List[SpeakerSegment]:
    """解析RTTM格式文件，获取说话人片段信息"""
    segments = []
    
    try:
        with open(rttm_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) < 9 or parts[0] != 'SPEAKER':
                    continue
                
                # 解析RTTM行
                # SPEAKER 20200706_L_R001S01C01 1 33.6354 2.3550 <NA> <NA> 001-M <NA> <NA>
                speaker_id = parts[7]  # 001-M
                start_time = float(parts[3])  # 33.6354
                duration = float(parts[4])  # 2.3550
                end_time = start_time + duration
                
                segments.append(SpeakerSegment(
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time
                ))
    except Exception as e:
        logger.error(f"解析RTTM文件({rttm_path})时出错: {e}")
    
    return sorted(segments, key=lambda x: x.start_time)


def parse_textgrid_file(textgrid_path: str) -> Dict[str, List[SpeakerSegment]]:
    """解析TextGrid格式文件，获取说话人及其对应的文本内容"""
    speaker_segments = {}
    
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        
        for tier in tg.tiers:
            speaker_id = tier.name  # 001-M
            
            if not re.match(r'\d+-[MF]', speaker_id):
                continue  # 跳过非说话人层
            
            segments = []
            for interval in tier.intervals:
                if interval.mark.strip():  # 如果有文本内容
                    segments.append(SpeakerSegment(
                        speaker_id=speaker_id,
                        start_time=interval.minTime,
                        end_time=interval.maxTime,
                        text=interval.mark.strip()
                    ))
            
            if segments:
                speaker_segments[speaker_id] = segments
    except Exception as e:
        logger.error(f"解析TextGrid文件({textgrid_path})时出错: {e}")
    
    return speaker_segments


def merge_speaker_segments(rttm_segments: List[SpeakerSegment], 
                          textgrid_dict: Dict[str, List[SpeakerSegment]]) -> List[SpeakerSegment]:
    """合并RTTM和TextGrid中的信息，得到完整的说话人片段信息"""
    merged_segments = []
    
    # 为RTTM的每个片段尝试找到对应的TextGrid文本内容
    for segment in rttm_segments:
        speaker_id = segment.speaker_id
        
        if speaker_id in textgrid_dict:
            # 查找与RTTM片段时间重叠的TextGrid片段
            for tg_segment in textgrid_dict[speaker_id]:
                # 如果TextGrid片段与RTTM片段有重叠，使用TextGrid的内容
                if (segment.start_time <= tg_segment.end_time and 
                    segment.end_time >= tg_segment.start_time and 
                    tg_segment.text):
                    
                    merged_segments.append(SpeakerSegment(
                        speaker_id=speaker_id,
                        start_time=max(segment.start_time, tg_segment.start_time),
                        end_time=min(segment.end_time, tg_segment.end_time),
                        text=tg_segment.text
                    ))
        else:
            # 如果在TextGrid中没有找到对应的说话人，则保留原样
            merged_segments.append(segment)
    
    return sorted(merged_segments, key=lambda x: x.start_time)


def extract_audio_segment(audio_path: str, start_time: float, end_time: float) -> Tuple[torch.Tensor, int]:
    """提取音频文件的指定时间段"""
    try:
        # 加载整个音频文件
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 计算起始和结束的样本索引
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # 提取片段
        segment = waveform[:, start_sample:end_sample]
        
        return segment, sample_rate
    except Exception as e:
        logger.error(f"提取音频片段时出错: {e}")
        # 返回空的音频片段
        return torch.zeros(1, 1), 16000


def load_dataset(data_dir: str, split: str = "train") -> List[Dict]:
    """加载数据集
    
    Args:
        data_dir: 数据目录
        split: 数据集划分，"train", "dev" 或 "test"
        
    Returns:
        包含音频路径、说话人片段和文本的字典列表
    """
    wav_dir = os.path.join(data_dir, "wav")
    textgrid_dir = os.path.join(data_dir, "TextGrid")
    
    dataset = []
    
    # 假设数据集中有一个split文件列表
    split_file = os.path.join(data_dir, f"{split}.txt")
    
    if os.path.exists(split_file):
        with open(split_file, 'r', encoding='utf-8') as f:
            file_list = [line.strip() for line in f if line.strip()]
    else:
        # 如果没有split文件，则扫描wav目录
        file_list = [f[:-5] for f in os.listdir(wav_dir) if f.endswith('.flac')]
        logger.warning(f"未找到{split}划分文件，使用所有{len(file_list)}个音频文件")
    
    for base_name in file_list:
        wav_path = os.path.join(wav_dir, f"{base_name}.flac")
        rttm_path = os.path.join(textgrid_dir, f"{base_name}.rttm")
        textgrid_path = os.path.join(textgrid_dir, f"{base_name}.TextGrid")
        
        if not os.path.exists(wav_path):
            logger.warning(f"音频文件不存在: {wav_path}")
            continue
        
        if not os.path.exists(rttm_path):
            logger.warning(f"RTTM文件不存在: {rttm_path}")
            continue
        
        # 解析RTTM文件
        rttm_segments = parse_rttm_file(rttm_path)
        
        # 如果存在TextGrid文件，解析它
        textgrid_dict = {}
        if os.path.exists(textgrid_path):
            textgrid_dict = parse_textgrid_file(textgrid_path)
        
        # 合并RTTM和TextGrid的信息
        segments = merge_speaker_segments(rttm_segments, textgrid_dict)
        
        # 添加到数据集
        dataset.append({
            "wav_path": wav_path,
            "rttm_path": rttm_path,
            "textgrid_path": textgrid_path if os.path.exists(textgrid_path) else None,
            "segments": segments
        })
    
    logger.info(f"加载了{len(dataset)}个{split}集样本")
    return dataset


def collate_fn(batch):
    """自定义的collate函数，处理不同长度的特征
    
    方法：将批次内的所有特征填充或裁剪到相同长度
    """
    # 提取特征、标签和音频
    features = [item["features"] for item in batch]  # 每个特征形状为 [time, freq]
    labels = [item["label"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]
    audios = [item["audio"] for item in batch]
    
    # 找出批次中特征的第一维度(时间维度)的最大和最小值
    time_lengths = [f.shape[0] for f in features]
    max_length = max(time_lengths)
    min_length = min(time_lengths)
    
    # 决定使用的长度（这里使用可配置的策略）
    # 这里使用一个折中方案：不超过最大长度的150%，避免填充太多
    target_length = min(max_length, int(min_length * 1.5))
    
    # 处理特征
    processed_features = []
    for feat in features:
        freq_dim = feat.shape[1]  # 获取频率维度
        
        if feat.shape[0] < target_length:
            # 需要填充
            padding = torch.zeros((target_length - feat.shape[0], freq_dim), 
                                 dtype=feat.dtype, device=feat.device)
            processed_feat = torch.cat([feat, padding], dim=0)
        else:
            # 需要裁剪
            processed_feat = feat[:target_length, :]
        
        processed_features.append(processed_feat)
    
    # 堆叠特征和标签
    stacked_features = torch.stack(processed_features)  # [batch, time, freq]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # 返回处理后的批次
    return {
        "features": stacked_features,
        "label": labels_tensor,
        "speaker_id": speaker_ids,
        "audio": audios  # 注意：audio不需要堆叠，保持为列表
    }


class SpeakerDataset(torch.utils.data.Dataset):
    """用于训练说话人识别模型的数据集类"""
    
    def __init__(self, data_dir: str, split: str = "train", feature_extractor=None):
        self.data = load_dataset(data_dir, split)
        self.feature_extractor = feature_extractor
        
        # 为每个唯一的说话人ID分配一个数字标签
        self.speaker_ids = sorted(list(set(
            segment.speaker_id for item in self.data 
            for segment in item["segments"]
        )))
        self.speaker_to_idx = {spk: idx for idx, spk in enumerate(self.speaker_ids)}
        
        # 创建样本列表，每个样本包含音频路径、时间范围和说话人标签
        self.samples = []
        for item in self.data:
            wav_path = item["wav_path"]
            for segment in item["segments"]:
                if segment.end_time - segment.start_time > 0.5:  # 只使用大于0.5秒的片段
                    self.samples.append({
                        "wav_path": wav_path,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "speaker_id": segment.speaker_id,
                        "label": self.speaker_to_idx[segment.speaker_id]
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 提取音频片段
        audio, sr = extract_audio_segment(
            sample["wav_path"], 
            sample["start_time"], 
            sample["end_time"]
        )
        
        # 确保音频是单通道的 - 如果是多通道，取均值转换为单通道
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # 如果提供了特征提取器，则提取特征
        if self.feature_extractor is not None:
            # 将音频转为特征
            with torch.no_grad():
                features = self.feature_extractor(audio)
                # 对数变换
                features = torch.log(features + 1e-6)
                # 标准化
                mean = torch.mean(features, dim=2, keepdim=True)
                std = torch.std(features, dim=2, keepdim=True)
                features = (features - mean) / (std + 1e-6)
                # 转置为 [time, freq]，去掉批次和通道维度
                features = features.squeeze(0).transpose(0, 1)  # [time, freq]
        else:
            features = audio
        
        return {
            "features": features,
            "audio": audio,
            "label": sample["label"],
            "speaker_id": sample["speaker_id"]
        }
