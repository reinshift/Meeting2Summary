import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.Meeting2Conv import Meeting2Conv, build_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MeetingDataset(Dataset):
    """会议音频数据集"""
    def __init__(self, 
                 wav_dir: str, 
                 textgrid_dir: str, 
                 rttm_dir: str, 
                 segment_length: int = 16000*3,
                 sample_rate: int = 16000):
        self.wav_dir = Path(wav_dir)
        self.textgrid_dir = Path(textgrid_dir)
        self.rttm_dir = Path(rttm_dir)
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        # 获取所有音频文件
        self.wav_files = list(self.wav_dir.glob("*.flac"))
        logger.info(f"Found {len(self.wav_files)} audio files")
        
        # 为每个文件创建segments
        self.segments = []
        self.speaker_ids = set()  # 跟踪所有唯一的说话人ID
        
        for wav_file in self.wav_files:
            file_id = wav_file.stem
            textgrid_file = self.textgrid_dir / f"{file_id}.TextGrid"
            rttm_file = self.rttm_dir / f"{file_id}.rttm"
            
            if textgrid_file.exists() and rttm_file.exists():
                # 解析RTTM文件获取说话人信息
                speaker_segments = self._parse_rttm(rttm_file)
                
                # 检查是否成功解析了说话人段落
                if not speaker_segments:
                    logger.warning(f"No speaker segments found in {rttm_file}")
                    continue
                
                try:
                    # 获取音频信息
                    audio_info = torchaudio.info(wav_file)
                    total_frames = audio_info.num_frames
                    
                    # 音频太短或无效，跳过
                    if total_frames < self.segment_length:
                        logger.warning(f"Audio file {wav_file} too short, skipping")
                        continue
                        
                    # 使用滑动窗口创建重叠的片段
                    step_size = self.segment_length // 2  # 50%重叠
                    for start_frame in range(0, total_frames - self.segment_length, step_size):
                        end_frame = start_frame + self.segment_length
                        
                        # 找出这个时间段的主要说话人
                        start_time = start_frame / self.sample_rate
                        end_time = end_frame / self.sample_rate
                        
                        speaker_id = self._get_main_speaker(speaker_segments, start_time, end_time)
                        
                        if speaker_id:
                            self.speaker_ids.add(speaker_id)  # 添加到唯一说话人集合
                            self.segments.append({
                                'wav_file': wav_file,
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'speaker_id': speaker_id
                            })
                except Exception as e:
                    logger.error(f"Error processing {wav_file}: {str(e)}")
                    continue
        
        logger.info(f"Created {len(self.segments)} segments with {len(self.speaker_ids)} unique speakers")
        
    def _parse_rttm(self, rttm_file):
        """解析RTTM文件获取说话人信息"""
        speaker_segments = []
        try:
            with open(rttm_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        speaker_id = parts[7]
                        speaker_segments.append({
                            'start_time': start_time,
                            'end_time': start_time + duration,
                            'speaker_id': speaker_id
                        })
        except Exception as e:
            logger.error(f"Error parsing RTTM file {rttm_file}: {str(e)}")
            return []
        return speaker_segments
    
    def _get_main_speaker(self, speaker_segments, start_time, end_time):
        """获取时间段内的主要说话人"""
        # 简单实现：找出在此时间段内说话时间最长的人
        speaker_durations = {}
        segment_duration = end_time - start_time
        
        for segment in speaker_segments:
            # 计算重叠部分
            overlap_start = max(start_time, segment['start_time'])
            overlap_end = min(end_time, segment['end_time'])
            
            if overlap_end > overlap_start:
                duration = overlap_end - overlap_start
                speaker_id = segment['speaker_id']
                
                if speaker_id in speaker_durations:
                    speaker_durations[speaker_id] += duration
                else:
                    speaker_durations[speaker_id] = duration
        
        if speaker_durations:
            # 检查说话时间是否足够长
            max_duration = max(speaker_durations.values())
            if max_duration > segment_duration * 0.2:  # 只有当说话时间超过片段时长的20%
                # 返回说话时间最长的人
                return max(speaker_durations.items(), key=lambda x: x[1])[0]
        
        return None  # 没有足够长的说话片段
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        
        try:
            # 加载音频片段
            waveform, sample_rate = torchaudio.load(
                segment['wav_file'],
                frame_offset=segment['start_frame'],
                num_frames=segment['end_frame'] - segment['start_frame']
            )
            
            # 检查加载的音频是否有效
            if waveform.numel() == 0:
                # 如果加载失败，返回零张量和特殊标记
                logger.warning(f"Failed to load segment from {segment['wav_file']}")
                return {
                    'waveform': torch.zeros(self.segment_length),
                    'speaker_id': segment['speaker_id'],
                    'valid': False
                }
            
            # 重采样（如果需要）
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
                
            # 确保单声道
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # 确保长度一致
            if waveform.size(1) < self.segment_length:
                # 如果音频太短，使用填充
                padding_length = self.segment_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding_length))
            elif waveform.size(1) > self.segment_length:
                # 如果音频太长，截断到指定长度
                waveform = waveform[:, :self.segment_length]
                
            # 检查最终波形的形状是否符合预期
            if waveform.size(1) != self.segment_length:
                logger.warning(f"Unexpected waveform length: {waveform.size(1)}, expected: {self.segment_length}")
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0), size=self.segment_length, mode='linear'
                ).squeeze(0)
                
            return {
                'waveform': waveform.squeeze(0),
                'speaker_id': segment['speaker_id'],
                'valid': True
            }
        except Exception as e:
            logger.error(f"Error loading segment {idx}: {str(e)}")
            return {
                'waveform': torch.zeros(self.segment_length),
                'speaker_id': segment['speaker_id'],
                'valid': False
            }

class SpeakerLoss(nn.Module):
    """用于说话人识别的对比损失"""
    def __init__(self, margin=0.2):
        super(SpeakerLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        # 检查批次大小，如果只有一个样本则无法计算对比损失
        if embeddings.size(0) <= 1:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        try:
            # 确保embeddings为浮点数并且正确归一化
            embeddings = embeddings.float()
            # 添加小的epsilon避免归一化时除以零
            eps = 1e-12
            norm = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=eps)
            embeddings = embeddings / norm
            
            # 计算嵌入向量之间的余弦相似度
            similarity_matrix = torch.matmul(embeddings, embeddings.t())
            # 限制相似度范围避免数值不稳定
            similarity_matrix = torch.clamp(similarity_matrix, min=-1.0 + eps, max=1.0 - eps)
            
            # 将标签转换为长整型张量
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=embeddings.device)
            labels = labels.long().view(-1, 1)
            
            # 创建标签矩阵
            mask = torch.eq(labels, labels.t()).float()
            
            # 对角线为1，表示自己和自己的相似度
            mask.fill_diagonal_(1)
            
            # 获取正例和负例的掩码
            positive_mask = mask.bool()
            negative_mask = ~positive_mask
            
            # 计算正例和负例的损失
            # 使用对称性，只考虑上三角矩阵
            triu_indices = torch.triu_indices(embeddings.size(0), embeddings.size(0), offset=1, device=embeddings.device)
            
            # 检查索引是否有效
            if triu_indices.size(1) == 0:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
                
            pos_indices = triu_indices[:, mask[triu_indices[0], triu_indices[1]].bool()]
            neg_indices = triu_indices[:, ~mask[triu_indices[0], triu_indices[1]].bool()]
            
            # 如果没有足够的正例或负例对，返回0损失
            if pos_indices.size(1) == 0 or neg_indices.size(1) == 0:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
                
            # 获取正例和负例得分
            pos_scores = similarity_matrix[pos_indices[0], pos_indices[1]]
            neg_scores = similarity_matrix[neg_indices[0], neg_indices[1]]
            
            # 计算最困难的正例（最小相似度）和最困难的负例（最大相似度）
            hardest_positive = torch.min(pos_scores)
            hardest_negative = torch.max(neg_scores)
            
            # 记录关键信息用于调试
            logger.debug(f"Hardest positive: {hardest_positive.item():.4f}, Hardest negative: {hardest_negative.item():.4f}")
            
            # 计算 triplet loss 并检查是否为NaN
            loss = torch.clamp(self.margin + hardest_negative - hardest_positive, min=0.0)
            
            # 检查loss是否为NaN
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected! pos={hardest_positive.item():.4f}, neg={hardest_negative.item():.4f}")
                # 如果发现NaN，返回一个小的常数损失
                return torch.tensor(0.1, device=embeddings.device, requires_grad=True)
            
            return loss
        except Exception as e:
            logger.error(f"Error in SpeakerLoss forward: {str(e)}")
            # 返回一个可反向传播的零损失
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

def train(config):
    """训练模型"""
    # 加载配置
    data_dir = config['data_dir']
    wav_dir = os.path.join(data_dir, 'wav')
    textgrid_dir = os.path.join(data_dir, 'TextGrid')
    rttm_dir = os.path.join(data_dir, 'TextGrid')  # RTTM和TextGrid在同一目录
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 创建数据集和数据加载器
    try:
        dataset = MeetingDataset(
            wav_dir=wav_dir, 
            textgrid_dir=textgrid_dir, 
            rttm_dir=rttm_dir,
            segment_length=config['segment_length'],
            sample_rate=config['sample_rate']
        )
        
        if len(dataset) == 0:
            logger.error("Dataset is empty, cannot train model")
            return
            
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True  # 丢弃最后一个不完整的批次
        )
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        return
    
    # 构建模型
    model = build_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    speaker_loss_fn = SpeakerLoss(margin=config['triplet_margin'])
    
    # 从检查点恢复（如果指定）
    start_epoch = 0
    if 'resume_from' in config and config['resume_from']:
        try:
            checkpoint = torch.load(config['resume_from'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
    
    # 训练循环
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        total_loss = 0.0
        valid_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")):
            waveforms = batch['waveform'].to(device)
            speaker_ids = batch['speaker_id']
            valid_mask = batch['valid']
            
            # 跳过无效的批次
            if not valid_mask.any():
                logger.warning(f"Skipping batch {batch_idx} with no valid samples")
                continue
                
            try:
                # 只使用有效样本
                if not valid_mask.all():
                    waveforms = waveforms[valid_mask]
                    speaker_ids = [sid for sid, is_valid in zip(speaker_ids, valid_mask) if is_valid]
                    
                # 如果批次为空或只有一个样本，跳过
                if len(speaker_ids) <= 1:
                    logger.warning(f"Skipping batch {batch_idx} with only {len(speaker_ids)} valid samples")
                    continue
                
                # 编码为数值ID以便计算损失
                unique_ids = list(set(speaker_ids))
                id_to_index = {id: i for i, id in enumerate(unique_ids)}
                speaker_indices = torch.tensor([id_to_index[id] for id in speaker_ids], device=device)
                
                # 前向传播
                outputs = model(waveforms)
                speaker_embeddings = outputs['speaker_embedding']
                
                # 确保形状正确
                if speaker_embeddings.size(0) != len(speaker_indices):
                    logger.warning(f"Shape mismatch: speaker_embeddings: {speaker_embeddings.size(0)}, speaker_indices: {len(speaker_indices)}")
                    continue
                
                # 检查嵌入向量是否包含NaN
                if torch.isnan(speaker_embeddings).any():
                    logger.warning(f"NaN detected in speaker embeddings at batch {batch_idx}, skipping")
                    continue
                
                # 计算损失
                loss = speaker_loss_fn(speaker_embeddings, speaker_indices)
                
                # 检查损失值
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or Inf loss detected at batch {batch_idx}, skipping")
                    continue
                
                if loss.item() > 10.0:  # 异常高的损失值
                    logger.warning(f"Abnormally high loss ({loss.item():.4f}) at batch {batch_idx}, skipping")
                    continue
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸），采用更保守的值
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 检查梯度是否包含NaN
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    logger.warning(f"NaN gradient detected at batch {batch_idx}, skipping optimizer step")
                    optimizer.zero_grad()  # 清除梯度
                    continue
                
                optimizer.step()
                
                batch_loss = loss.item()
                total_loss += batch_loss
                valid_batches += 1
                
                # 每N个批次记录一次损失
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {batch_loss:.4f}")
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())  # 打印完整的错误堆栈
                continue
        
        # 如果没有有效批次，跳过此epoch
        if valid_batches == 0:
            logger.warning(f"Epoch {epoch+1} had no valid batches, skipping")
            continue
            
        # 打印epoch损失
        avg_loss = total_loss / valid_batches
        logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # 保存模型
        if (epoch + 1) % config['save_interval'] == 0:
            save_path = os.path.join(config['output_dir'], f"model_epoch_{epoch+1}.pt")
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }, save_path)
                logger.info(f"Model saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {str(e)}")
            
    logger.info("Training completed!")

def test(config):
    """测试模型"""
    # 加载模型
    model = build_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        checkpoint = torch.load(config['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {config['model_path']}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
        
    model = model.to(device)
    model.eval()
    
    # 加载测试音频
    wav_file = config['test_wav']
    if not os.path.exists(wav_file):
        logger.error(f"Test file {wav_file} does not exist")
        return
        
    try:
        waveform, sample_rate = torchaudio.load(wav_file)
        logger.info(f"Loaded test audio: {wav_file}, shape: {waveform.shape}, sample rate: {sample_rate}")
        
        # 如果需要，重采样
        if sample_rate != config['sample_rate']:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=config['sample_rate']
            )
            waveform = resampler(waveform)
            logger.info(f"Resampled to {config['sample_rate']} Hz")
        
        # 确保单声道
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.info("Converted to mono")
        
        waveform = waveform.squeeze(0).to(device)
    except Exception as e:
        logger.error(f"Error loading test audio: {str(e)}")
        return
    
    # 处理音频并生成对话
    try:
        with torch.no_grad():
            conversation = model.process_meeting(waveform, segment_length=config['segment_length'])
        
        if not conversation:
            logger.warning("Model did not generate any conversation")
            
        # 保存结果
        output_file = config['output_file']
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in conversation:
                f.write(line + '\n')
        
        logger.info(f"Processed audio and saved conversation to {output_file}")
        logger.info(f"Generated {len(conversation)} utterances")
    except Exception as e:
        logger.error(f"Error during audio processing: {str(e)}")

def run(config_path):
    """运行训练或测试"""
    # 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {str(e)}")
        return
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 根据模式运行训练或测试
    if config['mode'] == 'train':
        train(config)
    elif config['mode'] == 'test':
        test(config)
    else:
        logger.error(f"Unknown mode: {config['mode']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Meeting2Conv训练与测试")
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    
    args = parser.parse_args()
    run(args.config)
