import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm
import logging
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, Optional
import matplotlib.pyplot as plt

from models.Meeting2Conv import Meeting2Conv, ECAPA_TDNN, ArcFaceLoss
from data_utils import SpeakerDataset, collate_fn

logger = logging.getLogger(__name__)

class Trainer:
    """模型训练器"""
    
    def __init__(self, config_path: str):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置设备
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(self.config.get('output_dir', 'algorithm/output'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置特征提取器
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            n_mels=80
        )
        
        # 初始化数据集
        self._init_datasets()
        
        # 初始化模型
        self._init_model()
        
        # 初始化优化器和损失函数
        self._init_optimizer()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"已加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件时出错: {e}")
            # 返回默认配置
            return {
                "data_dir": "data",
                "output_dir": "algorithm/output",
                "batch_size": 32,
                "num_epochs": 20,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "save_every": 5,
                "speaker_threshold": 0.75,
                "vad_threshold": 0.5,
                "asr_model_size": "tiny"
            }
    
    def _init_datasets(self):
        """初始化数据集和数据加载器"""
        data_dir = self.config.get('data_dir', 'data')
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        # 创建训练集
        self.train_dataset = SpeakerDataset(
            data_dir=data_dir,
            split="train",
            feature_extractor=self.feature_extractor
        )
        
        # 创建验证集
        self.val_dataset = SpeakerDataset(
            data_dir=data_dir,
            split="dev",
            feature_extractor=self.feature_extractor
        )
        
        # 创建数据加载器 - 使用自定义的collate_fn函数处理不同长度的特征
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn  # 添加自定义的collate函数
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn  # 添加自定义的collate函数
        )
        
        logger.info(f"训练集大小: {len(self.train_dataset)}, 验证集大小: {len(self.val_dataset)}")
        logger.info(f"说话人数量: {len(self.train_dataset.speaker_ids)}")
    
    def _init_model(self):
        """初始化模型"""
        # 获取说话人数量
        num_speakers = len(self.train_dataset.speaker_ids)
        
        # 创建ECAPA-TDNN模型
        self.ecapa_model = ECAPA_TDNN(
            input_size=80,
            channels=512,
            emb_dim=192
        ).to(self.device)
        
        # 创建ArcFace损失函数
        self.arc_loss = ArcFaceLoss(
            in_features=192,
            out_features=num_speakers,
            scale=30.0,
            margin=0.2
        ).to(self.device)
        
        # 如果存在预训练模型，加载它
        pretrained_path = self.config.get('pretrained_model_path', None)
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                self.ecapa_model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                logger.info(f"已加载预训练模型: {pretrained_path}")
            except Exception as e:
                logger.error(f"加载预训练模型时出错: {e}")
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        
        # 创建优化器
        self.optimizer = optim.Adam([
            {'params': self.ecapa_model.parameters()},
            {'params': self.arc_loss.parameters()}
        ], lr=lr, weight_decay=weight_decay)
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    def train(self):
        """训练模型"""
        num_epochs = self.config.get('num_epochs', 20)
        save_every = self.config.get('save_every', 5)
        
        logger.info(f"开始训练，共{num_epochs}个epoch")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            # 在验证集上评估
            val_loss = self._validate()
            self.val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存模型
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(is_best=False)
            
            # 如果是最佳模型，保存它
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(is_best=True)
            
            # 绘制损失曲线
            self._plot_losses()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        logger.info("训练完成")
    
    def _train_epoch(self) -> float:
        """训练一个epoch"""
        self.ecapa_model.train()
        self.arc_loss.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        start_time = time.time()
        
        # 获取梯度裁剪值
        clip_grad_norm = self.config.get('training', {}).get('clip_grad_norm', 3.0)
        
        # 创建进度条
        progress_bar = tqdm(self.train_loader, desc=f"训练 Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取特征和标签
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # 前向传播
            embeddings = self.ecapa_model(features)
            loss = self.arc_loss(embeddings, labels)
            
            # 检查损失是否为NaN
            if torch.isnan(loss).any():
                logger.warning(f"批次 {batch_idx} 中检测到NaN损失，跳过此批次")
                continue
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪可以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.ecapa_model.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.arc_loss.parameters(), clip_grad_norm)
            
            # 更新参数
            self.optimizer.step()
            
            # 更新总损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # 更新全局步数
            self.global_step += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 计算epoch耗时
        elapsed_time = time.time() - start_time
        logger.info(f"训练Epoch耗时: {elapsed_time:.2f}秒")
        
        return avg_loss
    
    def _validate(self) -> float:
        """在验证集上评估模型"""
        self.ecapa_model.eval()
        self.arc_loss.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # 创建进度条
        progress_bar = tqdm(self.val_loader, desc=f"验证 Epoch {self.current_epoch+1}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # 获取特征和标签
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # 前向传播
                embeddings = self.ecapa_model(features)
                loss = self.arc_loss(embeddings, labels)
                
                # 更新总损失
                total_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
                })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def _save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建检查点路径
        if is_best:
            checkpoint_path = self.output_dir / "best_model.pth"
        else:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch+1}.pth"
        
        # 保存ECAPA-TDNN模型
        torch.save(self.ecapa_model.state_dict(), checkpoint_path)
        
        # 如果是最佳模型，也保存完整的Meeting2Conv模型
        if is_best:
            # 创建Meeting2Conv模型
            meeting2conv = Meeting2Conv(
                device=self.device,
                vad_threshold=self.config.get('vad_threshold', 0.5),
                speaker_threshold=self.config.get('speaker_threshold', 0.75),
                asr_model_size=self.config.get('asr_model_size', 'tiny')
            )
            
            # 设置ECAPA-TDNN模型
            meeting2conv.speaker_recognition.model.load_state_dict(self.ecapa_model.state_dict())
            
            # 保存完整模型
            meeting2conv_path = self.output_dir / "best_meeting2conv.pth"
            meeting2conv.save_model(str(meeting2conv_path))
        
        logger.info(f"已保存{'最佳' if is_best else ''}模型检查点: {checkpoint_path}")
    
    def _plot_losses(self):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 6))
        
        # 绘制训练损失
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='训练损失', marker='o')
        
        # 绘制验证损失
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='验证损失', marker='x')
        
        # 设置图表
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.savefig(self.output_dir / 'loss_curve.png')
        plt.close()
    
    def test(self, audio_path: str, output_path: Optional[str] = None):
        """测试模型，处理音频文件"""
        # 加载最佳模型
        best_model_path = self.output_dir / "best_meeting2conv.pth"
        
        if not best_model_path.exists():
            logger.error(f"找不到最佳模型: {best_model_path}")
            return
        
        # 创建Meeting2Conv模型
        model = Meeting2Conv(
            device=self.device,
            vad_threshold=self.config.get('vad_threshold', 0.5),
            speaker_threshold=self.config.get('speaker_threshold', 0.75),
            asr_model_size=self.config.get('asr_model_size', 'tiny'),
            model_path=str(best_model_path)
        )
        
        # 处理音频文件
        if not output_path:
            # 生成默认输出路径
            audio_name = Path(audio_path).stem
            output_path = str(self.output_dir / f"{audio_name}_result.txt")
        
        # 处理并生成对话文本
        start_time = time.time()
        conversation = model(audio_path, output_path=output_path)
        elapsed_time = time.time() - start_time
        
        logger.info(f"音频处理完成，耗时: {elapsed_time:.2f}秒")
        logger.info(f"生成了 {len(conversation)} 条对话，已保存到: {output_path}")
        
        return conversation
