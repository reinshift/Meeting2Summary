import os
import argparse
import logging
import torch
import random
import numpy as np
from pathlib import Path

from trainer import Trainer
from models.Meeting2Conv import Meeting2Conv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('algorithm/output/meeting2conv.log')
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Meeting2Conv: 会议音频转对话文本')
    
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='运行模式: train (训练模型) 或 test (处理音频)')
    
    parser.add_argument('--config', type=str, default='algorithm/config.json',
                        help='配置文件路径')
    
    parser.add_argument('--audio', type=str, default=None,
                        help='测试模式下的输入音频文件路径')
    
    parser.add_argument('--output', type=str, default=None,
                        help='测试模式下的输出文本文件路径')
    
    return parser.parse_args()

def run():
    """运行主程序"""
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs('algorithm/output', exist_ok=True)
    
    # 创建训练器
    trainer = Trainer(args.config)
    
    # 设置随机种子
    set_seed(trainer.config.get('random_seed', 42))
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        logger.info("开始训练模式")
        trainer.train()
    elif args.mode == 'test':
        if args.audio is None:
            logger.error("测试模式下必须指定音频文件路径 (--audio)")
            return
        
        logger.info(f"开始测试模式，处理音频: {args.audio}")
        
        # 检查音频文件是否存在
        if not os.path.exists(args.audio):
            logger.error(f"音频文件不存在: {args.audio}")
            return
        
        # 测试模型
        trainer.test(args.audio, args.output)

def process_single_audio(audio_path, output_path=None, model_path=None, device=None):
    """处理单个音频文件
    
    这个函数可以从外部调用，用于在不通过命令行的情况下处理音频。
    
    Args:
        audio_path: 音频文件路径
        output_path: 输出文本路径，如果为None则使用默认路径
        model_path: 模型路径，如果为None则使用默认的最佳模型
        device: 设备，如果为None则自动选择
        
    Returns:
        生成的对话列表
    """
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置模型路径
    if model_path is None:
        model_path = 'algorithm/output/best_meeting2conv.pth'
    
    # 设置输出路径
    if output_path is None:
        output_dir = 'algorithm/output'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(audio_path).stem}_result.txt")
    
    # 创建模型
    try:
        model = Meeting2Conv(
            device=device,
            vad_threshold=0.5,
            speaker_threshold=0.75,
            asr_model_size='tiny',
            model_path=model_path if os.path.exists(model_path) else None
        )
        
        # 处理音频
        conversation = model(audio_path, output_path=output_path)
        
        logger.info(f"音频处理完成，生成了 {len(conversation)} 条对话")
        logger.info(f"结果已保存到: {output_path}")
        
        return conversation
    except Exception as e:
        logger.error(f"处理音频时出错: {e}")
        return []

if __name__ == '__main__':
    run()