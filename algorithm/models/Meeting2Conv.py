import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class EcapaTDNN(nn.Module):
    """ECAPA-TDNN模块，用于说话人识别"""
    def __init__(self, 
                 input_dim: int = 80, 
                 channels: List[int] = [512, 512, 512, 512, 1536],
                 kernel_sizes: List[int] = [5, 3, 3, 3, 1],
                 dilations: List[int] = [1, 2, 3, 4, 1],
                 attention_channels: int = 128,
                 embedding_dim: int = 192):
        super(EcapaTDNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, channels[0], kernel_size=kernel_sizes[0], dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU()
        
        # SE-Res2Net 块
        self.layers = nn.ModuleList()
        for i in range(1, len(channels) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(channels[i-1], channels[i], kernel_size=kernel_sizes[i], dilation=dilations[i], bias=False),
                    nn.BatchNorm1d(channels[i]),
                    nn.ReLU(),
                    SEModule(channels[i], 8)
                )
            )
        
        # 注意力统计池化
        self.attention = nn.Sequential(
            nn.Conv1d(channels[-2], attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, channels[-2], kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # 最终的线性层 - 修复输入通道数
        # 由于注意力统计池化后通道数翻倍(mu和sg拼接)，所以输入通道数是channels[-2]*2
        self.final_conv = nn.Conv1d(channels[-2]*2, channels[-1], kernel_size=kernel_sizes[-1], dilation=dilations[-1])
        self.final_bn = nn.BatchNorm1d(channels[-1])
        self.embedding = nn.Linear(channels[-1], embedding_dim)
        
    def forward(self, x):
        # 输入 x 的形状: [batch, time, freq]
        x = x.transpose(1, 2)  # [batch, freq, time]
        
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        
        # 添加残差连接时确保维度匹配
        for layer in self.layers:
            # 保存原始x，用于后续残差连接
            residual = x
            # 通过当前层
            layer_output = layer(x)
            
            # 确保残差连接的维度匹配
            if residual.size(2) != layer_output.size(2):
                # 使用插值调整时间维度
                residual = F.interpolate(residual, size=layer_output.size(2), mode='linear')
            
            # 残差连接
            x = residual + layer_output
        
        # 注意力统计池化 - 修复数值稳定性问题
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        
        # 添加一个小的epsilon值确保平方根内的值为正数
        epsilon = 1e-6
        var = torch.clamp(torch.sum(x**2 * w, dim=2) - mu**2, min=epsilon)
        sg = torch.sqrt(var)
        
        x = torch.cat((mu, sg), dim=1)  # 将通道数翻倍 [batch, channels*2]
        
        # 确保x的形状正确
        x = x.unsqueeze(-1)  # [batch, channels*2, 1]
        x = self.final_conv(x)  # 现在final_conv接受channels*2个输入通道
        x = self.relu(self.final_bn(x)).squeeze(-1)
        x = self.embedding(x)
        
        return F.normalize(x, p=2, dim=1)  # L2归一化

class SEModule(nn.Module):
    """压缩激励模块"""
    def __init__(self, channels, reduction=8):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(batch_size, channels, 1)
        return x * y

class ASRModule(nn.Module):
    """语音识别模块，将语音转换为文本"""
    def __init__(self, 
                 input_dim: int = 80,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 vocab_size: int = 5000):
        super(ASRModule, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 修正LSTM输入维度计算逻辑
        self.input_dim = input_dim
        # 卷积后的频率维度
        freq_dim = input_dim // 4  # 经过两次stride=2的下采样
        self.lstm_input_dim = 64 * freq_dim  # 64是最后一个卷积的通道数
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # 双向LSTM
        
    def forward(self, x):
        """
        处理变长序列的前向传播
        输入 x 的形状: [batch, time, freq]
        """
        batch_size, time_steps, freq = x.size()
        
        # 步骤1: 通过卷积层处理
        x = x.unsqueeze(1)  # [batch, channel=1, time, freq]
        x = self.conv(x)  # [batch, channels, time/4, freq/4]
        
        # 为避免维度不匹配，我们使用自适应池化固定输出时间维度
        _, channels, _, conv_freq = x.size()
        
        # 使用自适应池化将时间维度统一为固定值16，避免不同批次间的维度不一致
        x = F.adaptive_avg_pool2d(x, (16, conv_freq))  # 统一时间步为16
        
        # 重塑为LSTM输入
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch, 16, channels, freq/4]
        x = x.view(batch_size, 16, -1)  # [batch, 16, channels*freq/4]
        
        # 通过LSTM处理
        x, _ = self.lstm(x)  # [batch, 16, hidden_dim*2]
        
        # 通过全连接层映射到词汇表
        x = self.fc(x)  # [batch, 16, vocab_size]
        
        return x  # [batch, 16, vocab_size]

class Meeting2Conv(nn.Module):
    """会话转换模型，包含说话人识别和语音转文本模块"""
    def __init__(self, 
                 input_dim: int = 80,
                 speaker_embedding_dim: int = 192,
                 hidden_dim: int = 512,
                 vocab_size: int = 5000,
                 similarity_threshold: float = 0.75):
        super(Meeting2Conv, self).__init__()
        
        # 特征提取器 - 修复参数
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            n_mels=input_dim
        )
        
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_dim = input_dim
        
        # 说话人识别模块
        self.speaker_encoder = EcapaTDNN(
            input_dim=input_dim,
            embedding_dim=speaker_embedding_dim
        )
        
        # 语音识别模块
        self.asr_module = ASRModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size
        )
        
        # 说话人聚类参数
        self.similarity_threshold = similarity_threshold
        
        # 词汇表和解码器
        self.vocab_size = vocab_size
        # 使用词汇表映射替代占位符
        self.idx_to_char = self._load_vocabulary()
        
    def _load_vocabulary(self):
        """加载词汇表，或者创建默认映射"""
        try:
            # 尝试从文件加载词汇表
            vocab_path = "../data/vocabulary.txt"
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab = {}
                    for i, line in enumerate(f):
                        word = line.strip()
                        if word:  # 确保非空行
                            vocab[i] = word
                return vocab
            else:
                # 如果没有词汇表文件，使用常用中文字符作为默认映射
                # 这里仅作为示例包含一些常用中文字符
                common_chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
                return {i: char for i, char in enumerate(common_chars)}
        except Exception as e:
            print(f"加载词汇表时出错: {str(e)}")
            # 出错时返回占位符
            return {i: f"char_{i}" for i in range(self.vocab_size)}
    
    def extract_features(self, audio: torch.Tensor):
        """从音频中提取特征"""
        # 修正特征提取逻辑
        if audio.dim() == 2:
            # 如果是批处理，则对每个样本提取特征
            features = []
            for a in audio:
                # 确保音频维度正确
                if a.dim() == 1:
                    a = a.unsqueeze(0)  # [1, time]
                mel_spec = self.feature_extractor(a)  # [1, n_mels, time]
                mel_spec = self.to_db(mel_spec)  # 转换为分贝
                
                # 使用自适应池化确保时间维度一致
                time_dim = mel_spec.size(2)
                if time_dim > 1000:  # 如果时间维度太长，则缩短
                    mel_spec = F.adaptive_avg_pool2d(mel_spec, (mel_spec.size(1), 1000))
                
                features.append(mel_spec.squeeze(0).transpose(0, 1))  # [time, n_mels]
            
            # 找出批次中最短的时间长度
            min_time_length = min(f.size(0) for f in features)
            
            # 截断所有特征到相同长度
            features = [f[:min_time_length] for f in features]
            
            return torch.stack(features)  # [batch, time, n_mels]
        else:
            # 单个样本
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # [1, time]
            mel_spec = self.feature_extractor(audio)  # [1, n_mels, time]
            mel_spec = self.to_db(mel_spec)  # 转换为分贝
            
            # 限制过长的特征
            time_dim = mel_spec.size(2)
            if time_dim > 1000:
                mel_spec = F.adaptive_avg_pool2d(mel_spec, (mel_spec.size(1), 1000))
                
            return mel_spec.squeeze(0).transpose(0, 1)  # [time, n_mels]
    
    def forward(self, audio_segment: torch.Tensor):
        """
        前向传播，处理单个音频片段
        
        参数:
            audio_segment: 音频片段 [batch, time] 或 [time]
            
        返回:
            speaker_embedding: 说话人的嵌入向量
            asr_output: ASR模块的输出（词汇表上的概率分布）
        """
        # 确保输入维度正确
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)  # [1, time]
            
        features = self.extract_features(audio_segment)  # [batch, time, freq]
        
        # 提取说话人特征
        speaker_embedding = self.speaker_encoder(features)
        
        # 语音识别
        asr_output = self.asr_module(features)
        
        return {
            'speaker_embedding': speaker_embedding,
            'asr_output': asr_output
        }
    
    def identify_speaker(self, speaker_embedding: torch.Tensor, speaker_database: Dict[str, torch.Tensor]):
        """
        识别说话人，如果是新说话人则注册
        
        参数:
            speaker_embedding: 当前说话人的嵌入向量
            speaker_database: 已知说话人的数据库
            
        返回:
            speaker_id: 说话人ID
            is_new: 是否是新说话人
        """
        if not speaker_database:
            # 数据库为空，这是第一个说话人
            return "SPEAKER_1", True
            
        # 计算与已有说话人的相似度
        max_similarity = -1
        most_similar_id = None
        
        for speaker_id, embedding in speaker_database.items():
            # 计算余弦相似度
            similarity = F.cosine_similarity(speaker_embedding.unsqueeze(0), embedding.unsqueeze(0))
            if similarity > max_similarity:
                max_similarity = similarity.item()  # 转换为Python标量
                most_similar_id = speaker_id
                
        # 降低相似度阈值以更容易识别不同说话人
        if max_similarity > self.similarity_threshold * 0.85:
            # 识别为已有说话人
            return most_similar_id, False
        else:
            # 新说话人
            return f"SPEAKER_{len(speaker_database) + 1}", True
            
    def process_meeting(self, audio: torch.Tensor, segment_length: int = 16000*3):
        """
        处理完整会议音频，将其转换为对话形式
        
        参数:
            audio: 完整的音频数据 [time]
            segment_length: 片段长度（默认3秒）
            
        返回:
            conversation: 会议转写结果，格式为[timestamp]说话人：文本
        """
        speaker_database = {}  # 说话人数据库
        conversation = []
        
        # 确保输入是一维的
        if audio.dim() > 1:
            audio = audio.squeeze(0)
            
        # 通过不同的步长和更多重叠来捕获更多说话人变化
        # 减少步长以捕获更多的说话人变化点
        step_size = segment_length // 3  # 更多重叠 (66%)
        
        for i in range(0, len(audio) - segment_length // 2, step_size):
            segment = audio[i:i+segment_length]
            if len(segment) < segment_length // 2:
                # 如果片段太短，就跳过
                continue
                
            # 补齐长度
            if len(segment) < segment_length:
                segment = F.pad(segment, (0, segment_length - len(segment)))
                
            timestamp = i / 16000  # 假设采样率为16kHz
            
            # 降低静音检测阈值以捕获更多的语音片段
            energy = (segment ** 2).mean().item()
            if energy < 5e-5:  # 降低静音阈值
                continue
                
            # 处理当前片段
            outputs = self.forward(segment)
            speaker_embedding = outputs['speaker_embedding'].squeeze(0)
            asr_output = outputs['asr_output'].squeeze(0)
            
            # 识别说话人
            speaker_id, is_new = self.identify_speaker(speaker_embedding, speaker_database)
            
            # 如果是新说话人，将其添加到数据库
            if is_new:
                speaker_database[speaker_id] = speaker_embedding.detach().clone()  # 分离并复制张量
                
            # 解码ASR输出为文本
            text = self._decode_asr_output(asr_output)
            
            # 如果有文本，则添加到对话中
            if text and text.strip():
                conversation.append(f"[{timestamp:.2f}]{speaker_id}：{text}")
                
        return conversation
        
    def _decode_asr_output(self, asr_output):
        """将ASR输出解码为文本（简化版）"""
        # 在实际应用中，这里需要使用更复杂的解码算法（如CTC解码）
        # 或者使用预训练的语言模型进行解码
        # 这里仅作为示例，简单取argmax
        indices = torch.argmax(asr_output, dim=-1)
        
        # 将索引转换为字符
        chars = [self.idx_to_char.get(idx.item(), "") for idx in indices if idx.item() > 0]
        
        # 合并相同的连续字符（简化CTC解码）
        text = []
        for i, char in enumerate(chars):
            if i == 0 or char != chars[i-1]:
                text.append(char)
                
        return "".join(text)

def build_model(config):
    """根据配置构建模型"""
    return Meeting2Conv(
        input_dim=config.get('input_dim', 80),
        speaker_embedding_dim=config.get('speaker_embedding_dim', 192),
        hidden_dim=config.get('hidden_dim', 512),
        vocab_size=config.get('vocab_size', 5000),
        similarity_threshold=config.get('similarity_threshold', 0.75)
    )
