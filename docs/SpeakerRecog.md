# 说话人识别

整个过程通过 `SpeakerRecognition` 类负责实现，主要调用路径：
```bash
Meeting2Conv.forward -> SpeakerRecognition.forward -> SpeakerRecognition._extract_features -> ECAPA_TDNN -> SpeakerRecognition._find_best_match
```
## 特征提取
```python
def _extract_features(self, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    # 确保音频是单通道的，若不是，则对多通取平均
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
```
## 声纹嵌入
```python
def forward(self, segments: List[AudioSegment]) -> List[AudioSegment]:
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
```
由VAD模块裁切会议音频中有说话人的片段，再将这些片段交付给ecapa-tdnn模型，经过提取特征转换为mel频谱后流入前馈网络，得到声纹嵌入。将声纹嵌入信息和历史保存的声纹信息做比对（如果原来没有，则直接注册新说话人id），用余弦相似度来查找最匹配的说话人，如果无匹配到的说话人，则注册新说话人id，最终返回语音片段以及其绑定的说话人。
## ECAPA-TDNN网络架构
- 卷积层conv1d
- 3\*SERes2Block
- 注意力统计池化层asp
- 输出层
其中卷积层后用relu激活，以及做batchnorm。asp层后过一个全连接层，再作一个batchnorm。最终输出一个向量，即说话人的特征（声纹）。而在实际训练中，这个特征向量还会过一个全连接层进行分类（因为数据集的标签有限），表面是在训练分类能力，但更关注的是学习一个能够产生高区分性特征嵌入的模型。
## id 分配
```python
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
```
## 流程总结
1. 对每个VAD检测到的语音片段进行处理
2. 提取梅尔频谱特征并进行标准化
3. 通过ECAPA-TDNN模型将特征转换为固定维度的嵌入向量
4. 计算嵌入向量与已知说话人嵌入的余弦相似度
5. 如果找到相似度高于阈值的匹配，使用该说话人ID
6. 否则创建新的说话人ID并保存嵌入
7. 将说话人ID附加到语音片段上，以供后续ASR处理