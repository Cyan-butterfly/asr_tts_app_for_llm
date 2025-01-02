from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# 下载和配置 Wav2Vec 模型
# 加载预训练的 wav2vec 模型和处理器
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda" if torch.cuda.is_available() else "cpu")

# 测试模型加载是否成功
print("模型和处理器加载成功！")
