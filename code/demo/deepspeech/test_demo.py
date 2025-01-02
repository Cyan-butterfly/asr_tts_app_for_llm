import deepspeech
import numpy as np

# 加载模型文件路径
model_file_path = r'D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\deep speech\deepspeech-0.9.3-models-zh-CN.pbmm'
scorer_file_path = r'D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\deep speech\deepspeech-0.9.3-models-zh-CN.scorer'

# 加载 DeepSpeech 模型
model = deepspeech.Model(model_file_path)
model.enableExternalScorer(scorer_file_path)

# 加载 PCM 文件
# pcm_file_path = r'D:\BaiduSyncdisk\mywork\asr\code\demo\xunfei\test1.pcm'
pcm_file_path = r'D:\BaiduSyncdisk\mywork\asr\code\demo\xunfei\test1.pcm'

# 假设 PCM 文件的格式是 16-bit、单声道、采样率为 16kHz
with open(pcm_file_path, 'rb') as pcm_file:
    pcm_data = np.frombuffer(pcm_file.read(), dtype=np.int16)

# 执行语音识别
text = model.stt(pcm_data)
print("Recognized Text:", text)
