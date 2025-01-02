# 数据加载与预处理逻辑

import io
import librosa
import numpy as np

def preprocess_audio(audio_data):
    """
    预处理音频数据：加载并调整采样率为 16kHz。
    :param audio_data: 用户上传的音频数据（二进制流）
    :return: 处理后的音频信号（numpy 数组）
    """
    # 使用 librosa 加载音频，目标采样率为 16kHz
    audio, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
    # 返回处理后的音频
    return np.array(audio, dtype=np.float32)