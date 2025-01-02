import warnings
import logging
import torch
import os
import uuid
import tempfile
import shutil
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
import io
import wave

# 过滤警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(message)s]',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# 定义请求和响应模型
class TextRequest(BaseModel):
    text: str

class TranscriptionResponse(BaseModel):
    text: str

# 创建应用
app = FastAPI(
    title="语音识别与合成服务",
    description="",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_upload_file(upload_file: UploadFile) -> str:
    try:
        # 保存上传的文件
        file_path = os.path.join(tempfile.gettempdir(), upload_file.filename)
        with open(file_path, "wb") as f:
            f.write(upload_file.file.read())
        return file_path
    except Exception as e:
        logger.error(f"Failed to save upload file: {str(e)}")
        raise

def convert_to_wav(file_path: str) -> str:
    try:
        # 转换为 WAV
        output_file = os.path.splitext(file_path)[0] + ".wav"
        audio = AudioSegment.from_file(file_path, format="webm")
        audio.export(output_file, format="wav")
        return output_file
    except Exception as e:
        logger.error(f"Failed to convert to WAV: {str(e)}")
        raise

# 加载 Whisper 模型
whisper_storage_path = r"D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\whisper-small"
try:
    logger.info(f"Loading Whisper model from: {whisper_storage_path}")
    whisper_processor = AutoProcessor.from_pretrained(whisper_storage_path)
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_storage_path).to(device)
    whisper_model.eval()
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {str(e)}")
    whisper_processor = None
    whisper_model = None

@app.post("/transcribe", 
    response_model=TranscriptionResponse,
    summary="语音识别",
    description="将上传的音频文件转录为文本，支持多种音频格式。",
    response_description="返回识别后的文本内容",
    tags=["语音服务"]
)
async def transcribe_audio(audio_file: UploadFile = File(
    ...,
    description="要识别的音频文件（支持 wav, mp3, webm, ogg, flac 等格式）"
)
):
    try:
        if not whisper_model or not whisper_processor:
            raise HTTPException(status_code=500, detail="Whisper model not initialized")

        # 保存上传的音频文件
        file_path = save_upload_file(audio_file)
        logger.info(f"Received file: {audio_file.filename}, size: {os.path.getsize(file_path)} bytes")
        
        # 如果是webm格式，转换为wav
        if audio_file.filename.endswith('.webm'):
            logger.info("Converting from webm to WAV")
            wav_path = convert_to_wav(file_path)
            os.remove(file_path)  # 删除原始文件
            file_path = wav_path
            logger.info(f"Successfully converted to WAV: {wav_path}")

        # 读取音频文件
        audio, sr = librosa.load(file_path, sr=16000, mono=True)  # 确保单声道
        
        # 检查音频长度
        if len(audio) < sr * 0.1:  # 如果音频短于0.1秒
            raise ValueError("Audio too short")
        
        # 标准化音频
        audio = librosa.util.normalize(audio)
        
        # 检查音频是否为静音
        if np.abs(audio).max() < 0.001:  # 降低静音检测阈值
            raise ValueError("Audio is silent")
        
        # 处理音频特征
        inputs = whisper_processor(
            audio, 
            return_tensors="pt", 
            sampling_rate=16000,
            return_attention_mask=True
        )
        
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None

        # 获取中文语言token
        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="zh", task="transcribe")
        
        # 模型推理
        with torch.no_grad():
            generate_kwargs = {
                "forced_decoder_ids": forced_decoder_ids,
                "max_new_tokens": 64,
                "temperature": 0.0,
                "do_sample": False,
                "num_beams": 1
            }
            
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            
            generated_ids = whisper_model.generate(
                input_features,
                **generate_kwargs
            )
        
        # 解码文本
        transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        logger.info(f"Transcription result: {transcription}")
        
        # 清理临时文件
        os.remove(file_path)
        
        return TranscriptionResponse(text=transcription.strip())
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))