import warnings
import logging

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

# 过滤掉一些不必要的日志
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("multipart").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

from paddlespeech.cli.tts.infer import TTSExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pydantic import BaseModel
import torch
import os
import io
import wave
import tempfile
import uuid
import shutil
import time
from pydub import AudioSegment
import librosa
import numpy as np
import soundfile as sf

# 定义请求和响应模型
class TextRequest(BaseModel):
    text: str

class TranscriptionResponse(BaseModel):
    text: str

# 创建应用
app = FastAPI(
    title="语音识别与合成服务",
    description="""
    这是一个提供语音识别(ASR)和语音合成(TTS)功能的 Web API 服务。

    ## 功能特点

    * 支持语音识别：将语音转换为文本
    * 支持语音合成：将文本转换为语音
    * 支持多种音频格式
    * 实时处理
    * 跨平台支持

    ## 使用说明

    1. 语音识别 API (`/transcribe`):
       * 支持上传音频文件进行识别
       * 支持多种常见音频格式（wav, mp3, ogg等）
       * 返回识别后的文本内容

    2. 语音合成 API (`/synthesize`):
       * 输入中文或英文文本
       * 返回合成的音频文件（wav格式）
       * 支持实时流式播放

    ## 注意事项

    * 音频文件大小建议不超过10MB
    * 支持的音频格式：wav, mp3, ogg, flac
    * 语音合成支持中英文混合文本
    """,
    version="1.0.0",
    contact={
        "name": "技术支持",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置环境变量禁用代理
os.environ['NO_PROXY'] = 'huggingface.co'

# 加载 Whisper 模型
whisper_storage_path = r"D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\whisper-small"
try:
    whisper_processor = AutoProcessor.from_pretrained(whisper_storage_path)
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_storage_path).to(device)
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {str(e)}")
    whisper_processor = None
    whisper_model = None

# 加载 PaddleSpeech TTS 模型
try:
    tts_model = TTSExecutor()
except Exception as e:
    logger.error(f"Failed to initialize PaddleSpeech TTS model: {e}")
    raise RuntimeError("PaddleSpeech TTS model initialization failed.")

def convert_to_wav(input_file: str) -> str:
    """Convert audio file to wav format."""
    try:
        output_file = os.path.join(tempfile.gettempdir(), f"converted_{uuid.uuid4().hex[:8]}.wav")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        
        time.sleep(0.1)
        
        return output_file
    except Exception as e:
        logger.error(f"Error converting audio file: {str(e)}")
        raise

@app.post("/transcribe", 
    response_model=TranscriptionResponse,
    summary="语音识别",
    description="将上传的音频文件转录为文本，支持多种音频格式。",
    response_description="返回识别后的文本内容",
    tags=["语音服务"]
)
async def transcribe_audio(
    audio_file: UploadFile = File(
        ...,
        description="要识别的音频文件（支持 wav, mp3, ogg, flac 等格式）"
    )
):
    """
    将上传的音频文件转录为文本。

    ## 参数说明
    * audio_file: 要识别的音频文件，支持多种格式
        - 支持格式：wav, mp3, ogg, flac 等
        - 文件大小：建议不超过10MB
        - 采样率：建议16kHz或以上

    ## 返回说明
    * text: 识别后的文本内容

    ## 示例
    ```python
    import requests
    
    url = "http://localhost:8000/transcribe"
    files = {"audio_file": open("audio.wav", "rb")}
    response = requests.post(url, files=files)
    text = response.json()["text"]
    ```

    ## 错误说明
    * 400: 未提供音频文件或文件格式错误
    * 500: 服务器处理错误
    """
    try:
        if not whisper_model or not whisper_processor:
            raise HTTPException(status_code=500, detail="Whisper model not initialized")

        if not audio_file or not audio_file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")

        temp_input = os.path.join(tempfile.gettempdir(), f"input_{uuid.uuid4().hex[:8]}{os.path.splitext(audio_file.filename)[1]}")
        try:
            with open(temp_input, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)
            
            wav_file = convert_to_wav(temp_input)
            audio, sr = librosa.load(wav_file, sr=16000)
            
            input_features = whisper_processor(audio, return_tensors="pt", sampling_rate=16000).input_features.to(device)
            
            with torch.no_grad():
                generated_ids = whisper_model.generate(input_features)
            
            transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return TranscriptionResponse(text=transcription.strip())
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try:
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                if os.path.exists(wav_file):
                    os.remove(wav_file)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {str(e)}")
                
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize",
    summary="语音合成",
    description="将文本转换为语音，支持中英文",
    response_description="返回合成的音频文件（wav格式）",
    tags=["语音服务"]
)
async def synthesize_text(
    request: TextRequest = Body(
        ...,
        example={"text": "你好，世界！Hello, World!"},
        description="要转换为语音的文本内容"
    )
):
    """
    将文本转换为语音。

    ## 参数说明
    * text: 要转换的文本内容
        - 支持中英文混合
        - 建议长度不超过1000字符
        - 支持常见标点符号

    ## 返回说明
    * 音频文件流（wav格式）
        - 采样率：24kHz
        - 位深度：16bit
        - 声道：单声道

    ## 示例
    ```python
    import requests
    
    url = "http://localhost:8000/synthesize"
    data = {"text": "你好，世界！Hello, World!"}
    response = requests.post(url, json=data)
    
    # 保存音频文件
    with open("output.wav", "wb") as f:
        f.write(response.content)
    ```

    ## 错误说明
    * 400: 文本为空或格式错误
    * 500: 服务器处理错误
    """
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is empty")

        # 生成唯一的临时文件名
        output_file = os.path.join(tempfile.gettempdir(), f"tts_output_{uuid.uuid4().hex[:8]}.wav")
        
        try:
            tts_model(
                text=request.text,
                output=output_file,
                device=device
            )
            
            if not os.path.exists(output_file):
                raise HTTPException(status_code=500, detail="Failed to generate audio file")
            
            def iterfile():
                with open(output_file, mode="rb") as file_like:
                    yield from file_like
                    
            return StreamingResponse(
                iterfile(),
                media_type="audio/wav",
                headers={
                    'Content-Disposition': f'attachment; filename="speech.wav"'
                }
            )
            
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try:
                if os.path.exists(output_file):
                    os.remove(output_file)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))