import os
import logging
import tempfile
import warnings
from typing import Optional
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
import paddle
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.cli.text.infer import TextExecutor
from pydantic import BaseModel
import uuid
from io import BytesIO
import subprocess

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 过滤警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class TextRequest(BaseModel):
    text: str

class TranscriptionResponse(BaseModel):
    text: str

# 创建 FastAPI 应用
app = FastAPI(
    title="语音识别与合成服务",
    description="基于 PaddleSpeech 的语音识别与合成 API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化执行器
asr_executor = None
tts_executor = None
text_executor = None

def get_asr_executor():
    global asr_executor
    if asr_executor is None:
        logger.info("Initializing ASR executor...")
        try:
            asr_executor = ASRExecutor()
            logger.info("ASR executor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ASR executor: {str(e)}")
            raise
    return asr_executor

def get_tts_executor():
    global tts_executor
    if tts_executor is None:
        logger.info("Initializing TTS executor...")
        try:
            tts_executor = TTSExecutor()
            logger.info("TTS executor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TTS executor: {str(e)}")
            raise
    return tts_executor

def get_text_executor():
    global text_executor
    if text_executor is None:
        logger.info("Initializing Text executor...")
        try:
            text_executor = TextExecutor()
            logger.info("Text executor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Text executor: {str(e)}")
            raise
    return text_executor

def convert_audio_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    使用ffmpeg直接转换音频文件为16kHz采样率的WAV格式
    """
    try:
        # 检查输入文件
        if not os.path.exists(input_path):
            logger.error(f"Input file does not exist: {input_path}")
            return False
            
        # 检查ffmpeg是否可用
        try:
            version_cmd = ['ffmpeg', '-version']
            version_process = subprocess.Popen(
                version_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            version_stdout, version_stderr = version_process.communicate()
            if version_process.returncode != 0:
                logger.error("FFmpeg is not available")
                return False
            logger.info("FFmpeg is available")
        except Exception as e:
            logger.error(f"Error checking ffmpeg: {str(e)}")
            return False

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',  # 覆盖已存在的文件
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # 执行命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        # 检查命令是否成功执行
        if process.returncode != 0:
            logger.error(f"FFmpeg error: {stderr.decode()}")
            return False
            
        # 检查输出文件
        if not os.path.exists(output_path):
            logger.error(f"Output file was not created: {output_path}")
            return False
            
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            logger.error(f"Output file is empty: {output_path}")
            return False
            
        logger.info(f"Successfully converted audio to: {output_path} (size: {file_size} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"Error running ffmpeg: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(None)
):
    """
    接收音频文件并进行语音识别
    支持格式：wav, mp3, webm 等
    """
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")
        
    temp_files = []
    temp_dir = None
    
    try:
        logger.info(f"Received audio file: {audio_file.filename}, content_type: {audio_file.content_type}")
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='asr_')
        logger.info(f"Created temp directory: {temp_dir}")
        
        # 保存上传的文件
        input_path = os.path.join(temp_dir, 'input' + os.path.splitext(audio_file.filename)[1])
        content = await audio_file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file")
            
        logger.info(f"Read {len(content)} bytes from uploaded file")
        
        # 写入原始文件
        with open(input_path, 'wb') as f:
            f.write(content)
        temp_files.append(input_path)
        
        # 检查文件
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            raise ValueError("Failed to save input file or file is empty")
            
        logger.info(f"Saved input file: {input_path}, size: {os.path.getsize(input_path)} bytes")
        
        # 设置输出WAV文件路径
        wav_path = os.path.join(temp_dir, 'output.wav')
        temp_files.append(wav_path)
        
        # 使用ffmpeg转换音频
        logger.info("Converting audio using ffmpeg...")
        if not convert_audio_ffmpeg(input_path, wav_path):
            raise ValueError("Failed to convert audio using ffmpeg")
            
        # 验证转换后的文件
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise ValueError("WAV file not created or empty")
            
        logger.info(f"Created WAV file: {wav_path}, size: {os.path.getsize(wav_path)} bytes")
        
        # 检查音频文件的格式
        try:
            import wave
            with wave.open(wav_path, 'rb') as wav_file:
                params = wav_file.getparams()
                logger.info(f"WAV file parameters: channels={params.nchannels}, "
                          f"sampwidth={params.sampwidth}, framerate={params.framerate}, "
                          f"nframes={params.nframes}")
        except Exception as e:
            logger.error(f"Error checking WAV file: {str(e)}")
            
        # 设置文件权限
        os.chmod(wav_path, 0o644)
        
        # 执行语音识别
        logger.info("Starting speech recognition...")
        asr = get_asr_executor()
        result = asr(audio_file=wav_path)
        logger.info(f"Raw recognition result: {result}")
        
        # 添加标点符号
        if result:
            text_punc = get_text_executor()
            result = text_punc(text=result)
            logger.info(f"Text with punctuation: {result}")
        else:
            logger.warning("Speech recognition returned empty result")
        
        # 确保结果是字符串
        if not isinstance(result, str):
            result = str(result)
            logger.info(f"Converted result to string: {result}")
        
        return JSONResponse(content={"text": result})
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.chmod(temp_file, 0o644)  # 确保有删除权限
                    os.unlink(temp_file)
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {str(e)}")
        
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {str(e)}")

@app.post("/synthesize")
async def synthesize_text(text: TextRequest):
    """
    将文本转换为语音
    返回音频文件的字节流
    """
    try:
        logger.info(f"Synthesizing text: {text.text}")
        
        # 创建临时目录用于存储合成的音频
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")

        # 使用 PaddleSpeech 进行语音合成
        tts = get_tts_executor()
        tts(text=text.text, output=output_file)

        logger.info(f"语音合成完成，输出文件：{output_file}")
        
        # 读取音频文件并返回
        def iterfile():
            with open(output_file, 'rb') as f:
                yield from f
            # 读取完成后删除临时文件
            try:
                os.unlink(output_file)
                logger.info(f"Cleaned up temp file: {output_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {output_file}: {str(e)}")

        return StreamingResponse(
            iterfile(),
            media_type="audio/wav"
        )
        
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_punctuation")
async def add_punctuation(text: TextRequest):
    """
    为文本添加标点符号
    """
    try:
        logger.info(f"Adding punctuation to text: {text.text}")
        
        # 使用 PaddleSpeech 添加标点
        text_punc = get_text_executor()
        result = text_punc(text=text.text)
        
        logger.info(f"Text with punctuation: {result}")
        
        return JSONResponse(content={"text": result})
        
    except Exception as e:
        logger.error(f"Error adding punctuation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "code.main_code.fast_api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )