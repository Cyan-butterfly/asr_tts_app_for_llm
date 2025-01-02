from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
import io
import os
from pydub import AudioSegment
import wave
import tempfile
import logging
import shutil

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 配置 CORS
origins = [
    "http://localhost",
    "http://localhost:8001",  # 前端静态服务器端口
    # 添加您的前端域名
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 或者设置为 ["*"] 允许所有
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置环境变量禁用代理： 在运行程序之前，先设置以下环境变量
os.environ['NO_PROXY'] = 'huggingface.co'

# 加载 Whisper 模型

whisper_storage_path = r"D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\whisper-small"
whisper_processor = AutoProcessor.from_pretrained(whisper_storage_path)
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_storage_path).to(device)

# 如果需要加载 Wav2Vec2 模型，可以在此处加载
# wav2vec_processor = Wav2Vec2Processor.from_pretrained("path_to_wav2vec2_model")
# wav2vec_model = Wav2Vec2ForCTC.from_pretrained("path_to_wav2vec2_model").to(device)

# windsurf修改
def convert_to_wav(file_content: bytes, file_format: str, output_file: str, **kwargs):
    try:
        logger.info(f"Converting from {file_format} to WAV")
        if file_format in ["mp3", "webm", "ogg"]:
            audio = AudioSegment.from_file(io.BytesIO(file_content), format=file_format)
            audio.export(output_file, format="wav")
        elif file_format == "pcm":
            channels = kwargs.get("channels", 1)
            sampwidth = kwargs.get("sampwidth", 2)
            framerate = kwargs.get("framerate", 16000)
            with wave.open(output_file, "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sampwidth)
                wav_file.setframerate(framerate)
                wav_file.writeframes(file_content)
        elif file_format == "wav":
            with open(output_file, "wb") as f:
                f.write(file_content)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Successfully converted to WAV: {output_file}")
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}", exc_info=True)
        raise


# def convert_to_wav(file_content: bytes, file_format: str, output_file: str, **kwargs):
#     if file_format == "mp3":
#         audio = AudioSegment.from_file(io.BytesIO(file_content), format="mp3")
#         audio.export(output_file, format="wav")
#     elif file_format == "pcm":
#         channels = kwargs.get("channels", 1)
#         sampwidth = kwargs.get("sampwidth", 2)
#         framerate = kwargs.get("framerate", 16000)
#         with wave.open(output_file, "wb") as wav_file:
#             wav_file.setnchannels(channels)
#             wav_file.setsampwidth(sampwidth)
#             wav_file.setframerate(framerate)
#             wav_file.writeframes(file_content)
#     elif file_format == "wav":
#         with open(output_file, "wb") as f:
#             f.write(file_content)
#     else:
#         raise ValueError("Unsupported file format.")
#     return output_file

@app.post("/transcribe/")
async def transcribe_whisper(file: UploadFile = File(...)):
    try:
        # 读取文件内容
        contents = await file.read()
        logger.info(f"Received file: {file.filename}, size: {len(contents)} bytes")

        if not isinstance(contents, bytes):
            raise ValueError("File content is not in bytes format.")
        
        filename = file.filename.lower()

        # 使用临时文件处理
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            if filename.endswith(".mp3"):
                convert_to_wav(contents, "mp3", temp.name)
            elif filename.endswith(".pcm"):
                convert_to_wav(contents, "pcm", temp.name)
            elif filename.endswith(".wav"):
                temp.write(contents)
            else:
                raise ValueError("Unsupported file format. Please upload MP3, PCM, or WAV files.")
        
        wav_file = temp.name

        # 检查文件是否存在
        if not os.path.exists(wav_file):
            raise ValueError("WAV file could not be processed.")

        # 读取音频文件
        audio, sr = librosa.load(wav_file, sr=16000)  # 采样率为16kHz
        input_features = whisper_processor(audio, return_tensors="pt", sampling_rate=16000).input_features.to(device)

        # 模型推理
        with torch.no_grad():
            generated_ids = whisper_model.generate(input_features)
        
        # 解码文本
        transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info(f"Transcription: {transcription}")

        # 删除临时文件
        os.remove(wav_file)

        return {"transcription": transcription}

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
