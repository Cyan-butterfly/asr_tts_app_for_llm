
from paddlespeech.cli.tts.infer import TTSExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
import io
import os
from pydub import AudioSegment
import wave
import tempfile
import logging

import numpy as np


# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 配置 CORS
origins = [
    "http://localhost",
    "http://localhost:8001",  # 前端静态服务器端口
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


# 加载 PaddleSpeech TTS 模型
try:
    tts_model = TTSExecutor()
    logger.info("PaddleSpeech TTS model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to initialize PaddleSpeech TTS model: {e}")
    raise RuntimeError("PaddleSpeech TTS model initialization failed.")
# 加载 Paddlespeech TTS 模型
# tts_model = Text2Speech.get_model("fastspeech2_ljspeech", device=device)
# 根据需要选择合适的中文模型
# tts_model = Text2Speech.get_model("fastspeech2_chinese", device=device)

# 加载 Wav2Vec2 模型
# wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(device)

def convert_to_wav(file_content: bytes, file_format: str, output_file: str, **kwargs):
    if file_format == "mp3":
        audio = AudioSegment.from_file(io.BytesIO(file_content), format="mp3")
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
        raise ValueError("Unsupported file format.")
    return output_file

# windsurf修改
@app.post("/transcribe/")
async def transcribe_whisper(file: UploadFile = File(...)):
    try:
        # 读取文件内容
        contents = await file.read()
        logger.info(f"Received file: {file.filename}, size: {len(contents)} bytes")

        if not isinstance(contents, bytes):
            logger.error("File content is not in bytes format")
            raise ValueError("File content is not in bytes format.")
        
        filename = file.filename.lower()
        logger.info(f"Processing file: {filename}")

        # 使用临时文件处理
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            try:
                if filename.endswith((".mp3", ".webm", ".ogg")):
                    logger.info(f"Converting {filename} to WAV format")
                    convert_to_wav(contents, filename.split('.')[-1], temp.name)
                elif filename.endswith(".wav"):
                    temp.write(contents)
                else:
                    logger.error(f"Unsupported file format: {filename}")
                    raise ValueError(f"Unsupported file format: {filename}")
                
                logger.info(f"Temporary file created: {temp.name}")
                
                # 检查文件是否存在
                if not os.path.exists(temp.name):
                    logger.error("WAV file could not be processed")
                    raise ValueError("WAV file could not be processed.")

                # 读取音频文件
                audio, sr = librosa.load(temp.name, sr=16000)
                logger.info(f"Audio loaded: shape={audio.shape}, sr={sr}")

                input_features = whisper_processor(audio, return_tensors="pt", sampling_rate=16000).input_features.to(device)
                logger.info(f"Input features shape: {input_features.shape}")

                # 模型推理
                with torch.no_grad():
                    generated_ids = whisper_model.generate(input_features)
                
                # 解码文本
                transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                logger.info(f"Transcription completed: {transcription}")

                return {"transcription": transcription}

            except Exception as e:
                logger.error(f"Error during audio processing: {str(e)}", exc_info=True)
                raise
            finally:
                # 清理临时文件
                try:
                    os.remove(temp.name)
                    logger.info("Temporary file cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {str(e)}")

    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



# @app.post("/transcribe/")
# async def transcribe_whisper(file: UploadFile = File(...)):
#     try:
#         # 读取文件内容
#         contents = await file.read()
#         logger.info(f"Received file: {file.filename}, size: {len(contents)} bytes")

#         if not isinstance(contents, bytes):
#             raise ValueError("File content is not in bytes format.")
#         # 在文件读取后
#         logger.info(f"File format: {file.filename}")
#         filename = file.filename.lower()

#         # 使用临时文件处理
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
#             if filename.endswith(".mp3"):
#                 convert_to_wav(contents, "mp3", temp.name)
#             elif filename.endswith(".pcm"):
#                 convert_to_wav(contents, "pcm", temp.name)
#             elif filename.endswith(".wav"):
#                 temp.write(contents)
#             else:
#                 raise ValueError("Unsupported file format. Please upload MP3, PCM, or WAV files.")
        
#         wav_file = temp.name

#         # 检查文件是否存在
#         if not os.path.exists(wav_file):
#             raise ValueError("WAV file could not be processed.")

#         # 读取音频文件
#         audio, sr = librosa.load(wav_file, sr=16000)  # 采样率为16kHz
#         input_features = whisper_processor(audio, return_tensors="pt", sampling_rate=16000).input_features.to(device)
#         # 在模型推理前
#         logger.info(f"Input features shape: {input_features.shape}")
#         # 在音频加载后
#         logger.info(f"Audio loaded: shape={audio.shape}, sr={sr}")

#         # 模型推理
#         with torch.no_grad():
#             generated_ids = whisper_model.generate(input_features)
        
#         # 解码文本
#         transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#         logger.info(f"Transcription: {transcription}")

#         # 删除临时文件
#         os.remove(wav_file)

#         return {"transcription": transcription}

#     except Exception as e:
#         logger.error(f"Error during transcription: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/transcribe_wav2vec2/")
# async def transcribe_wav2vec2(file: UploadFile = File(...)):
#     try:
#         # 读取文件内容
#         contents = await file.read()
#         logger.info(f"Received file: {file.filename}, size: {len(contents)} bytes")

#         if not isinstance(contents, bytes):
#             raise ValueError("File content is not in bytes format.")
        
#         filename = file.filename.lower()

#         # 使用临时文件处理
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
#             if filename.endswith(".mp3"):
#                 convert_to_wav(contents, "mp3", temp.name)
#             elif filename.endswith(".pcm"):
#                 convert_to_wav(contents, "pcm", temp.name)
#             elif filename.endswith(".wav"):
#                 temp.write(contents)
#             else:
#                 raise ValueError("Unsupported file format. Please upload MP3, PCM, or WAV files.")
        
#         wav_file = temp.name

#         # 检查文件是否存在
#         if not os.path.exists(wav_file):
#             raise ValueError("WAV file could not be processed.")

#         # 读取音频文件
#         audio, sr = librosa.load(wav_file, sr=16000)  # 采样率为16kHz
#         input_values = wav2vec_processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)

#         # 模型推理
#         with torch.no_grad():
#             logits = wav2vec_model(input_values).logits

#         # 获取预测的ID
#         predicted_ids = torch.argmax(logits, dim=-1)

#         # 解码文本
#         transcription = wav2vec_processor.batch_decode(predicted_ids)[0]

#         logger.info(f"Transcription: {transcription}")

#         # 删除临时文件
#         os.remove(wav_file)

#         return {"transcription": transcription}

    # except Exception as e:
    #     logger.error(f"Error during transcription: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/")
async def synthesize_text(text: str):
    try:
        if not text:
            raise ValueError("Text is empty.")

        logger.info(f"Received text for synthesis: {text}")

        # 使用 TTS 模型合成语音
        audio = tts_model(text=text, output=None)  # PaddleSpeech TTS 返回 numpy 数组
        # audio 是一个 numpy 数组，代表音频信号

        # 将 numpy 数组转换为 bytes（WAV 格式）
        # 使用 soundfile 库更可靠
        import soundfile as sf

        with io.BytesIO() as wav_io:
            sf.write(wav_io, audio, samplerate=tts_model.samplerate, format='WAV')
            wav_io.seek(0)
            audio_bytes = wav_io.read()

        # 返回音频流
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))