# vosk_server.py
import os
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from vosk import Model, KaldiRecognizer
import uvicorn
import logging
import wave
import uuid
import tempfile  # 引入 tempfile 模块

# 初始化日志
logging.basicConfig(
    level=logging.INFO,  # 确保日志级别为 INFO
    format='[%(asctime)s] [%(levelname)s] [%(message)s]',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI(
    title="Vosk Speech Recognition Service",
    description="A dedicated FastAPI service for Vosk-based speech recognition.",
    version="1.0.0"
)

# Vosk 模型路径
MODEL_PATH = os.getenv("VOSK_MODEL_PATH", r"D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\vosk\vosk-model-cn-0.22")  # 更新为实际路径

# API Key 用于认证
API_KEY = os.getenv("API_KEY", "ksgnsapf91321")  # 替换为实际的 API 密钥

# 验证 API Key 的依赖
async def verify_api_key(x_api_key: str = Header(...)):
    logger.info("Verifying API key...")
    if x_api_key != API_KEY:
        logger.warning("Invalid API key received.")
        raise HTTPException(status_code=403, detail="Forbidden")
    logger.info("API key verified successfully.")

# 加载 Vosk 模型
if not os.path.exists(MODEL_PATH):
    logger.error(f"Vosk model not found at: {MODEL_PATH}")
    raise FileNotFoundError(f"Vosk model not found at: {MODEL_PATH}")

logger.info(f"Loading Vosk model from: {MODEL_PATH}")
try:
    model = Model(MODEL_PATH)
    logger.info("Vosk model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Vosk model: {str(e)}")
    raise

@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="WAV audio file for transcription"),
    x_api_key: str = Header(...)
):
    await verify_api_key(x_api_key)
    try:
        logger.info(f"Received file: {audio_file.filename}")

        # 确保上传的文件是 WAV 格式
        if not audio_file.filename.lower().endswith(".wav"):
            logger.error("Uploaded file is not a WAV file.")
            raise HTTPException(status_code=400, detail="Only WAV files are supported.")

        # 使用 tempfile 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        logger.info(f"Saved uploaded file to: {temp_file_path}")

        # 打开 WAV 文件
        wf = wave.open(temp_file_path, "rb")
        logger.info("Opened WAV file successfully.")

        # 确保音频格式符合 Vosk 的要求
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            wf.close()
            os.remove(temp_file_path)
            logger.error("Audio file format mismatch.")
            raise HTTPException(status_code=400, detail="Audio file must be WAV format mono PCM with 16kHz sample rate.")

        # 初始化识别器
        rec = KaldiRecognizer(model, wf.getframerate())
        logger.info("KaldiRecognizer initialized.")

        # 读取数据块进行识别
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                logger.info(f"Partial transcription: {res.get('text', '')}")
            else:
                partial = json.loads(rec.PartialResult())
                logger.info(f"Partial result: {partial.get('partial', '')}")

        # 获取最终结果
        final_result = rec.FinalResult()
        transcription = json.loads(final_result).get("text", "")
        logger.info(f"Transcription result: {transcription}")

        # 清理临时文件
        wf.close()
        os.remove(temp_file_path)
        logger.info("Cleaned up temporary files.")

        return JSONResponse(content={"text": transcription})

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    logger.info("Health check requested.")
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting Vosk server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
