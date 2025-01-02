# FastAPI 服务逻辑


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import AutoProcessor, AutoModelForPreTraining

import torch
import librosa
import io
import sys
import os
from code.module_code.data_loader import preprocess_audio
from pydub import AudioSegment
import wave


app = FastAPI()  # 初始化 FastAPI 应用

# 加载 Wav2Vec2 模型
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(r"D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\chinese-wav2vec2-large")
model = AutoModelForPreTraining.from_pretrained(r"D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\chinese-wav2vec2-large").to("cuda" if torch.cuda.is_available() else "cpu")


def convert_mp3_to_wav(mp3_file_content: bytes, output_file="output.wav"):
    # 将 MP3 文件字节流加载为音频
    audio = AudioSegment.from_file(io.BytesIO(mp3_file_content), format="mp3")
    # 保存为 WAV 格式
    audio.export(output_file, format="wav")
    return output_file

def convert_pcm_to_wav(pcm_file_content: bytes, output_file="output.wav", channels=1, sampwidth=2, framerate=16000):
    # 将 PCM 数据写入 WAV 文件
    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(channels)  # 通道数
        wav_file.setsampwidth(sampwidth)  # 采样宽度
        wav_file.setframerate(framerate)  # 采样率
        wav_file.writeframes(pcm_file_content)
    return output_file

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        # 确保文件读取为 bytes
        contents = await file.read()
        print(f"File type: {type(contents)}, File length: {len(contents)}")
        if not isinstance(contents, bytes):
            raise ValueError("File content is not in bytes format.")
        filename = file.filename

        # 初始化 wav_file 变量
        wav_file = None

        # 根据文件扩展名确定处理方式
        if filename.endswith(".mp3"):
            wav_file = convert_mp3_to_wav(contents)
        elif filename.endswith(".pcm"):
            wav_file = convert_pcm_to_wav(contents)
        elif filename.endswith(".wav"):
            wav_file = contents  # 直接使用字节数据

        else:
            return {"error": "Unsupported file format. Please upload MP3, PCM, or WAV files."}

        # 如果 wav_file 未正确赋值，抛出异常
        if not wav_file:
            raise ValueError("WAV file could not be processed.")

        # 预处理 WAV 文件并进行转录
        audio = preprocess_audio(wav_file)
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(model.device)

        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        return {"transcription": transcription}

    except Exception as e:
        return {"error": str(e)}, 500




# FastAPI 服务逻辑

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
import io
import os
from pydub import AudioSegment
import wave


app = FastAPI()  # 初始化 FastAPI 应用

# 加载 Wav2Vec2 模型
# processor = Wav2Vec2Processor.from_pretrained(
#     r"D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\chinese-wav2vec2-large"
# )
# model = Wav2Vec2ForCTC.from_pretrained(
#     r"D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\chinese-wav2vec2-large"
# ).to("cuda" if torch.cuda.is_available() else "cpu")

# 加载Wav2Vec2 模型2
# processor = Wav2Vec2Processor.from_pretrained("voidful/wav2vec2-large-xlsr-53-chinese")
# model = Wav2Vec2ForCTC.from_pretrained("voidful/wav2vec2-large-xlsr-53-chinese").to("cuda" if torch.cuda.is_available() else "cpu")
# 加载Whisper 模型
model_path = r"D:\BaiduSyncdisk\mywork\asr\model\pre-trained-models\whisper-small"


processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
# processor = WhisperProcessor.from_pretrained(model_path)
# model = WhisperForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

def convert_mp3_to_wav(mp3_file_content: bytes, output_file="output.wav"):
    # 将 MP3 文件字节流加载为音频
    audio = AudioSegment.from_file(io.BytesIO(mp3_file_content), format="mp3")
    # 保存为 WAV 格式
    audio.export(output_file, format="wav")
    return output_file


def convert_pcm_to_wav(pcm_file_content: bytes, output_file="output.wav", channels=1, sampwidth=2, framerate=16000):
    # 将 PCM 数据写入 WAV 文件
    with wave.open(output_file, "wb") as wav_file:
        wav_file.setnchannels(channels)  # 通道数
        wav_file.setsampwidth(sampwidth)  # 采样宽度
        wav_file.setframerate(framerate)  # 采样率
        wav_file.writeframes(pcm_file_content)
    return output_file


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        # 确保文件读取为 bytes
        contents = await file.read()
        print(f"File type: {type(contents)}, File length: {len(contents)}")
        if not isinstance(contents, bytes):
            raise ValueError("File content is not in bytes format.")
        filename = file.filename

        # 初始化 wav_file 变量
        wav_file = None

        # 根据文件扩展名确定处理方式
        if filename.endswith(".mp3"):
            wav_file = convert_mp3_to_wav(contents)
        elif filename.endswith(".pcm"):
            wav_file = convert_pcm_to_wav(contents)
        elif filename.endswith(".wav"):
            # 写入临时文件
            temp_file = "temp.wav"
            with open(temp_file, "wb") as f:
                f.write(contents)
            wav_file = temp_file
        else:
            return {"error": "Unsupported file format. Please upload MP3, PCM, or WAV files."}

        # 如果 wav_file 未正确赋值，抛出异常
        if not wav_file or not os.path.exists(wav_file):
            raise ValueError("WAV file could not be processed.")

        # 读取音频文件并进行处理
        audio, sr = librosa.load(wav_file, sr=16000)  # 采样率必须为 16kHz
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(model.device)

        # 执行模型推理
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        # 删除临时文件
        if wav_file == "temp.wav":
            os.remove(wav_file)

        return {"transcription": transcription}

    except Exception as e:
        return {"error": str(e)}, 500


@app.post("/transcribe/whisper")
async def transcribe(file: UploadFile = File(...)):
    try:
        # 确保文件读取为 bytes
        contents = await file.read()
        print(f"File type: {type(contents)}, File length: {len(contents)}")
        if not isinstance(contents, bytes):
            raise ValueError("File content is not in bytes format.")
        filename = file.filename

        # 初始化 wav_file 变量
        wav_file = None

        # 根据文件扩展名确定处理方式
        if filename.endswith(".mp3"):
            wav_file = convert_mp3_to_wav(contents)
        elif filename.endswith(".pcm"):
            wav_file = convert_pcm_to_wav(contents)
        elif filename.endswith(".wav"):
            # 写入临时文件
            temp_file = "temp.wav"
            with open(temp_file, "wb") as f:
                f.write(contents)
            wav_file = temp_file
        else:
            return {"error": "Unsupported file format. Please upload MP3, PCM, or WAV files."}

        # 如果 wav_file 未正确赋值，抛出异常
        if not wav_file or not os.path.exists(wav_file):
            raise ValueError("WAV file could not be processed.")

        # 读取音频文件并进行处理
        audio, sr = librosa.load(wav_file, sr=16000)  # 采样率必须为 16kHz
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000).to(model.device)

        # 使用 Whisper 模型生成文本
        with torch.no_grad():
            generated_ids = model.generate(inputs.input_features)

        # 解码生成的文本
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 删除临时文件
        if wav_file == "temp.wav":
            os.remove(wav_file)

        return {"transcription": transcription}

    except Exception as e:
        return {"error": str(e)}, 500