# ASR & TTS Web Application

一个基于 FastAPI 和 Vue 的在线语音识别和语音合成应用。

## 功能特性

- 实时语音录制和识别（ASR）
- 文本转语音合成（TTS）
- 支持中英文混合语音识别
- 支持多种音频格式（wav, mp3, ogg, flac）
- 跨平台和跨设备访问支持
- 提供完整的 RESTful API
- 美观的 Vue 3 用户界面
- 实时音频可视化

## 系统要求

- Python 3.8 或更高版本
- CUDA 支持（可选，用于 GPU 加速）
- Chrome 浏览器（推荐）

## 安装步骤

1. 克隆项目：
```bash
git clone [项目地址]
cd asr
```

2. 创建虚拟环境：
```bash
# 使用 environment.yml（推荐）
conda env create -f environment.yml

# 或者使用 requirements.txt
conda create --name new_env --file asr_tts_environment.txt
```

3. 激活环境：
```bash
conda activate [环境名称]
```

## 启动应用

1. 启动后端服务：
```bash
python main.py
```

2. 启动前端服务：
```bash
cd code/js_src/
python -m http.server 8001
```

3. 访问应用：
- 前端界面：`http://localhost:8001`
- API 文档：`http://localhost:8000/docs` 或 `http://localhost:8000/redoc`
- Swagger UI：`http://localhost:8000/docs`

## API 文档

### 语音识别 API

**接口**：`/transcribe`
- 方法：POST
- 功能：将语音文件转换为文本
- 支持格式：wav, mp3, ogg, flac
- 示例：
```python
import requests

url = "http://localhost:8000/transcribe"
files = {"audio_file": open("audio.wav", "rb")}
response = requests.post(url, files=files)
text = response.json()["text"]
```

### 语音合成 API

**接口**：`/synthesize`
- 方法：POST
- 功能：将文本转换为语音
- 支持：中英文混合文本
- 示例：
```python
import requests

url = "http://localhost:8000/synthesize"
data = {"text": "你好，世界！Hello, World!"}
response = requests.post(url, json=data)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## 跨设备访问设置

要在其他设备上访问并使用麦克风功能，需要在 Chrome 浏览器中进行以下设置：

1. 访问：`chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. 找到 "Insecure origins treated as secure" 设置
3. 添加你的服务器地址（例如：`http://192.168.1.xxx:8000`）
4. 点击 "Relaunch" 重启浏览器
5. 重新访问应用

## 技术栈

### 后端
- FastAPI：Web 框架
- Whisper：语音识别模型
- PaddleSpeech：语音合成引擎
- PyTorch：深度学习框架
- Pydub：音频处理

### 前端
- Vue 3：前端框架
- Web Audio API：音频录制和处理
- JavaScript MediaRecorder：音频录制
- Canvas：音频可视化

## 更新日志

### 2025-1-2
- 添加了完整的 API 文档
- 优化了错误处理
- 改进了跨域支持

### 2025-1-1
- 初版提交
- 包括 ASR 和 TTS 两个模块的前后端代码
- 用 Vue 重构了原生的前端代码

## 注意事项

- 确保麦克风设备正常工作并已授权
- 音频文件大小建议不超过 10MB
- 语音合成文本长度建议不超过 1000 字符
- 建议使用 Chrome 浏览器以获得最佳体验
- 首次使用需要允许浏览器访问麦克风

## 常见问题

1. **麦克风无法使用**
   - 检查浏览器权限设置
   - 确认是否已配置跨设备访问设置
   - 验证设备驱动是否正确安装

2. **识别结果不准确**
   - 确保录音环境安静
   - 检查麦克风音量设置
   - 尽量使用高质量麦克风

3. **跨设备访问问题**
   - 确保设备在同一网络下
   - 检查防火墙设置
   - 按照跨设备访问设置正确配置

