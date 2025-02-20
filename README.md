# ASR & TTS Web Application

一个基于 FastAPI 和 PaddleSpeech 的在线语音识别和语音合成应用。

## 功能特性

- 实时语音录制和识别（ASR）
- 文本转语音合成（TTS）
- 支持中英文混合语音识别
- 支持多种音频格式（wav, mp3, webm）
- 跨平台和跨设备访问支持
- 提供完整的 RESTful API
- 美观的用户界面
- 实时音频可视化

## 系统架构

### 前端技术栈
- HTML5 + CSS3 + JavaScript
- 原生 Web Audio API 用于音频录制
- Material Design Icons 图标库
- 响应式设计，支持移动设备

### 后端技术栈
- FastAPI：高性能 Web 框架
- PaddleSpeech：飞桨语音模型库
  - ASR：语音识别模型
  - TTS：语音合成模型
  - Text：文本处理模型（标点符号预测）
- FFmpeg：音频格式转换

## 系统要求

- Python 3.8 或更高版本
- CUDA 支持（可选，用于 GPU 加速）
- Chrome 浏览器（推荐）
- FFmpeg（用于音频处理）

## 安装步骤

1. 安装 Python 依赖：
```bash
pip install -r requirements.txt
```

2. 安装 FFmpeg（如果尚未安装）：
```bash
# Windows (使用 chocolatey)
choco install ffmpeg

# Linux
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## 启动应用

1. 启动服务器：
```bash
python -m uvicorn code.main_code.fast_api_service:app --reload --host 0.0.0.0 --port 8000
```

2. 访问应用：
- 本地访问：`http://localhost:8000`
- 局域网访问：`http://你的IP:8000`

## 跨设备访问设置

要在局域网其他设备上使用麦克风功能，需要在 Chrome 浏览器中进行以下设置：

1. 在地址栏输入：`chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. 找到 "Insecure origins treated as secure" 选项
3. 将应用地址 `http://你的IP:8000` 添加到输入框中
4. 将选项改为 "Enabled"
5. 点击底部的 "Relaunch" 按钮重启浏览器

## API 文档

### 语音识别 API

**接口**：`/transcribe`
- 方法：POST
- 功能：将语音文件转换为文本
- 支持格式：wav, mp3, webm
- 参数：
  - audio_file：音频文件（multipart/form-data）
- 返回：
  ```json
  {
    "text": "识别的文本内容"
  }
  ```

### 语音合成 API

**接口**：`/synthesize`
- 方法：POST
- 功能：将文本转换为语音
- 参数：
  ```json
  {
    "text": "要转换的文本"
  }
  ```
- 返回：音频文件（WAV格式）

## 文件结构

```
asr/
├── code/
│   └── main_code/
│       └── fast_api_service.py  # 主服务器代码
├── static/
│   └── index.html              # 前端界面
└── requirements.txt            # Python 依赖
```

## 性能优化

1. 音频处理优化：
   - 使用 FFmpeg 进行高效的音频格式转换
   - 统一使用 16kHz 采样率和单声道

2. 前端优化：
   - 简化音频录制流程
   - 优化音频数据传输格式
   - 添加录音大小检查

3. 后端优化：
   - 使用异步处理提高并发性能
   - 添加临时文件自动清理
   - 完善的错误处理和日志记录

## 注意事项

1. 首次启动时模型加载可能较慢
2. 建议使用 Chrome 浏览器以获得最佳体验
3. 录音最长支持 30 秒
4. 使用 GPU 可以显著提升识别速度

## 常见问题

1. Q: 为什么其他设备访问时无法使用麦克风？
   A: 需要在 Chrome 浏览器中设置允许不安全源使用麦克风，详见"跨设备访问设置"。

2. Q: 为什么第一次识别较慢？
   A: 首次识别时需要加载模型，之后的识别会更快。

3. Q: 支持哪些语言？
   A: 目前主要支持中文和英文的混合识别。

## 更新日志

### 2025-02-20
- 优化音频录制流程，提高跨设备兼容性
- 简化音频格式处理，移除复杂的重采样过程
- 改进错误处理和日志记录
- 添加详细的跨设备访问说明

### 2025-02-19
- 添加麦克风设备测试功能
- 优化前端界面，添加录音倒计时显示
- 改进语音识别结果的显示方式
- 添加自动朗读识别结果功能

### 2025-02-18
- 初始版本发布
- 实现基本的语音识别和合成功能
- 支持局域网访问
- 添加基本的用户界面
