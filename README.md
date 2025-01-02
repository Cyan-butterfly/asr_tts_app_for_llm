
**环境配置**：
conda create --name new_env --file asr_tts_environment.txt

或者
conda env create -f environment.yml（信息更全）

**提交记录**
2025-1-1
初版提交:包括ASR和TTS两个模块的前后端代码，用vue重构了原生的前端代码

项目地址：
http://localhost:8001/

前端代码启动命令
cd .\code\js_src\
python -m http.server 8001

后端代码启动命令
python .\main.py

# ASR Web Application

一个基于 FastAPI 和 Vue 的在线语音识别应用。

## 功能特性

- 实时语音录制和识别
- 中文语音识别支持
- 语音合成（TTS）支持
- 跨设备访问支持

## 快速开始

1. 启动后端服务：
```bash
python main.py
```

2. 访问应用：
- 本地访问：http://localhost:8000
- 局域网访问：http://192.168.3.116:8000

## 跨设备访问设置

如果需要在其他设备上访问并使用麦克风功能，需要在 Chrome 浏览器中进行以下设置：

1. 在地址栏输入：`chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. 找到 "Insecure origins treated as secure" 设置
3. 添加你的网站地址（例如：`http://192.168.1.xxx:8000`）
4. 点击 "Relaunch" 重启浏览器
5. 重新访问网站，现在应该可以正常使用麦克风功能了

## 技术栈

- 后端：FastAPI + Whisper + PaddleSpeech
- 前端：Vue 3 + Web Audio API
- 语音识别：Whisper
- 语音合成：PaddleSpeech TTS

## 注意事项

- 确保麦克风设备正常工作
- 首次使用需要允许浏览器访问麦克风
- 建议使用 Chrome 浏览器以获得最佳体验
