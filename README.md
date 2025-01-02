环境配置：
conda create --name new_env --file asr_tts_environment.txt

或者
conda env create -f environment.yml（信息更全）

2025-1-1
初版提交:包括ASR和TTS两个模块的前后端代码

项目地址：
http://localhost:8001/

前端代码启动命令
cd .\code\js_src\
python -m http.server 8001

后端代码启动命令
python .\main.py


