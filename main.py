# 主要运行入口
# 这些代码必须放在所有其他导入之前
import warnings
import logging
import uvicorn
import os
import ssl

# 过滤警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置日志级别
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

# 导入应用
from code.main_code.fast_api_service import app

if __name__ == "__main__":
    # 获取本机IP地址
    host = "0.0.0.0"  # 允许外部访问
    port = 8000
    
    # 检查SSL证书
    ssl_dir = os.path.join(os.path.dirname(__file__), "ssl")
    cert_path = os.path.join(ssl_dir, "cert.pem")
    key_path = os.path.join(ssl_dir, "key.pem")
    
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        print("\n未找到SSL证书，将使用HTTP模式运行。")
        print("注意：在非HTTPS模式下，其他设备访问时可能无法使用麦克风。")
        print("要启用HTTPS：")
        print("1. 运行 ssl/generate_cert.bat 生成证书")
        print("2. 重启服务器\n")
        ssl_context = None
    else:
        print("\n找到SSL证书，将使用HTTPS模式运行。")
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(cert_path, key_path)
    
    print(f"\n{'='*50}")
    protocol = "https" if ssl_context else "http"
    print(f"Server running at {protocol}://{host}:{port}")
    print(f"For local access, use: {protocol}://localhost:{port}")
    print(f"For external access, use your machine's IP address")
    print(f"{'='*50}\n")
    
    # 启动服务器
    uvicorn.run(
        "code.main_code.fast_api_service:app",
        host=host,
        port=port,
        reload=True,
        log_level="error",
        ssl_keyfile=key_path if ssl_context else None,
        ssl_certfile=cert_path if ssl_context else None
    )