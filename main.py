# 主要运行入口
# 这些代码必须放在所有其他导入之前
import warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import uvicorn
from code.main_code.fast_api_service import app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)