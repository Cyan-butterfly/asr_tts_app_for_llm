import os
from pathlib import Path
import json
import subprocess

def create_directory(path):
    """
    创建目录，如果目录不存在则创建。
    
    参数:
    - path (Path): 要创建的目录路径
    """
    path.mkdir(parents=True, exist_ok=True)
    print(f"创建目录: {path}")

def create_file(path, content=""):
    """
    创建文件并写入内容，如果文件已存在则跳过。
    
    参数:
    - path (Path): 要创建的文件路径
    - content (str): 要写入文件的内容
    """
    if not path.exists():
        with path.open('w', encoding='utf-8') as f:
            f.write(content)
        print(f"创建文件: {path}")
    else:
        print(f"文件已存在: {path}")

def generate_project_structure(base_dir):
    """
    根据预设的项目结构生成文件夹和示例文件。
    
    参数:
    - base_dir (Path): 项目根目录路径
    """
    # 定义项目结构和示例内容
    project_structure = {
        '数据': {
            '原始数据': {},
            '清洗数据': {},
            '数据说明文档': {
                '数据说明文档模板.md': '# 数据说明文档\n\n这里编写数据说明文档。'
            },
            '数据可视化': {
                '示例图表.png': ''  # 可以手动添加图片
            }
        },
        '特征工程': {
            '特征选择': {},
            '特征构建': {},
            '特征描述文档': {
                '特征描述文档模板.md': '# 特征描述文档\n\n这里编写特征描述文档。'
            }
        },
        '模型': {
            '训练脚本': {
                'train_model.py': '''# 训练脚本示例
def train_model():
    pass

if __name__ == "__main__":
    train_model()
'''
            },
            '调参结果': {},
            '模型保存': {},
            '模型评估': {
                'evaluate_model.py': '''# 模型评估脚本示例
def evaluate_model():
    pass

if __name__ == "__main__":
    evaluate_model()
'''
            }
        },
        '评估报告': {
            '指标报告': {
                '指标报告模板.md': '# 指标报告\n\n这里编写指标报告。'
            },
            '可视化图表': {
                '示例图表.png': ''  # 可以手动添加图片
            },
            '综合分析报告': {
                '综合分析报告模板.md': '# 综合分析报告\n\n这里编写综合分析报告。'
            }
        },
        '代码': {
            '主代码': {
                'main.py': '''# 主代码示例
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
'''
            },
            '模块代码': {
                '__init__.py': '',
                'data_loader.py': '''# 模块代码示例：数据加载
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
'''
            },
            '工具脚本': {
                '__init__.py': '',
                'utils.py': '''# 工具脚本示例
def helper_function():
    pass
'''
            }
        },
        '文档': {
            '项目计划': {
                '项目计划模板.md': '# 项目计划\n\n这里编写项目计划。'
            },
            '会议记录': {
                '会议记录模板.md': '# 会议记录\n\n这里编写会议记录。'
            },
            '用户手册': {
                '用户手册模板.md': '# 用户手册\n\n这里编写用户手册。'
            }
        },
        '分享材料': {
            'PPT': {
                '示例PPT.pptx': ''  # 可以手动添加PPT文件
            },
            '演示视频': {
                '示例视频.mp4': ''  # 可以手动添加视频文件
            }
        },
        '实验记录': {
            '实验1': {
                '实验1记录.md': '# 实验1\n\n这里记录实验1的内容。'
            },
            '实验2': {
                '实验2记录.md': '# 实验2\n\n这里记录实验2的内容。'
            },
            '实验总结': {
                '实验总结模板.md': '# 实验总结\n\n这里编写实验总结。'
            }
        },
        '环境': {
            'requirements.txt': '''# 项目依赖
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.0
matplotlib==3.7.1
seaborn==0.12.2
pyyaml==6.0
''',
            '环境配置说明': {
                '配置说明.md': '# 环境配置说明\n\n这里编写环境配置的说明。'
            }
        },
        'README.md': '''# 项目名称
    
这是一个用于测站水文预报的项目，包含数据处理、特征工程、模型训练与评估等模块。
    
## 目录结构
    
