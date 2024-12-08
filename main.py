import os
from pathlib import Path
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

def create_project_structure(base_dir, structure):
    """
    递归创建项目目录结构和文件。
    
    参数:
    - base_dir (Path): 项目根目录路径
    - structure (dict): 项目结构字典
    """
    # 定义项目结构和示例内容
   
    for name, content in structure.items():
        path = base_dir / name
        if isinstance(content, dict):
            # 如果内容是字典，说明是文件夹
            create_directory(path)
            # 递归调用
            create_project_structure(path, content)
        else:
            # 如果内容不是字典，说明是文件
            create_file(path, content)

def initialize_git_repo(base_dir):
    """初始化Git仓库并添加初始提交。"""
    try:
        # 初始化Git仓库
        subprocess.run(['git', 'init'], cwd=base_dir, check=True)
        print("Git仓库已初始化。")
        
        # 创建 .gitignore 文件
        gitignore_path = base_dir / '.gitignore'
        gitignore_content = 'helloworld'
        create_file(gitignore_path, gitignore_content)
        
        # 添加所有文件到Git
        subprocess.run(['git', 'add', '.'], cwd=base_dir, check=True)
        print("所有文件已添加到Git。")
        
        # 初始提交
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=base_dir, check=True)
        print("初始提交已完成。")
    except subprocess.CalledProcessError as e:
        print(f"Git操作失败: {e}")


def main():
    from pathlib import Path

    # 获取当前工作目录作为项目根目录
    base_directory = Path.cwd()
    
    # 创建项目目录结构和文件
    structure = {
        '数据': {
            '原始数据': {},
            '清洗数据': {},
            '数据说明文档': {
                '数据说明文档模板.md': ''
            },
            '数据可视化': {
                '示例图表.png': ''  # 可以手动添加图片
            }
        },
        '特征工程': {
            '特征选择': {},
            '特征构建': {},
            '特征描述文档': {
                '特征描述文档模板.md': ''
            }
        },
        '模型': {
            '训练脚本': {
                'train_model.py': ''
            },
            '调参结果': {},
            '模型保存': {},
            '模型评估': {
                'evaluate_model.py': ''
            }
        },
        '评估报告': {
            '指标报告': {
                '指标报告模板.md': ''
            },
            '可视化图表': {
                '示例图表.png': ''  # 可以手动添加图片
            },
            '综合分析报告': {
                '综合分析报告模板.md': ''
            }
        },
        '代码': {
            '主代码': {
                'main.py': ''
            },
            '模块代码': {
                '__init__.py': '',
                'data_loader.py': ''
            },
            '工具脚本': {
                '__init__.py': '',
                'utils.py': ''
            }
        },
        '文档': {
            '项目计划': {
                '项目计划模板.md': ''
            },
            '会议记录': {
                '会议记录模板.md': ''
            },
            '用户手册': {
                '用户手册模板.md': ''
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
                '实验1记录.md': ''
            },
            '实验2': {
                '实验2记录.md': ''
            },
            '实验总结': {
                '实验总结模板.md': ''
            }
        },
        '环境': {
            'requirements.txt': '',
            '环境配置说明': {
                '配置说明.md': ''
            }
        },
        'README.md': ''
    }
    create_project_structure(base_directory, structure)
    
    # 初始化Git仓库并进行初始提交
    initialize_git_repo(base_directory)
    
    print("\n项目目录结构、示例文件和Git仓库已成功生成。")

if __name__ == "__main__":
    main()