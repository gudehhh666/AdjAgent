#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python代码执行工具模块

在独立子进程中执行 Python 代码。

注意（安全说明）：本模块通过子进程运行代码并支持超时与可选的资源限制
（内存 / CPU 时间 / core dump），但它**不是安全隔离沙箱**：子进程仍继承
当前环境变量并在当前工作目录下运行，可读写文件系统、发起网络请求。
请勿用它执行不受信任的第三方代码。
"""

import sys
import io
import contextlib
import traceback
import subprocess
import tempfile

try:
    import resource  # 仅 Unix 可用，用于设置子进程资源上限
    RESOURCE_AVAILABLE = True
except ImportError:
    resource = None
    RESOURCE_AVAILABLE = False
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any
import logging
import os
import base64

# 尝试导入科学计算库
try:
    import scipy
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    scipy = None
    SCIPY_AVAILABLE = False

try:
    import sklearn
    from sklearn import datasets, model_selection, metrics
    SKLEARN_AVAILABLE = True
except ImportError:
    sklearn = None
    SKLEARN_AVAILABLE = False

try:
    import statsmodels
    import statsmodels.api as sm
    import statsmodels.stats as sm_stats
    STATSMODELS_AVAILABLE = True
except ImportError:
    statsmodels = None
    sm = None
    sm_stats = None
    STATSMODELS_AVAILABLE = False

# 设置matplotlib后端为非交互式
matplotlib.use('Agg')

# 配置matplotlib中文字体支持
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

logger = logging.getLogger(__name__)


def find_qwen_agent_python():
    """
    查找qwen_agent虚拟环境的Python解释器

    Returns:
        Python解释器路径，如果找不到则返回None
    """
    # 方法1: 检查VIRTUAL_ENV环境变量
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path and 'qwen' in venv_path.lower():
        python_path = os.path.join(venv_path, 'bin', 'python')
        if os.path.exists(python_path):
            logger.info(f"找到qwen_agent虚拟环境: {python_path}")
            return python_path

    # 方法2: 检查当前Python是否在qwen相关的虚拟环境中
    current_python = sys.executable
    if 'qwen' in current_python.lower() or 'qwen' in os.environ.get('VIRTUAL_ENV', '').lower():
        logger.info(f"当前Python在qwen环境中: {current_python}")
        return current_python

    # 方法3: 尝试查找常见的qwen_agent虚拟环境路径
    possible_paths = [
        os.path.expanduser('~/qwen_agent/bin/python'),
        os.path.expanduser('~/venv/qwen_agent/bin/python'),
        '/opt/qwen_agent/bin/python',
    ]

    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"找到qwen_agent虚拟环境: {path}")
            return path

    # 如果都找不到，使用当前Python（假设已经在正确的环境中）
    logger.warning(f"未找到qwen_agent虚拟环境，使用当前Python: {sys.executable}")
    return sys.executable


def _build_resource_limiter(memory_limit_mb: int = None, cpu_time_limit: int = None):
    """
    构建用于子进程 preexec_fn 的资源限制函数（仅 Unix）。

    参数:
        memory_limit_mb: 地址空间上限（MB）；None 表示不限制
        cpu_time_limit: CPU 时间上限（秒）；None 表示不限制

    返回:
        可作为 subprocess preexec_fn 的回调；不可用时返回 None
    """
    if not RESOURCE_AVAILABLE:
        return None

    def _set_limits():
        # 禁止生成 core dump
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass
        if memory_limit_mb:
            try:
                limit_bytes = memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            except (ValueError, OSError):
                pass
        if cpu_time_limit:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
            except (ValueError, OSError):
                pass

    return _set_limits


def execute_python_code(code: str, timeout: int = 30,
                        memory_limit_mb: int = None,
                        cpu_time_limit: int = None) -> str:
    """
    在独立子进程中执行Python代码并返回结果。

    安全说明：这不是安全隔离沙箱，子进程仍可访问文件系统、网络与环境变量，
    请勿用于执行不受信任的代码。可通过资源限制参数降低失控风险。

    参数:
        code: 要执行的Python代码
        timeout: 墙钟超时时间（秒）
        memory_limit_mb: 子进程地址空间上限（MB），None 表示不限制（仅 Unix 生效）
        cpu_time_limit: 子进程 CPU 时间上限（秒），None 表示不限制（仅 Unix 生效）

    返回:
        执行结果的字符串表示，包括输出、错误信息和生成的图片
    """
    try:
        # 查找qwen_agent虚拟环境的Python解释器
        python_executable = find_qwen_agent_python()

        # 创建临时文件保存代码
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            temp_code_file = f.name
            f.write(code)

        result_parts = []

        try:
            # 使用subprocess在虚拟环境中执行代码
            logger.info(f"在虚拟环境中执行代码: {python_executable}")

            # 准备环境变量（保持当前环境，但确保使用虚拟环境）
            env = os.environ.copy()

            # 可选的资源限制（内存 / CPU 时间 / core dump），仅 Unix 生效
            preexec_fn = _build_resource_limiter(memory_limit_mb, cpu_time_limit)

            # 执行代码
            process = subprocess.Popen(
                [python_executable, temp_code_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                env=env,
                cwd=os.getcwd(),
                preexec_fn=preexec_fn
            )

            try:
                stdout_content, stderr_content = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout_content, stderr_content = process.communicate()
                result_parts.append(f"执行超时（超过{timeout}秒）")

            # 获取返回码
            return_code = process.returncode

            # 构建结果
            if stdout_content:
                result_parts.append(f"程序输出:\n{stdout_content}")

            if stderr_content:
                result_parts.append(f"警告/错误信息:\n{stderr_content}")

            if return_code != 0:
                result_parts.append(f"执行失败，返回码: {return_code}")

            # 检查是否生成了图片（在plots目录中查找）
            plots_dir = None
            current_dir = os.getcwd()

            # 查找session_outputs目录
            for item in os.listdir(current_dir):
                if item.startswith('session_outputs_') and os.path.isdir(item):
                    plots_dir = os.path.join(current_dir, item, 'plots')
                    break

            # 如果没找到，创建一个临时plots目录
            if plots_dir is None or not os.path.exists(plots_dir):
                plots_dir = os.path.join(current_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)

            # 检查plots目录中是否有新生成的图片
            if os.path.exists(plots_dir):
                import time
                # 获取当前时间戳，查找最近生成的图片
                recent_files = []
                for filename in os.listdir(plots_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                        filepath = os.path.join(plots_dir, filename)
                        # 检查文件是否在最近1分钟内修改
                        if time.time() - os.path.getmtime(filepath) < 60:
                            recent_files.append(filepath)

                if recent_files:
                    for img_path in recent_files:
                        result_parts.append(f"已生成图片: {img_path}")

            if not result_parts:
                result_parts.append("代码执行成功，无输出")

            return "\n\n".join(result_parts)

        finally:
            # 清理临时文件
            try:
                if 'temp_code_file' in locals() and os.path.exists(temp_code_file):
                    os.unlink(temp_code_file)
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")

    except subprocess.TimeoutExpired:
        error_msg = f"代码执行超时（超过{timeout}秒）"
        logger.error(error_msg)
        return f"执行失败:\n{error_msg}"

    except Exception as e:
        error_msg = f"执行过程异常: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return f"执行失败:\n{error_msg}"


def validate_python_code(code: str) -> str:
    """
    验证Python代码的语法正确性

    参数:
        code: 要验证的Python代码

    返回:
        验证结果信息
    """
    try:
        compile(code, '<string>', 'exec')
        return "代码语法正确"
    except SyntaxError as e:
        return f"语法错误: {str(e)}\n行 {e.lineno}: {e.text}"
    except Exception as e:
        return f"验证时出错: {str(e)}"


def get_code_help(topic: str = None) -> str:
    """
    获取Python代码帮助信息

    参数:
        topic: 要获取帮助的主题

    返回:
        帮助信息
    """
    if not topic:
        return """
可用的Python库和功能:
- pandas (pd): 数据处理和分析
- numpy (np): 数值计算
- matplotlib.pyplot (plt): 绘图
- seaborn (sns): 统计可视化

常用功能:
1. 数据读取: pd.read_csv(), pd.read_excel()
2. 数据分析: df.describe(), df.info(), df.head()
3. 数据可视化: plt.plot(), sns.histplot(), sns.scatterplot()
4. 数学计算: np.mean(), np.std(), np.sum()

示例代码:
```python
# 数据分析示例
import pandas as pd
import matplotlib.pyplot as plt

# 创建示例数据
data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
df = pd.DataFrame(data)

# 基本统计
print(df.describe())

# 绘图
plt.plot(df['x'], df['y'])
plt.title('示例图表')
plt.show()
```
"""

    try:
        # 尝试获取特定主题的帮助
        if topic in ['pd', 'pandas']:
            return "Pandas主要功能:\n- 数据读取: read_csv(), read_excel()\n- 数据操作: groupby(), merge(), pivot()\n- 数据分析: describe(), info(), value_counts()"
        elif topic in ['np', 'numpy']:
            return "NumPy主要功能:\n- 数组操作: array(), reshape(), transpose()\n- 数学函数: mean(), std(), sum(), max(), min()\n- 线性代数: dot(), linalg.inv()"
        elif topic in ['plt', 'matplotlib']:
            return "Matplotlib主要功能:\n- 基本绘图: plot(), scatter(), bar(), hist()\n- 图表设置: title(), xlabel(), ylabel(), legend()\n- 保存图片: savefig()"
        elif topic in ['sns', 'seaborn']:
            return "Seaborn主要功能:\n- 统计图表: histplot(), scatterplot(), boxplot()\n- 关系图: pairplot(), heatmap()\n- 分类图: barplot(), countplot()"
        else:
            return f"未找到关于 '{topic}' 的帮助信息"

    except Exception as e:
        return f"获取帮助信息时出错: {str(e)}"
