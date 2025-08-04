#!/usr/bin/env python3
"""
TDXAgent 自动安装脚本

这个脚本会自动安装 TDXAgent 的所有依赖项，配置环境，
并验证安装是否成功。
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
import platform
import argparse
from typing import List, Optional

# 颜色输出
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message: str, status: str = "INFO"):
    """打印带颜色的状态信息"""
    color_map = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED
    }
    
    color = color_map.get(status, Colors.WHITE)
    print(f"{color}[{status}]{Colors.ENDC} {message}")

def run_command(command: List[str], capture_output: bool = False) -> tuple:
    """运行命令并返回结果"""
    try:
        if capture_output:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return True, result.stdout
        else:
            result = subprocess.run(command, check=True)
            return True, ""
    except subprocess.CalledProcessError as e:
        return False, str(e)

def check_python_version() -> bool:
    """检查 Python 版本"""
    print_status("检查 Python 版本...")
    
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 9):
        print_status(f"需要 Python 3.9 或更高版本，当前版本: {version_info.major}.{version_info.minor}", "ERROR")
        return False
    
    print_status(f"Python 版本: {version_info.major}.{version_info.minor}.{version_info.micro}", "SUCCESS")
    return True

def check_pip() -> bool:
    """检查 pip 是否可用"""
    print_status("检查 pip...")
    
    success, _ = run_command([sys.executable, "-m", "pip", "--version"], capture_output=True)
    if not success:
        print_status("pip 不可用，请先安装 pip", "ERROR")
        return False
    
    print_status("pip 检查通过", "SUCCESS")
    return True

def install_dependencies(use_dev: bool = False) -> bool:
    """安装项目依赖"""
    print_status("安装项目依赖...")
    
    # 升级 pip
    print_status("升级 pip...")
    success, _ = run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    if not success:
        print_status("pip 升级失败", "WARNING")
    
    # 安装依赖
    if use_dev:
        requirements_file = "requirements-dev.txt"
    else:
        requirements_file = "requirements.txt"
    
    if not Path(requirements_file).exists():
        print_status(f"依赖文件 {requirements_file} 不存在", "ERROR")
        return False
    
    print_status(f"从 {requirements_file} 安装依赖...")
    success, error = run_command([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    
    if not success:
        print_status(f"依赖安装失败: {error}", "ERROR")
        return False
    
    print_status("依赖安装成功", "SUCCESS")
    return True

def install_playwright_browsers() -> bool:
    """安装 Playwright 浏览器"""
    print_status("安装 Playwright 浏览器...")
    
    success, error = run_command([sys.executable, "-m", "playwright", "install", "chromium"])
    
    if not success:
        print_status(f"Playwright 浏览器安装失败: {error}", "ERROR")
        return False
    
    print_status("Playwright 浏览器安装成功", "SUCCESS")
    return True

def create_config_file() -> bool:
    """创建配置文件"""
    print_status("创建配置文件...")
    
    default_config = Path("src/config/default_config.yaml")
    config_file = Path("config.yaml")
    
    if not default_config.exists():
        print_status("默认配置文件不存在", "ERROR")
        return False
    
    if config_file.exists():
        print_status("配置文件已存在，跳过创建", "WARNING")
        return True
    
    try:
        shutil.copy(default_config, config_file)
        print_status("配置文件创建成功", "SUCCESS")
        print_status("请编辑 config.yaml 文件配置您的 API 密钥和其他设置", "INFO")
        return True
    except Exception as e:
        print_status(f"配置文件创建失败: {e}", "ERROR")
        return False

def create_directories() -> bool:
    """创建必要的目录结构"""
    print_status("创建数据目录...")
    
    directories = [
        "TDXAgent_Data",
        "TDXAgent_Data/data",
        "TDXAgent_Data/data/twitter",
        "TDXAgent_Data/data/telegram",
        "TDXAgent_Data/data/discord",
        "TDXAgent_Data/reports",
        "TDXAgent_Data/logs",
        "discord_exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_status("数据目录创建成功", "SUCCESS")
    return True

def verify_installation() -> bool:
    """验证安装"""
    print_status("验证安装...")
    
    # 检查能否导入主要模块
    try:
        sys.path.insert(0, str(Path("src")))
        
        from config.config_manager import ConfigManager
        from main import TDXAgent
        from utils.config_validator import ConfigValidator
        
        print_status("模块导入测试通过", "SUCCESS")
        
        # 验证配置文件
        if Path("config.yaml").exists():
            try:
                config_manager = ConfigManager("config.yaml")
                validator = ConfigValidator()
                validator.validate_config(config_manager.raw_config)
                print_status("配置文件验证通过", "SUCCESS")
            except Exception as e:
                print_status(f"配置文件验证失败: {e}", "WARNING")
        
        return True
        
    except Exception as e:
        print_status(f"安装验证失败: {e}", "ERROR")
        return False

def show_next_steps():
    """显示后续步骤"""
    print_status("安装完成！", "SUCCESS")
    print()
    print(f"{Colors.BOLD}后续步骤:{Colors.ENDC}")
    print(f"1. 编辑 {Colors.CYAN}config.yaml{Colors.ENDC} 文件，配置您的 API 密钥")
    print(f"2. 运行 {Colors.CYAN}python run_tdxagent.py{Colors.ENDC} 启动应用")
    print(f"3. 或使用 CLI: {Colors.CYAN}python src/main.py --help{Colors.ENDC}")
    print()
    print(f"{Colors.BOLD}重要提示:{Colors.ENDC}")
    print(f"• Twitter: 需要手动登录保存 cookies")
    print(f"• Telegram: 需要从 https://my.telegram.org 获取 API 密钥")
    print(f"• Discord: 建议使用安全模式（官方数据导出）")
    print(f"• 确保在 config.yaml 中配置正确的 LLM API 密钥")
    print()

def main():
    """主安装函数"""
    parser = argparse.ArgumentParser(description="TDXAgent 自动安装脚本")
    parser.add_argument("--dev", action="store_true", help="安装开发依赖")
    parser.add_argument("--skip-browsers", action="store_true", help="跳过浏览器安装")
    parser.add_argument("--skip-verification", action="store_true", help="跳过安装验证")
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{Colors.CYAN}TDXAgent 自动安装脚本{Colors.ENDC}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print("=" * 50)
    
    # 检查前置条件
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # 安装依赖
    if not install_dependencies(args.dev):
        sys.exit(1)
    
    # 安装浏览器
    if not args.skip_browsers:
        if not install_playwright_browsers():
            print_status("浏览器安装失败，但可以稍后手动安装", "WARNING")
    
    # 创建配置文件
    if not create_config_file():
        sys.exit(1)
    
    # 创建目录结构
    if not create_directories():
        sys.exit(1)
    
    # 验证安装
    if not args.skip_verification:
        if not verify_installation():
            print_status("安装验证失败，但基本功能可能可用", "WARNING")
    
    # 显示后续步骤
    show_next_steps()

if __name__ == "__main__":
    main()