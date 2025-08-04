# 贡献指南 | Contributing Guide

感谢您对 TDXAgent 项目的关注！我们欢迎所有形式的贡献。

## 🚀 快速开始

### 开发环境设置

```bash
# 1. Fork并克隆项目
git clone https://github.com/yourusername/TDXAgent.git
cd TDXAgent

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. 安装开发依赖
pip install -r requirements-dev.txt
pip install -e .

# 4. 安装Playwright浏览器
playwright install chromium

# 5. 运行测试确保环境正常
pytest tests/
```

## 📋 贡献类型

### 🐛 Bug 报告
- 使用 [Bug Report 模板](.github/ISSUE_TEMPLATE/bug_report.md)
- 提供详细的复现步骤
- 包含系统信息和错误日志

### ✨ 功能建议
- 使用 [Feature Request 模板](.github/ISSUE_TEMPLATE/feature_request.md)
- 说明功能的使用场景和价值
- 提供设计思路和实现建议

### 📚 文档改进
- 修正文档错误或不完整内容
- 添加使用示例和最佳实践
- 翻译文档到其他语言

### 💻 代码贡献
- Bug 修复
- 新功能开发
- 性能优化
- 代码重构

## 🔄 开发流程

### 1. 创建分支
```bash
# 从main分支创建功能分支
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# 分支命名规范
feature/add-new-platform    # 新功能
bugfix/fix-twitter-login     # Bug修复
docs/update-installation     # 文档更新
refactor/optimize-storage    # 代码重构
```

### 2. 开发规范

#### 代码风格
```bash
# 格式化代码
black src/ tests/

# 检查代码质量
flake8 src/ tests/

# 类型检查
mypy src/
```

#### 提交规范
使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```bash
feat: 添加Gmail支持
fix: 修复Twitter登录问题  
docs: 更新安装文档
refactor: 重构存储模块
test: 添加集成测试
perf: 优化数据处理性能
```

#### 测试要求
```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_storage.py -v

# 检查测试覆盖率
pytest --cov=src tests/
```

### 3. 提交 Pull Request

#### PR 检查清单
- [ ] 代码通过所有测试
- [ ] 遵循代码风格规范  
- [ ] 添加必要的测试用例
- [ ] 更新相关文档
- [ ] CHANGELOG.md 已更新
- [ ] 提交信息符合规范

#### PR 模板
请使用 [PR 模板](.github/pull_request_template.md) 提供完整信息：

- 变更内容说明
- 测试情况
- 相关 Issue 链接
- 破坏性变更说明

## 🏗️ 项目架构

### 核心模块
```
src/
├── config/          # 配置管理
├── scrapers/        # 平台数据收集
├── storage/         # 数据存储
├── llm/            # LLM提供商
├── processors/     # 数据处理
└── utils/          # 工具函数
```

### 开发原则
- **模块化设计**: 每个平台独立实现
- **异步编程**: 使用 async/await 模式
- **错误处理**: 完善的异常处理和重试机制
- **可配置性**: 通过配置文件控制行为
- **可测试性**: 编写单元测试和集成测试

## 🧪 测试指南

### 测试结构
```
tests/
├── test_config/     # 配置模块测试
├── test_scrapers/   # 爬虫模块测试
├── test_storage/    # 存储模块测试
├── test_llm/       # LLM模块测试
└── integration/    # 集成测试
```

### 编写测试
```python
import pytest
from unittest.mock import Mock, patch
from src.scrapers.twitter_scraper import TwitterScraper

class TestTwitterScraper:
    @pytest.fixture
    def scraper(self):
        config = Mock()
        return TwitterScraper(config)
    
    @patch('src.scrapers.twitter_scraper.Playwright')
    async def test_authenticate(self, mock_playwright, scraper):
        # 测试认证功能
        result = await scraper.authenticate()
        assert result.success is True
```

## 📖 文档贡献

### 文档结构
```
docs/
├── getting-started/    # 用户入门
├── user-guide/        # 用户指南  
├── technical/         # 技术文档
├── development/       # 开发文档
└── examples/          # 使用示例
```

### 文档规范
- 使用清晰的标题层次
- 提供代码示例和配置示例
- 添加截图和图表说明
- 保持内容更新和准确

## 🎯 特殊贡献领域

### 平台支持扩展
欢迎添加新的社交媒体平台支持：
1. 继承 `BaseScraper` 类
2. 实现平台特定的认证和数据收集
3. 添加配置选项和文档
4. 编写测试用例

### LLM 提供商支持  
欢迎添加更多 LLM 提供商：
1. 继承 `BaseLLMProvider` 类
2. 实现提供商的 API 调用
3. 添加配置选项
4. 编写测试用例

### 国际化支持
我们欢迎多语言支持：
- 翻译现有文档
- 添加界面多语言支持
- 本地化配置和错误信息

## 🏆 贡献者认可

### 贡献者列表
所有贡献者将被添加到 [CONTRIBUTORS.md](CONTRIBUTORS.md) 文件中。

### 贡献类型标识
我们使用 [All Contributors](https://allcontributors.org/) 规范识别不同类型的贡献：
- 💻 代码
- 📖 文档  
- 🐛 Bug报告
- 💡 想法
- 🔧 工具
- 🌍 翻译

## 📞 联系我们

- 💬 **问题讨论**: [GitHub Discussions](../../discussions)
- 🐛 **Bug 报告**: [GitHub Issues](../../issues)
- 📧 **私人联系**: [创建 Issue 并标记为 private]

## 📄 许可证

通过贡献代码，您同意您的贡献将在 [MIT License](LICENSE) 下发布。

---

感谢您的贡献！🎉