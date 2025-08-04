# Changelog

All notable changes to TDXAgent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 开源项目标准文档架构
- 核心开源项目文件（LICENSE, CONTRIBUTING, etc.）

## [2.2.0] - 2025-08-02

### Added
- 🆕 **统一AI投资分析模板系统**: 极简架构，一个模板智能适配所有平台和场景
- 📊 **投资导向分析**: 专注加密货币、金融市场、AI科技的投资机会识别
- 🎯 **智能场景切换**: 自动判断常规分析 vs 整合分析
- 🏗️ **五层投资分析框架**: 分类→筛选→深析→验证→机会识别

### Changed
- 📝 **模板架构精简**: 从10个模板精简为1个统一主模板，减少90%配置复杂度
- 🤖 **智能适配机制**: 自动识别平台类型，智能切换引用格式和分析重点
- ⚡ **性能优化**: 大幅提升模板加载和处理效率

### Fixed
- 🔧 修复模板选择逻辑，确保统一模板在所有场景下正常工作

## [2.1.0] - 2025-08-01

### Added
- 📱 **Telegram单群组消息限制**: 可配置每群组最大消息数，提升收集性能
- ⚙️ **颗粒化Telegram配置**: 支持per-group和global消息限制，向后兼容
- 🎯 **智能限制调整**: 防止超出全局配额的动态调整机制

### Changed
- 🔧 **配置系统增强**: TelegramConfig新增max_messages_per_group等字段
- 📊 **性能监控**: 增强日志记录，实时监控收集性能

## [2.0.0] - 2025-07-31

### Added
- 🧠 **渐进式报告整合系统**: 彻底解决分批次处理导致的报告割裂问题
- 🔗 **内联链接引用系统**: 消息内容与引用链接直接整合，提升AI分析效率
- 📝 **AI对话收集系统**: 统一收集所有LLM对话，支持模型测试和性能优化
- ⚡ **智能存储优化**: 减少61.2%存储空间和96.2%Token消耗
- 📧 **Gmail集成**: 完整的Gmail OAuth 2.0支持，安全的邮件数据收集

### Changed
- 🎯 **提示词管理重构**: PromptManager系统管理，移除config.yaml中的prompts配置
- 🔄 **批次处理优化**: IntegrationManager智能上下文压缩和传递
- 📊 **报告格式统一**: 统一markdown报告生成，集成引用链接

### Fixed
- 🐛 修复Twitter转发时间处理逻辑
- 🔧 优化长推文展开机制，支持31+CSS选择器
- ⏰ 修复时区转换显示问题

## [1.5.0] - 2025-07-29

### Added
- 🎯 **Telegram群组智能过滤**: 支持关键字模糊匹配，处理特殊符号群组名
- ⚡ **Twitter滚动性能优化**: 三种模式配置，快速模式效率提升300%
- 🔄 **智能转发时间处理**: 优先使用转发时间进行过滤判断

### Changed
- 📱 **Telegram配置增强**: 新增group_filter配置，支持keywords/exact/all模式
- 🐦 **Twitter收集策略优化**: Following页面时间优先，For You页面数量优先

### Fixed
- 🕒 修复转发推文时间过滤逻辑
- 🔍 改进长推文检测和展开机制

## [1.0.0] - 2025-07-15

### Added
- 🚀 **首个稳定版本发布**
- 🐦 **Twitter数据收集**: 支持Following和For You页面
- 📱 **Telegram集成**: 官方API安全收集群组消息
- 💬 **Discord支持**: 官方数据导出功能
- 🤖 **AI智能分析**: OpenAI和Gemini双引擎支持
- 📊 **智能报告生成**: Markdown格式分析报告
- 🔒 **本地数据存储**: 完全本地化，保护隐私安全

### Security
- ✅ 所有数据本地存储，不上传第三方服务
- 🔐 Gmail使用OAuth 2.0官方认证，只读权限
- 🛡️ Discord使用官方数据导出，无风险
- 📱 Telegram使用官方API，完全合规

---

## 版本命名规则

- **Major (x.0.0)**: 架构重大变更、不兼容更新
- **Minor (x.y.0)**: 新功能、改进、兼容性更新  
- **Patch (x.y.z)**: Bug修复、安全更新、小幅优化

## 发布流程

1. 更新 CHANGELOG.md
2. 更新版本号 (pyproject.toml)
3. 创建 Git tag
4. 发布 GitHub Release
5. 更新文档链接