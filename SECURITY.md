# 安全政策 | Security Policy

## 🛡️ 安全承诺

TDXAgent 致力于为用户提供安全可靠的个人信息管理工具。我们采用多层安全措施保护用户隐私和数据安全。

## 🔒 安全特性

### 数据安全
- ✅ **本地存储**: 所有数据完全存储在用户本地，不上传任何第三方服务
- ✅ **加密传输**: 所有 API 调用使用 HTTPS/TLS 加密
- ✅ **权限控制**: Gmail 使用只读权限，Discord 使用官方数据导出
- ✅ **会话管理**: 安全的认证会话管理和自动过期

### 平台安全等级

| 平台 | 安全等级 | 认证方式 | 风险说明 |
|------|----------|----------|----------|
| **Gmail** | 🟢 **高** | OAuth 2.0 官方API | 官方认证，只读权限，完全安全 |
| **Telegram** | 🟢 **高** | 官方 API | 手机验证，官方API，完全合规 |
| **Discord** | 🟢 **高** | 官方数据导出 | 使用官方导出功能，无风险 |
| **Twitter** | 🟡 **中** | 浏览器自动化 | 存在检测风险，建议使用小号 |

### 代码安全
- ✅ **依赖检查**: 定期扫描和更新依赖包安全漏洞
- ✅ **输入验证**: 严格的输入数据验证和清理
- ✅ **错误处理**: 安全的错误处理，不泄露敏感信息
- ✅ **最小权限**: 遵循最小权限原则，仅请求必要权限

## 🚨 支持的版本

我们为以下版本提供安全更新：

| 版本 | 支持状态 |
| ---- | -------- |
| 2.x.x | ✅ 支持 |
| 1.5.x | ✅ 支持 |
| < 1.5 | ❌ 不支持 |

## 🔍 漏洞报告

### 报告流程

如果您发现安全漏洞，请**不要**公开提交 Issue，而是按以下方式私密报告：

1. **GitHub Security Advisory** (推荐)
   - 访问 [Security Advisories](../../security/advisories)
   - 点击 "Report a vulnerability"
   - 填写详细信息

2. **私人 Issue**
   - 创建 Issue 并标记为 `security` 和 `private`
   - 在标题前加上 `[SECURITY]` 前缀
   - 详细描述漏洞信息

### 报告内容

请在报告中包含以下信息：

- **漏洞类型**: 分类和严重程度评估
- **影响范围**: 受影响的版本和组件
- **复现步骤**: 详细的漏洞复现方法
- **概念验证**: 如果可能，提供 PoC 代码
- **影响评估**: 潜在的安全风险和业务影响
- **修复建议**: 您认为的修复方案（可选）

### 漏洞处理时间

- **确认时间**: 48小时内确认收到报告
- **初步评估**: 7天内完成初步风险评估
- **修复时间**: 根据严重程度确定修复时间
  - 🔴 **严重**: 72小时内
  - 🟡 **中等**: 2周内  
  - 🟢 **低级**: 4周内
- **公开披露**: 修复后30天内发布安全公告

## 🏆 漏洞奖励

虽然我们无法提供现金奖励，但会为负责任的安全研究者提供：

- ✨ 在安全公告中公开致谢（如果您希望）
- 🎖️ 项目贡献者认证
- 📧 官方感谢信

## 🛠️ 安全最佳实践

### 用户建议

#### Gmail 使用
```yaml
gmail:
  enabled: true
  credentials_file: "gmail_credentials.json"  # 从 Google Cloud Console 下载
  # 仅使用只读权限，不会修改或删除邮件
```

#### Twitter 使用
```yaml
twitter:
  enabled: true
  # 🔐 安全建议
  anti_detection:
    enabled: true              # 启用反检测
    random_delays: true        # 随机延迟
    human_behavior: true       # 人类行为模拟
  # ⚠️ 建议使用非主要账户
```

#### Telegram 使用
```yaml
telegram:
  enabled: true
  api_id: "your_api_id"       # 官方 API，完全安全
  api_hash: "your_api_hash"
  # ✅ 使用官方 API，无风险
```

#### Discord 使用
```yaml
discord:
  enabled: true
  mode: "safe"                # 推荐安全模式
  export_path: "discord_exports"
  # ✅ 使用官方数据导出，完全安全
```

### 开发者指南

#### API 密钥管理
```bash
# ✅ 正确方式 - 使用环境变量
export OPENAI_API_KEY="sk-xxx"
export GEMINI_API_KEY="xxx"

# ❌ 错误方式 - 硬编码在代码中
# api_key = "sk-xxx"  # 永远不要这样做
```

#### 敏感文件保护
```bash
# 确保以下文件不被提交到版本控制
echo "*.key" >> .gitignore
echo "*.json" >> .gitignore  # 包含 gmail_credentials.json
echo "config.yaml" >> .gitignore
echo ".env" >> .gitignore
```

#### 依赖安全
```bash
# 定期检查依赖漏洞
pip audit

# 更新依赖包
pip install --upgrade -r requirements.txt
```

## 🔐 隐私保护

### 数据处理原则
- **数据最小化**: 仅收集分析所需的最少数据
- **本地处理**: 所有数据处理完全在本地进行
- **用户控制**: 用户完全控制数据的收集、存储和删除
- **透明度**: 所有数据处理过程完全开源透明

### 第三方服务
- **LLM API**: 仅发送去标识化的文本内容进行分析
- **平台 API**: 仅使用官方 API 进行数据收集
- **无追踪**: 不集成任何用户行为追踪服务

### 数据删除
用户可以随时删除所有本地数据：
```bash
# 删除所有数据
rm -rf TDXAgent_Data/

# 删除认证信息
rm -f gmail_token.json
rm -f twitter_cookies.json
```

## 📋 安全检查清单

### 部署前检查
- [ ] 确认所有 API 密钥通过环境变量配置
- [ ] 验证敏感文件不在版本控制中
- [ ] 检查依赖包安全漏洞
- [ ] 确认日志不包含敏感信息
- [ ] 验证文件权限设置正确

### 定期安全维护
- [ ] 每月更新依赖包
- [ ] 每季度安全审计
- [ ] 监控安全公告
- [ ] 更新安全文档

## 📞 联系方式

- 🔒 **安全报告**: 使用 GitHub Security Advisory
- 💬 **安全讨论**: [GitHub Discussions](../../discussions) (标记 security 标签)
- 📧 **紧急联系**: 创建私人 Issue 并标记 `urgent-security`

---

**感谢您帮助保护 TDXAgent 和所有用户的安全！** 🛡️