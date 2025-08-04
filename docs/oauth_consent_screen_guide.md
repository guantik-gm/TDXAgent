# 找到 Google Cloud OAuth 同意屏幕的完整指南

## 问题：找不到 OAuth 同意屏幕

在 Google Cloud Console 中，OAuth 同意屏幕选项可能由于以下原因不显示：
1. 项目类型不同
2. 权限不足
3. 界面布局变化
4. 需要先启用相关 API

## 解决方案

### 方法 1: 直接 URL 访问（推荐）

1. **打开浏览器访问以下 URL**：
   ```
   https://console.cloud.google.com/apis/credentials/consent
   ```

2. **选择正确的项目**
   - 确保页面顶部显示的是你的 TDXAgent 项目
   - 如果不是，点击项目选择器切换到正确项目

3. **如果提示选择用户类型**：
   - 选择 "外部" (External)
   - 点击 "创建"

### 方法 2: 通过 API 库触发

1. **先启用 Gmail API**：
   ```
   https://console.cloud.google.com/apis/library/gmail.googleapis.com
   ```
   点击 "启用"

2. **然后访问凭据页面**：
   ```
   https://console.cloud.google.com/apis/credentials
   ```

3. **创建凭据时会自动提示配置 OAuth**：
   - 点击 "+ 创建凭据"
   - 选择 "OAuth 客户端 ID"
   - 系统会提示先配置 OAuth 同意屏幕

### 方法 3: 搜索功能

1. **在 Google Cloud Console 顶部搜索框中输入**：
   ```
   OAuth consent screen
   ```

2. **从搜索结果中选择正确的选项**

### 方法 4: 完整导航路径

有些界面版本的完整路径是：
```
左侧菜单 → APIs & Services → Credentials → OAuth consent screen 标签页
```

或者：
```
左侧菜单 → APIs & Services → OAuth consent screen
```

## 如果仍然找不到

### 检查项目设置

1. **确认项目类型**：
   - 访问：https://console.cloud.google.com/iam-admin/settings
   - 检查项目是否为个人项目

2. **确认权限**：
   - 确保你是项目的所有者或编辑者
   - 如果是组织项目，可能需要管理员权限

### 替代方案：创建新项目

如果上述方法都不行，创建一个新的个人项目：

1. **创建新项目**：
   ```
   https://console.cloud.google.com/projectcreate
   ```

2. **项目名称**：TDXAgent-Gmail-Personal

3. **确保选择个人账户**（不是组织账户）

## 配置 OAuth 同意屏幕

找到 OAuth 同意屏幕后，按以下步骤配置：

### 第一步：基本信息
```
用户类型：外部 (External)
应用名称：TDXAgent
用户支持电子邮件：your-email@gmail.com
开发者联系信息：your-email@gmail.com
```

### 第二步：作用域（可选）
- 保持默认，不需要添加特殊作用域
- Gmail API 会自动请求必要的权限

### 第三步：测试用户
```
点击 "+ 添加用户"
输入你的 Gmail 邮箱地址
保存
```

### 第四步：摘要
- 检查所有信息
- 点击 "保存并继续"

## 验证配置

配置完成后：

1. **检查状态**：
   - OAuth 同意屏幕状态应显示为 "测试中"
   - 测试用户列表应包含你的邮箱

2. **创建 OAuth 客户端 ID**：
   ```
   返回：APIs & Services → Credentials
   点击：+ 创建凭据 → OAuth 客户端 ID
   应用类型：桌面应用程序
   名称：TDXAgent Desktop
   ```

3. **下载凭据文件**：
   - 下载 JSON 文件
   - 重命名为 gmail_credentials.json
   - 放置在项目根目录

## 常见问题

### Q: 为什么找不到 OAuth 同意屏幕？
A: 可能需要先启用 Gmail API，或者使用直接 URL 访问。

### Q: 提示需要验证应用？
A: 选择 "测试模式" 并添加你的邮箱为测试用户。

### Q: 创建凭据时提示错误？
A: 确保已正确配置 OAuth 同意屏幕。

## 快速解决链接

- **直接访问 OAuth 同意屏幕**：https://console.cloud.google.com/apis/credentials/consent
- **Gmail API 启用**：https://console.cloud.google.com/apis/library/gmail.googleapis.com
- **凭据管理**：https://console.cloud.google.com/apis/credentials