# TDXAgent

一个由AI驱动的多平台信息情报系统，自动从Twitter、Telegram、Discord和Gmail收集数据，利用包括自主外部信息检索在内的高级LLM能力生成全面的分析报告。

## 概述

TDXAgent通过先进的AI编排和自主研究能力，将各个平台间大量的信息碎片化转化为可行的情报信息与洞察总结报告，让使用者可以在一份报告能查看所有平台的关键信息与总结，节省大量时间的同时不错过任何核心信息，包括数据采集、分析在内的所有流程自动化、智能化执行。

**核心能力：**
- **智能体AI分析**：由具备自主外部信息检索能力的Gemini CLI驱动
- **多平台数据收集**：整合Twitter/X、Telegram、Discord、Gmail
- **自主研究能力**：AI智能体独立搜索和验证来自外部来源的信息
- **智能上下文整合**：跨平台信息综合与事实核查
- **统一分析管道**：跨平台统一分析，具备质量检测的高级批处理
- **隐私优先**：完全本地处理，无云端数据依赖

## 架构

系统使用Python 3.9+构建，采用async/await模式：

```
TDXAgent/
├── src/
│   ├── scrapers/          # 平台数据收集模块
│   ├── llm/              # LLM提供商抽象层  
│   ├── processors/       # 数据分析管道
│   ├── storage/          # 本地数据持久化
│   └── config/           # 配置管理
├── prompts/              # 分析模板
├── config.yaml           # 主配置文件
└── TDXAgent_Data/        # 本地数据目录
```

## 快速开始

### 系统要求

- Python 3.9+
- Node.js 18+（智能体AI能力必需）
- 互联网连接（AI智能体外部研究用）

### 安装

```bash
git clone https://github.com/your-repo/TDXAgent.git
cd TDXAgent
pip install -r requirements.txt
playwright install chromium  # 用于Twitter数据收集
```

### 配置

1. **复制默认配置：**
```bash
cp src/config/default_config.yaml config.yaml
```

2. **配置智能体AI提供商（带自主能力的Gemini CLI）：**
```bash
# 安装带智能体扩展的Gemini CLI
npm install -g @google/generative-ai-cli
gemini auth login

# 验证智能体能力已启用
gemini --help | grep -E "(search|web|browse)"
```

3. **在`config.yaml`中设置平台凭据：**
```yaml
llm:
  provider: "gemini_cli"
  
platforms:
  telegram:
    api_id: "your_api_id"        # 从 https://my.telegram.org 获取
    api_hash: "your_api_hash"
    
  gmail:
    enabled: false               # 可选，需要OAuth设置
```

### 使用方法

```bash
# 交互式模式
python run_tdxagent.py

# 命令行模式
python src/main.py run --hours 6           # 收集并分析最近6小时数据
python src/main.py status                  # 检查系统状态
python src/main.py collect --hours 12      # 仅收集数据
python src/main.py analyze --hours 6       # 仅分析数据
```

## 平台支持

| 平台 | 安全性 | 数据类型 | 备注 |
|------|--------|----------|------|
| Telegram | ✅ 安全 | 群组聊天 | 官方API |
| Gmail | ✅ 安全 | 邮件内容 | OAuth 2.0只读权限 |
| Discord | ✅ 安全 | 聊天记录 | 官方数据导出 |  
| Twitter/X | ⚠️ 中等 | 推文内容 | 浏览器自动化 |

## 数据采集机制

TDXAgent采用**多平台自适应采集架构**，通过平台特定的优化策略实现高效、安全的数据收集，随后进行统一多平台分析。

### 核心采集优势

- **高性能收集**：平均收集效率提升3-4倍，支持大规模数据采集
- **安全第一**：多种安全模式适配不同风险场景
- **智能优化**：自适应滚动、批量处理、去重机制
- **精准过滤**：基于时间窗口、关键词、群组的智能过滤
- **存储优化**：减少61.2%存储空间，提升查询效率

### Twitter/X 采集机制

**智能浏览器自动化**：
- **转发时间优先处理**：智能识别转发推文，使用转发时间进行时间过滤
- **长推文自动展开**：31+个CSS选择器适应Twitter DOM结构变化
- **渐进式滚动策略**：三种模式(快速/平衡/安全)平衡效率与检测风险
- **防检测机制**：模拟人类行为模式，降低账户风险

**性能配置选项**：
```yaml
twitter:
  scroll_settings:
    mode: "fast"                    # 快速模式：3-4倍效率提升
    distance_range: [3000, 4000]   # 每次滚动获取12-15条推文
    content_wait_range: [1, 2]     # 智能等待时间
  collection_strategy: "time_based" # 时间优先/数量策略
```

### Telegram 采集机制

**官方API安全采集**：
- **群组智能过滤**：支持关键词、精确匹配、全量采集模式
- **认证流程优化**：一次认证，长期使用，支持双重验证
- **消息格式标准化**：统一处理文本、媒体、转发消息
- **群组上下文保持**：完整记录群组信息和消息关联

**配置示例**：
```yaml
telegram:
  group_filter:
    mode: "keywords"                        # 关键词过滤模式
    group_keywords: ["加密", "defi", "ai"]  # 投资相关群组
  max_messages_per_group: 200               # 单群组消息限制
```

**安全特性**：
- ✅ 官方API：100%安全，无封号风险
- ✅ 只读权限：仅获取消息，不发送或修改
- ✅ 批量高效：单次可获取数千条消息

### Gmail 采集机制

**OAuth 2.0安全集成**：
- **智能邮件过滤**：自动过滤投资、技术、商业相关邮件
- **主题内容提取**：智能解析邮件主题和关键内容
- **线程关联处理**：保持邮件对话的完整性
- **权限最小化**：仅请求必要的只读权限

**认证流程**：
```bash
# 首次设置Gmail凭据
1. 下载credentials.json (Google Cloud Console)
2. python src/main.py collect --platforms gmail
3. 浏览器自动打开 → Google账户授权
4. 自动生成gmail_token.json → 后续无需重新认证
```

**技术优势**：
- 🔒 OAuth 2.0：银行级安全认证
- 📧 智能过滤：仅采集相关邮件内容
- 🔄 增量同步：避免重复下载

### Discord 采集机制

**数据导出模式**：
- **官方导出支持**：使用Discord官方数据导出功能
- **服务器频道分析**：按服务器和频道组织数据
- **消息关联处理**：保持对话线程的完整性
- **安全模式优先**：避免Token风险

### 通用采集特性

**存储优化系统**：
- **智能字段过滤**：移除冗余数据，保留核心信息
- **日期组织存储**：按消息发布日期存储，提升查询效率
- **去重机制**：自动识别和处理重复消息
- **增量采集**：支持基于时间窗口的增量数据收集

**数据质量保证**：
- 统一消息格式验证
- 时间戳准确性检查
- 媒体文件完整性验证
- 数据完整性恢复机制

## 智能体AI分析

TDXAgent利用由Gemini CLI驱动的**自主AI智能体**，能够独立研究、验证和交叉引用来自外部来源的信息，提供全面分析。

### 自主能力

- **外部研究**：AI智能体自动搜索网络获取额外上下文和验证信息
- **事实验证**：将收集的信息与权威来源进行交叉引用
- **实时更新**：获取相关讨论主题的最新发展
- **上下文整合**：将多个来源的信息综合成连贯的洞察
- **自适应分析**：AI根据信息复杂性和重要性调整分析深度

### 智能信息综合

- 搜索外部数据库和新闻源
- 验证社交媒体帖子中的声明和统计数据
- 提供来自权威来源的额外上下文
- 标记潜在的错误信息或过时信息
- 为分析的信息生成可信度分数

## 自定义配置

### 分析模板

通过编辑`prompts/pure_investment_analysis.yaml`修改分析行为：

```yaml
template: |
  你是一位顶级的投资研究总监，拥有强大的自主研究能力，能够独立搜索外部信息源
  进行事实验证和上下文补充，确保分析的准确性和完整性。
  
  <agentic_capabilities>
  - 🔍 **自主外部研究**：主动搜索相关信息进行验证和补充
  - 📊 **事实核查能力**：交叉验证信息来源和数据准确性
  - 🌐 **实时信息获取**：获取最新发展和市场动态
  - 🧠 **智能推理整合**：综合多源信息形成深度洞察
  </agentic_capabilities>
  
  <core_expertise>
  - 🪙 **加密货币专家**：项目分析、价格趋势、技术更新
  - 💰 **金融市场分析师**：股票动态、经济数据、政策影响  
  - 🤖 **AI科技专家**：技术突破、产品发布、行业趋势
  - 🔍 **开源项目专家**：技术创新、社区动态、发展前景
  </core_expertise>
```

### 配置选项

`config.yaml`中的关键配置参数：

```yaml
llm:
  provider: "gemini_cli"           # 推荐用于智能体能力
  batch_size: 5000                 # 每批次消息数
  buffer_delays:
    multi_platform_delay: 4        # 提升AI响应质量
    multi_batch_delay: 3           # 减少无效响应
  
agentic_features:
  enable_external_research: true   # 允许AI搜索外部来源
  fact_verification: true          # 与权威来源交叉引用
  confidence_scoring: true         # 生成可靠性分数
  max_research_depth: 3            # 递归研究层级

platforms:
  twitter:
    scroll_settings:
      mode: "fast"                 # 数据收集模式：fast/balanced/safe
    max_tweets_per_run: 100
    
  telegram:
    group_filter:
      mode: "keywords"             # keywords/exact/all
      group_keywords: ["加密", "defi", "ai", "技术"]

performance:
  enable_storage_optimization: true    # 61%存储减少，96%token节省
  enable_progressive_integration: true # 统一跨平台报告
  enable_quality_detection: true      # AI响应验证
```

## 输出格式

分析结果保存为Markdown报告在`TDXAgent_Data/reports/`目录：

```
TDXAgent_Data/reports/
├── TDXAgent_Telegram_投资分析报告_20250804_1430.md
├── TDXAgent_Twitter_投资分析报告_20250804_1430.md
└── TDXAgent_多平台汇总报告_20250804_1430.md
```

每份报告包含：
- **🔥 热门话题**：经过自主研究和验证的热门讨论主题
- **💡 重要信息汇总**：AI验证的关键信息及可信度分数
- **📊 分领域整理**：专业AI智能体的领域特定分析
- **⚠️ 需要关注的信息**：AI标记的需要跟进的项目及外部上下文
- **🎯 行动建议**：基于综合分析的AI生成建议
- **📝 附录：AI对话详情**：完整的AI推理过程，确保透明度

## 开发

### 架构组件

- **数据收集器**：具备速率限制和错误处理的平台特定数据收集，支持统一分析
- **存储系统**：基于日期组织的JSONL存储系统
- **LLM集成**：支持多提供商的统一接口（OpenAI、Gemini、Claude CLI）
- **处理器**：统一多平台分析，针对大数据集的批处理和渐进式分析

### 数据流程

1. **收集**：平台爬虫在指定时间范围内收集消息
2. **存储**：消息存储在按日期组织的JSONL文件中（`YYYY-MM-DD.jsonl`）
3. **统一分析**：LLM以平台标签化格式统一分析所有平台数据
4. **渐进式处理**：大数据集通过批次处理，使用previous_analysis传递结果
5. **输出**：带有来源归属的统一分析Markdown报告

### 性能优化

- **存储优化**：通过智能字段过滤减少61%空间占用
- **Token使用**：通过优化数据格式减少96%Token消耗
- **处理速度**：通过批量优化提升3-4倍处理速度
- **内存优化**：针对大数据集的流式JSONL处理
- **质量保证**：基于模板的AI响应质量检测
- **缓冲机制**：多平台和多批次处理缓冲，提升成功率

## 故障排除

**配置问题：**
```bash
python src/main.py status  # 检查系统状态
```

**平台认证：**
- Telegram：首次运行需要手机验证
- Gmail：凭据设置需要OAuth浏览器流程
- Twitter：由于自动化检测，建议使用次要账户

**常见问题：**
- Gemini CLI未找到：`npm install -g @google/generative-ai-cli`
- 权限被拒绝：检查`TDXAgent_Data/`目录写入权限
- 速率限制：调整平台配置中的收集频率
- AI响应质量差：系统会自动检测并记录无效响应
- 缓冲时间调整：根据网络情况调整`buffer_delays`配置

## 许可证

MIT许可证 - 详见[LICENSE](LICENSE)文件。

---

**注意**：此工具设计用于个人信息管理和研究。用户有责任遵守平台服务条款和适用法规。智能体AI功能需要互联网连接进行外部研究，请确保网络连接稳定。