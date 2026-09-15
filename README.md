# KnowledgeOS — 企业知识库多 Agent 问答系统

基于 **FastAPI、LangChain、FAISS、MySQL 和 Redis** 实现的企业内部知识问答与知识治理系统。普通聊天由 RouterAgent 识别意图和问题复杂度，再路由到闲聊、知识问答或深度推理工作流；知识问答采用固定、可控的 Planner–Orchestrator–Executor 编排。知识巡检不由普通聊天触发，而是作为新知识发布前的质量关卡。

> 当前项目已建立30条人工标注的检索评测集，用于比较FAISS直接检索与候选召回后重排序的效果；路由准确率和答案生成质量尚未建立正式评测集。
## 核心能力

- LLM 意图分类，失败时使用关键词规则降级。
- L1/L2/L3 分级处理简单问答、上下文问题和复杂推理问题。
- 新知识上传、解析、分块、Embedding、自动巡检、管理员审核与受控发布。
- 临时附件与聊天截图 OCR；附件总结直接基于附件，客服案例结合企业知识库生成处理建议。
- Redis 多轮对话记忆，以及超过 10 轮后的增量 LLM 摘要压缩。
- 规则级语义重复、潜在规则冲突、Chunk 质量问题和长期未更新文档巡检。
- SSE 流式输出回答、执行步骤和参考来源。
- 用户注册登录、管理员权限、知识库管理和会话隔离。
- 工具层统一输入校验、超时和失败重试。

## 系统架构

~~~text
用户提问
   │
   ▼
RouterAgent
├─ IntentClassifier：LLM 分类，失败时关键词降级
├─ ChitChatAgent：问候、闲聊、身份类问题
├─ KnowledgeQAAgent：L1/L2 固定工作流知识问答
└─ ReasoningAgent：L3 子问题拆解、分别检索与汇总

KnowledgeQAAgent
   │
   ▼
Orchestrator 创建 AgentState
   │
   ▼
Planner 规划固定步骤
   │
   ▼
Executor 按顺序执行工具和步骤
   │
   ▼
SSE 返回步骤事件、回答 Token 与参考来源

管理员上传新文档
   │
   ▼
解析与分块 → 新 Chunk Embedding → PENDING_REVIEW
   │
   ▼
InspectionAgent：新知识与 ACTIVE FAISS 比较
   │
   ├─ Chunk 质量检测
   ├─ 语义重复规则检测
   └─ 潜在规则冲突检测
   │
   ▼
管理员审核 ──通过──▶ ACTIVE + 写入线上 FAISS
             └驳回──▶ REJECTED（不进入线上 FAISS）
~~~

这里的“多 Agent”指 Router 将不同任务路由到具有不同目标和执行流程的专项 Agent。专项 Agent 之间不会自由对话，知识问答主链路也不是 ReAct。

## Agent 职责

| Agent | 触发场景 | 实际处理逻辑 |
|---|---|---|
| RouterAgent | 所有用户输入 | 意图识别、复杂度判断和工作流路由 |
| ChitChatAgent | 问候、感谢、日常闲聊、系统身份询问 | LLM 生成自然回复并写入会话记忆 |
| KnowledgeQAAgent | 企业制度、产品、运营等知识问题 | Planner 规划步骤，Orchestrator 调度，Executor 执行 RAG 链路 |
| ReasoningAgent | 对比、分析、评估、权衡等复杂问题 | 最多拆解 4 个子问题，分别检索、判断充分性、推理并汇总 |
| InspectionAgent | 新文档上传后的自动巡检，以及管理员对待审核文档重新巡检 | 使用新 Chunk 向量查询 ACTIVE FAISS 候选，再由现有 LLM 做规则级 duplicate/conflict/novel 判断；不属于普通聊天路由 |

## L1/L2/L3 处理链路

| 级别 | 当前判定方式 | 执行链路 |
|---|---|---|
| L1 简单 | 不含复杂分析词，也不依赖上下文 | 知识检索 → 充分性判断 → 流式答案生成 → 写入记忆 |
| L2 上下文 | 含“它、这个、上面、之前、刚才”等指代，或 API 提供了上下文 | 读取记忆 → LLM 问题改写 → 知识检索 → 充分性判断 → 流式答案生成 → 写入记忆 |
| L3 复杂 | 含“对比、优缺点、区别、分析、总结、评估、权衡”等关键词 | 路由到 ReasoningAgent → 拆解子问题 → 各自检索与判断 → 分步推理 → 流式汇总 |


## 临时附件与客服案例辅助

聊天框支持 PDF、DOCX、TXT、Markdown 和常见图片格式，也支持在输入框中使用 Ctrl+V 粘贴截图。图片在上传阶段使用本地 Tesseract OCR，原文件解析后立即删除，永远不写入 FAISS。解析得到的文字和附件元数据会按 `conversation_id` 临时写入 Redis，默认 60 分钟过期，用于同一会话内的后续追问。

~~~text
上传附件
  → 文档解析或 OCR
  → Router 在原有一次 LLM 分类中判断附件用途并生成检索 Query
  ├─ summary：直接根据附件总结，不检索公共知识库
  ├─ case_assist：检索、重排序企业规则，再生成处理步骤与建议回复
  └─ reasoning：结合附件内容进入复杂分析链路
~~~

没有重新上传附件时，Router 可以读取同一会话的缓存附件；只有当前问题确实引用附件时才复用，其他问题仍走普通路由。关键词规则只在 LLM 分类失败时降级使用。最终 Prompt 明确区分“附件事实”“企业知识库规则”和“会话历史”，避免公共知识库内容覆盖本次附件。删除会话时会同步删除附件缓存。

## RAG 流程

### 文档入库

~~~text
管理员上传文档
  → 文档解析
  → 结构感知的文本分块
  → 新 Chunk Embedding
  → Chunk 写入 MySQL，向量写入独立待审核缓存
  → 状态 PENDING_REVIEW
  → 自动巡检：新知识 VS 当前 ACTIVE 知识
  → 管理员审核
  ├─ 通过：状态 ACTIVE，向量写入线上 FAISS
  └─ 驳回：状态 REJECTED，不进入线上 FAISS
~~~

正式 FAISS 只保存 `ACTIVE` 知识。待审核向量位于 `pending_vectors`，不会参与普通 RAG；旧知识已经存在 FAISS 中，巡检不会重新计算整个知识库的 Embedding。

默认分块参数：

- CHUNK_STRATEGY=semantic：按段落、句子和标点进行结构化切分；它不是基于 Embedding 的语义断点模型。
- CHUNK_SIZE=500
- CHUNK_OVERLAP=50
- MIN_CHUNK_SIZE=100

### 在线检索

~~~text
用户问题（L2 使用改写后的 Query）
  → Embedding
  → FAISS 召回候选片段
  → Simple Reranker 重排序
  → 规则式充分性判断
  → 将筛选结果交给 LLM 生成答案
~~~

当前默认配置：

- 知识问答最终保留 5 个片段，先从 FAISS 召回 5 × 3 = 15 个候选。
- 深度推理的每个子问题最终保留 3 个片段，先召回 3 × 3 = 9 个候选。
- RERANKER_TYPE=simple，使用字符集合 Jaccard 相似度和 Query 覆盖度进行规则打分，不调用重排序大模型。
- 代码保留 BGE/Cohere Reranker 的可选实现，但默认主链路没有启用，不能当作当前效果描述。
- 检索充分性使用规则判断：没有片段、Simple Rerank 分数均不大于 0，或片段少于 2 个时判为资料不足。

## 会话记忆

原始对话消息保存在 Redis，并以 conversation_id 隔离。当前压缩策略按对话轮数执行：

1. 不超过 10 轮时，直接读取原始历史。
2. 超过 10 轮时，保留最近 5 轮完整对话。
3. 更早的消息交给 LLM 生成摘要，保留用户问题、AI 回答要点和用户偏好。
4. 已存在摘要时，只总结上次游标之后新增的旧消息，并与旧摘要合并，避免每次全量压缩。
5. 摘要和摘要游标在 Redis 中缓存 1 小时；过期后根据仍然保存的原始消息重新生成。

## 工具与执行控制

当前注册到 ToolRegistry 的工具只有：

- question_rewrite
- knowledge_search
- conversation_memory_read
- conversation_memory_write

工具注册器负责参数校验，并通过线程池执行工具。每个工具可以配置超时时间和最大重试次数；超过重试次数后向上抛出异常。需要注意，线程池的 future.cancel() 不能强制终止已经开始运行的 Python 函数，因此这里实现的是调用方超时返回，不是硬杀执行线程。

知识问答编排状态默认最多 20 个已完成或失败步骤，总运行时间上限为 300 秒。

## 技术栈

| 层 | 当前实现 |
|---|---|
| Web 后端 | FastAPI、Uvicorn |
| LLM 接入 | LangChain ChatOpenAI 对接阿里云 DashScope OpenAI 兼容接口 |
| Embedding | 默认 DashScope text-embedding-v1；未配置密钥或显式选择本地模式时使用 HuggingFace Embedding |
| 向量库 | 默认 FAISS；代码保留可选 Milvus 支持 |
| 重排序 | 默认 Simple Reranker；BGE/Cohere 为可选实现 |
| 关系数据库 | MySQL：用户、文档、Chunk、运行统计、用户画像 |
| 会话记忆 | Redis：原始消息、增量摘要、摘要游标 |
| 文档处理 | PyPDF、python-docx、docx2txt；解析器保留图片 OCR 支持 |
| 前端 | 原生 HTML、CSS、JavaScript，Lucide Icons |
| 流式通信 | SSE（Server-Sent Events） |
| 测试 | Pytest 单元测试：路由、Planner、状态、记忆、分块和工具注册器 |

## 快速开始

### 环境要求

- Python 3.10+
- MySQL 8.0+
- Redis 7.0+
- 阿里云 DashScope API Key（如改用本地 Embedding，LLM 功能仍需模型配置）
- Tesseract OCR（仅在需要识别聊天截图时安装）

### 1. 安装依赖

~~~powershell
cd python-service
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

如需运行自动化测试，再安装开发依赖：

~~~powershell
pip install -r requirements-dev.txt
~~~

### 2. 配置环境变量

~~~powershell
Copy-Item .env.example .env
~~~

复制示例后，至少填写 DashScope、MySQL 和初始管理员配置。真实 `.env` 已被 `.gitignore` 排除，不会提交到 GitHub：

~~~ini
DASHSCOPE_API_KEY=your-dashscope-key

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=ai_knowledge_db
MYSQL_USERNAME=root
MYSQL_PASSWORD=your-mysql-password

REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

DEFAULT_ADMIN_USERNAME=admin
DEFAULT_ADMIN_PASSWORD=your-private-admin-password

USE_MILVUS=false
RERANKER_TYPE=simple
KNOWLEDGE_TOP_K=5
REASONING_TOP_K=3
RETRIEVAL_CANDIDATE_MULTIPLIER=3
~~~

### 3. 启动服务

~~~powershell
cd python-service
python main.py
~~~

浏览器访问 http://127.0.0.1:8000。

首次启动时，如果 `DEFAULT_ADMIN_PASSWORD` 非空且管理员账号尚不存在，系统会按环境变量创建初始管理员。源码和 README 不提供通用默认密码；已经存在的管理员账号不会被覆盖。

### 4. 运行单元测试

~~~powershell
cd python-service
pytest -q
~~~

### 5. 运行检索结果评测

项目提供30条人工标注问题，只评价检索结果，不调用最终答案生成模型：

~~~powershell
cd python-service
python -m evaluation.run_retrieval_eval
~~~

评测会对比FAISS直接Top 5与“候选召回 + 当前Reranker Top 5”，输出Precision@5、Recall@5、HitRate@5、MRR@5、NDCG@5及耗时。具体说明见 python-service/evaluation/README.md。

当前保留的一次30题评测中，Simple Reranker 将 MRR@5 从 0.802 提升到 0.922、NDCG@5 从 0.796 提升到 0.870，平均耗时增加约 1.5 ms。完整口径见 [检索评测基线](python-service/evaluation/benchmark.md)。

## 主要 API

除注册和登录外，业务接口需要 Bearer Token；文档上传、文档删除、知识巡检、Agent 统计和自动报表要求管理员权限。知识巡检使用登录令牌中的 `is_admin` 判断，不能由前端参数伪造。

| 端点 | 方法 | 说明 |
|---|---|---|
| /api/auth/register | POST | 用户注册 |
| /api/auth/login | POST | 登录并获取 Bearer Token |
| /api/agent/run/stream | POST | Agent SSE 流式问答 |
| /api/agent/stats | GET | Agent 运行统计（管理员） |
| /api/agent/report | GET | 自动报表数据（管理员） |
| /api/documents/upload | POST | 上传文档并自动进入待审核巡检（管理员） |
| /api/documents | GET | 分页查询知识库文档 |
| /api/documents/{doc_id}/inspection | GET | 查看持久化巡检报告（管理员） |
| /api/documents/{doc_id}/inspect | POST | 重新巡检待审核文档（管理员） |
| /api/documents/{doc_id}/approve | POST | 审核通过并发布到正式 FAISS（管理员） |
| /api/documents/{doc_id}/reject | POST | 驳回文档，不进入正式 FAISS（管理员） |
| /api/documents/{doc_id} | DELETE | 删除文档、Chunk、向量与上传文件（管理员） |
| /api/documents/parse-temp | POST | 临时解析当前对话附件，不写入知识库 |
| /api/conversations | GET | 查询当前用户的会话列表 |
| /api/conversations/{conversation_id} | GET | 查询当前用户的会话消息 |
| /api/conversations/{conversation_id} | DELETE | 删除当前用户的会话和摘要 |
| /health | GET | 服务与向量存储配置检查 |

## 项目结构

~~~text
my agent/
├─ python-service/
│  ├─ agent/
│  │  ├─ state.py              # AgentState、步骤状态与终止条件
│  │  ├─ planner.py            # 固定步骤规划、问题改写与充分性判断
│  │  ├─ orchestrator.py       # 状态创建、循环调度和流式事件
│  │  ├─ executor.py           # 执行检索、评估、记忆等具体步骤
│  │  └─ memory_agent.py       # 会话记忆和用户偏好封装
│  ├─ workflows/
│  │  ├─ router_agent.py       # 意图与复杂度路由
│  │  ├─ chitchat_agent.py     # 闲聊工作流
│  │  ├─ knowledge_qa_agent.py # L1/L2 知识问答工作流
│  │  ├─ reasoning_agent.py    # L3 深度推理工作流
│  │  └─ inspection_agent.py   # 知识库质量巡检
│  ├─ tools/                   # 工具定义、注册、超时与重试
│  ├─ intent/                  # LLM 意图分类与关键词降级
│  ├─ core/                    # LLM、Embedding、FAISS、MySQL、Redis、分块与重排序
│  ├─ api/                     # 登录、Agent、文档和会话接口
│  ├─ tests/                   # 单元测试
│  └─ main.py                  # FastAPI 入口
├─ static/
│  ├─ index.html               # 页面结构
│  ├─ styles.css               # KnowledgeOS 统一视觉样式
│  ├─ app.js                   # 页面交互、权限与 SSE 渲染
│  └─ favicon.svg              # 产品图标
└─ README.md
~~~

## 当前边界

- 已建立30条人工标注的检索评测集；路由准确率和答案生成质量尚未建立正式评测集，不能仅凭演示问题证明这两个阶段的效果。
- 当前默认 Simple Reranker 速度快，但语义相关性能力弱于 Cross-Encoder/BGE Reranker。
- FAISS 更适合当前本地演示规模；文档频繁删除或知识库显著增长时，应考虑定期重建索引或切换支持过滤和删除的向量数据库。
- FastAPI 外层为异步接口，但 Agent、向量检索和部分模型调用仍是同步执行，高并发能力尚未经过压测。
- 登录会话 Token 保存在进程内存中，服务重启后失效；密码使用加盐 SHA-256，适用于作品集演示，不应直接作为生产级认证方案。
