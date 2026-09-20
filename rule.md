# 科研 RAG 项目执行规则

这份规则用于约束后续负责实现项目的模型。目标不是把每一步都限制死，而是避免偏离需求、破坏现有文件、跳过验证或虚报完成。

## 1. 开始工作前

开始任何修改前，先阅读：

1. `rule.md`：当前执行规则。
2. `plan/plan1_step.md`：详细实施步骤，主要执行依据。
3. `plan/plan1.md`：项目整体设计和背景。

然后运行：

```bash
pwd
git status --short --branch
git log --oneline -5
```

确认自己位于项目根目录，并了解当前有哪些未提交修改。已有修改可能属于用户，不要随意覆盖、删除或回滚。

## 2. 工作范围

只完成当前极简科研 RAG，不主动扩展为复杂系统。

必须完成的主流程是：

```text
papers/ PDF
  -> 提取文本
  -> 500 字符切块，相邻块重叠 50 字符
  -> 轻量 Embedding
  -> ChromaDB 持久化
  -> 问题 Embedding
  -> Top-3 检索
  -> Ollama deepseek-r1:7b
  -> 回答 + 引用论文文件名
```

首版不实现以下内容，除非用户后来明确要求：

- Web 页面或 API 服务
- OCR
- 多轮聊天记忆
- 用户系统
- BM25、混合检索或 reranker
- 多模型自动切换
- 云端模型 API
- 复杂配置中心
- Docker、CI/CD 或部署系统

## 3. 不可随意改变的项目约定

以下默认值不得擅自更换：

- PDF 目录：`papers/`
- `chunk_size=500`
- `overlap=50`
- `top_k=3`
- Embedding 模型：`intfloat/multilingual-e5-small`
- Chroma collection：`science_papers`
- Ollama 模型：`deepseek-r1:7b`
- Ollama 默认地址：`http://127.0.0.1:11434`

固定技术栈：

- Python CLI
- PyMuPDF
- sentence-transformers
- ChromaDB
- requests 调用 Ollama HTTP API
- pytest

不要自行引入 LangChain、LlamaIndex、FastAPI、Streamlit 或另一个向量数据库。确实需要新增依赖时，先说明原因，确认它解决的是现有依赖无法解决的问题。

## 4. 工作方式

- 一次只处理一个明确功能。
- 优先完成当前步骤，不同时重构无关代码。
- 先读现有代码，再修改；不要凭文件名猜测实现。
- 优先采用简单、直接、容易测试的写法。
- 可以修复执行中发现的明显小问题，但不要借机扩大需求。
- 细节不明确时，先查看 `plan/plan1_step.md`。
- 如果计划没有规定，选择改动最小、风险最低的方案，并在结果中说明关键假设。
- 不要为了“显得完整”创建大量空文件、抽象层或占位类。
- 不要复制两套相同逻辑。

推荐顺序：

1. 检查当前状态。
2. 完成一个小功能。
3. 运行该功能相关测试。
4. 检查 diff。
5. 创建独立 commit。
6. 再进入下一个功能。

## 5. 代码基本要求

- 函数和变量使用能说明用途的名称。
- 关键函数添加类型标注。
- 用户可处理的错误使用清楚的中文提示。
- 不要使用空的 `except:` 或静默吞掉异常。
- 不要把绝对路径写死在代码中。
- 不要在日志中输出密钥、完整 prompt 或整篇论文内容。
- PDF 文件名、中文问题和中文回答必须正常处理。
- 文档与问题必须使用同一个 Embedding 模型。
- E5 文档使用 `passage:` 前缀，问题使用 `query:` 前缀。
- 文档和问题向量都要归一化。
- 引用论文名必须来自 Chroma metadata，不能依赖模型自己编造。
- Ollama 请求必须设置超时，并处理服务未启动、模型不存在和空响应。
- DeepSeek 的思考内容不作为最终回答输出。

## 6. 文件和数据安全

未经用户明确要求，不得：

- 删除或修改 `papers/` 中的 PDF。
- 删除计划文档或本规则。
- 使用 `git reset --hard`、强制 checkout、强制 push 等破坏性操作。
- 覆盖用户未提交的修改。
- 把 token、密码或 `.env` 内容提交到 Git。
- 把 `.venv/`、`chroma_db/`、模型缓存、Python 缓存提交到 Git。

索引可以按设计重建 `science_papers` collection，但必须先完成 PDF 提取和 Embedding，避免前置步骤失败时过早删除旧索引。

## 7. 测试和验证

每个功能完成后至少运行与它直接相关的测试。项目完成前运行：

```bash
python -m py_compile rag.py tests/test_rag.py
python -m pytest -q
python rag.py --help
```

测试规则：

- 单元测试不得依赖真实网络。
- 单元测试不得下载真实 Embedding 模型。
- HTTP 调用使用 mock。
- 测试数据库使用临时目录。
- 测试不得修改真实 `papers/`。
- 测试失败时先找真实原因，不删除关键断言来“让测试通过”。

条件允许时，再执行真实验证：

```bash
python rag.py index
python rag.py ask "这些论文主要研究了什么问题？"
```

如果网络、模型下载、Ollama 服务或机器资源导致真实验证无法进行，应保留已经完成的实现和单元测试，并准确说明未验证的部分。不得编造测试结果。

## 8. Git 规则

- 使用 `main` 分支。
- 远程仓库名保持为 `origin`。
- 每完成一个独立功能再 commit。
- commit 前运行相关测试和 `git diff --check`。
- 只暂存本功能涉及的文件。
- 使用清楚的 commit 信息，例如：

```text
feat: add PDF extraction and chunking
feat: persist embeddings in ChromaDB
feat: add retrieval and Ollama answering
test: cover RAG error handling
docs: add setup and usage guide
fix: handle empty PDF text
```

不要使用 `update`、`changes`、`fix stuff` 等含义不清的消息，也不要为了凑提交数量制造无意义 commit。

push 前先简短告诉用户将推送的分支、主要提交和测试结果。简单且已验证的修改在告知后可以直接 push。如果认证失败，保留本地 commit 并报告，不要反复尝试，不要要求用户在聊天中发送 token。

## 9. 遇到问题时

先进行最小范围排查：

1. 阅读完整错误信息。
2. 确认依赖、路径、环境变量和服务状态。
3. 检查问题是否已经在计划或 README 中说明。
4. 尝试风险较低的修复并重新运行相关测试。

以下情况应停止扩大修改并向用户说明：

- 需要用户选择会明显改变项目方向的方案。
- 必须删除或覆盖用户数据。
- 需要密钥、账户登录或新的外部权限。
- 同一个外部环境问题反复出现，代码本身无法解决。
- 需求与详细计划存在无法同时满足的冲突。

报告问题时说明：执行了什么、实际错误是什么、已经排除了什么、下一步需要什么。不要只说“失败了”。

## 10. 完成标准

只有同时满足以下条件，才可以说项目实现完成：

- 能读取带文本层的 PDF。
- 切块确实为 500/50。
- 文档向量已持久化到 ChromaDB，并带来源 metadata。
- 问题使用同模型生成向量。
- 每次检索 Top-3。
- 检索片段和问题被发送给 `deepseek-r1:7b`。
- 输出包含最终回答和去重后的论文文件名。
- 自动化测试通过，或明确列出未通过项和原因。
- README 中的安装和运行命令与实际代码一致。
- Git commit 按独立功能划分，工作区没有误提交运行产物。

最终汇报应包含：

1. 实现了哪些功能。
2. 实际运行了哪些测试及结果。
3. 真实 PDF、Embedding 和 Ollama 是否验证成功。
4. 创建了哪些 commits。
5. push 是否成功。
6. 仍有哪些限制或阻塞。
