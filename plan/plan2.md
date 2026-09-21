# 极简科研 RAG 网页问答：逐步实施手册

> 本文档交给后续执行模型使用。执行模型能力可能有限，因此必须按步骤顺序实施，
> 每一步只做指定范围内的修改，完成测试后再进入下一步。不得跳过验证后直接声称完成。

## 0. 文档定位与最终目标

当前项目已经完成并验证了命令行版科研 RAG：

```text
papers/ PDF
  -> 500/50 字符切块
  -> multilingual-e5-small Embedding
  -> ChromaDB science_papers
  -> Top-3 检索
  -> Ollama deepseek-r1:7b
  -> 回答 + 去重论文文件名
```

本计划只增加一个本机网页问答入口。用户必须能够在项目根目录运行：

```bash
.venv/bin/python web_app.py
```

然后用浏览器访问：

```text
http://127.0.0.1:8000
```

网页必须允许用户：

1. 输入一个问题。
2. 点击“提交问题”。
3. 等待现有 RAG 流程完成。
4. 在同一页面看到最终回答。
5. 在回答下方看到从 Chroma metadata 得到的去重论文文件名。
6. 在索引、Embedding 或 Ollama 不可用时看到简洁中文错误，而不是浏览器 traceback。

网页和 CLI 必须复用同一套问答函数。不得复制检索、提示词或 Ollama 调用逻辑。

用户已经明确要求增加网页端，因此这次允许在原有 CLI 之外加入一个极简 Web UI。
本文是网页阶段的主要执行依据；原有 RAG 行为仍以 `rule.md` 和
`plan/plan1_step.md` 为准。如果网页计划与原有 RAG 硬约定冲突，必须保留原有硬约定。

---

## 1. 当前基线

开始实现前应确认以下基线；实际状态不同则先记录，不要覆盖用户修改：

- 分支：`main`。
- 当前已知提交：`5d8938b fix: harden loopback proxy bypass and required env validation`。
- CLI 自动化测试：98 项通过。
- 真实索引已存在：`chroma_db/` 中的 `science_papers` collection 有 1053 条记录。
- 真实 Ollama 使用过 `http://127.0.0.1:11436`，但代码默认值仍必须保持
  `http://127.0.0.1:11434`。
- 当前 `.venv/bin/python` 可运行，但本机 `.venv/bin/activate` 可能不存在。
  执行命令时优先直接使用 `.venv/bin/python`，不要为了网页功能删除或重建 `.venv/`。
- `plan/plan2.md` 在编写本计划前是一个空的未跟踪文件。

开始工作时完整阅读：

1. `rule.md`
2. `plan/plan1_step.md`
3. `plan/plan1.md`
4. 本文档 `plan/plan2.md`
5. `rag.py`
6. `tests/test_rag.py`
7. `README.md`

然后运行：

```bash
pwd
git status --short --branch
git log --oneline -5
.venv/bin/python -m pytest -q
```

要求：

- 当前目录必须是仓库根目录。
- 不得覆盖或删除已有未提交修改。
- 原有 98 项测试必须先通过；若基线已经失败，先报告，不要把失败归因于网页改动。

---

## 2. 本轮范围

### 2.1 必须实现

- 一个本机 Flask 网页服务。
- 一个包含问题输入框和提交按钮的页面。
- 页面显示回答与去重后的引用论文文件名。
- 网页复用现有 Top-3、E5、Chroma 和 Ollama 流程。
- 服务启动时加载一次 Embedding 模型并打开一次 collection；每次提问不得重新加载模型。
- 中文问题、回答、错误和论文文件名正常显示。
- 保留原有 `rag.py index` 和 `rag.py ask` 行为及输出格式。
- Flask 路由自动化测试不得访问真实网络、真实 Ollama或下载真实模型。
- README 增加网页启动和使用说明。

### 2.2 明确不实现

以下内容不属于本作业，不得顺手加入：

- PDF 上传、删除、改名或在线管理。
- 网页中的“建立索引”按钮或索引进度页面。
- 多轮聊天、会话历史或数据库保存问题。
- 登录、用户、权限、配额或管理员页面。
- 流式输出、WebSocket、SSE。
- React、Vue、Node.js、npm 或其他前端构建工具。
- JSON 公共 API、OpenAPI 文档或移动端接口。
- OCR、BM25、reranker、混合检索或更换模型。
- 云端部署、Docker、Nginx、HTTPS、systemd、CI/CD。
- 多进程模型共享、任务队列、Redis 或性能压测。
- 论文片段全文展示、相似度调试面板或复杂可视化。

网页只面向本机单用户。不要把 Flask 开发服务器描述为公网生产服务。

---

## 3. 技术方案

### 3.1 Web 框架

新增唯一直接依赖：

```text
Flask>=3.0,<4.0
```

选择 Flask 的原因：

- 用户明确要求网页端，必须有 HTTP 服务和 HTML 渲染能力。
- Flask 依赖较小，服务端表单和测试客户端足以完成本作业。
- 不需要前端构建链。
- 比手写 `http.server` 的表单解析、错误处理和模板转义更简单可靠。

不要改用 FastAPI、Streamlit、Django、Gradio、React 或 Vue。

### 3.2 页面实现

使用 Flask + Jinja 模板 + 一个本地 CSS 文件：

- HTML 由服务端渲染。
- 表单使用普通 `POST`。
- 不依赖 JavaScript。
- 不引用 CDN、远程字体或远程图片。
- 依赖 Jinja 默认自动转义，禁止对问题、回答和错误使用 `|safe`。

### 3.3 服务地址

固定本阶段 Web 地址：

```text
host = 127.0.0.1
port = 8000
```

不要把 Web 地址和 Ollama 地址混淆：

- 浏览器访问：`http://127.0.0.1:8000`
- Ollama 默认：`http://127.0.0.1:11434`
- 当前机器真实 Ollama 可能需要：`OLLAMA_BASE_URL=http://127.0.0.1:11436`

本阶段不新增 `WEB_HOST`、`WEB_PORT` 等配置项，避免扩大配置范围。

### 3.4 资源生命周期

Flask 服务启动时完成：

1. `Config.from_env()`。
2. `create_chroma_client(config.chroma_dir)`。
3. `get_index_collection(client, config)`。
4. `load_embedding_model(config.embedding_model)`。

以上资源在该进程内复用。每次浏览器提交问题时只做：

```text
问题 Embedding -> Top-3 -> prompt -> Ollama -> 回答与引用
```

不要在每个请求中重新加载 SentenceTransformer，也不要重新读取 PDF 或重建索引。

为避免在简单作业中处理共享模型的并发细节，Flask 启动时使用：

```python
app.run(
    host="127.0.0.1",
    port=8000,
    debug=False,
    use_reloader=False,
    threaded=False,
)
```

`use_reloader=False` 可避免模型被开发服务器重复加载。

---

## 4. 完成后的目录结构

只增加必要文件：

```text
science_rag/
├── rag.py
├── web_app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── tests/
│   ├── test_rag.py
│   └── test_web_app.py
├── plan/
│   ├── plan1.md
│   ├── plan1_step.md
│   └── plan2.md
├── papers/
└── chroma_db/             # 运行数据，继续由 Git 忽略
```

不要建立 `services/`、`controllers/`、`repositories/`、`frontend/` 等多层目录。

---

## 5. 核心代码契约

网页不能通过捕获 `answer_question()` 的 stdout 获得答案。必须先把现有问答计算与 CLI
打印做一次很小的分离。

### 5.1 新增 `AnswerResult`

在 `rag.py` 的其他 dataclass 附近增加：

```python
@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: tuple[str, ...]
```

字段含义：

- `answer`：已经执行 `clean_model_answer()` 的最终回答。
- `sources`：按检索排名第一次出现顺序去重后的论文文件名。

不得把完整 prompt、Embedding、Chroma 对象或整篇论文放入返回对象。

### 5.2 新增纯问答编排函数

在 `rag.py` 中增加：

```python
def generate_answer(
    question: str,
    config: Config,
    model: SentenceTransformer,
    collection: Any,
) -> AnswerResult:
```

固定流程：

1. `question.strip()`。
2. 空问题抛出原有 `RagError("问题不能为空……")`。
3. `embed_query(model, cleaned_question)`。
4. `retrieve(collection, query_embedding, config.top_k)`。
5. `build_ollama_messages(cleaned_question, hits)`。
6. `call_ollama(messages, config)`。
7. `clean_model_answer(...)`。
8. `tuple(cited_sources(hits))`。
9. 返回 `AnswerResult`。

该函数不得：

- 打印终端输出。
- 创建 Chroma client。
- 加载 Embedding 模型。
- 修改 collection。
- 读取 PDF。
- 捕获并吞掉 `RagError`。

这样 CLI 与网页可以共享完全相同的检索和生成流程，同时网页服务能复用已加载的资源。

### 5.3 保留 CLI `answer_question`

现有函数签名保持不变：

```python
def answer_question(question: str, config: Config) -> None:
```

重构后的固定流程：

1. 在访问数据库和模型之前继续验证空问题。
2. 创建 Chroma client。
3. 调用 `get_index_collection`。
4. 加载一次 Embedding 模型。
5. 调用 `generate_answer(question, config, model, collection)`。
6. 按原格式打印：

   ```text
   回答：
   <answer>

   引用论文：
   - <source>
   ```

CLI 输出内容和错误语义不得变化。已有 `tests/test_rag.py` 的断言不得删除或放宽。

---

## 6. `web_app.py` 契约

### 6.1 工厂函数

实现：

```python
def create_app(
    config: Config | None = None,
    *,
    model: Any | None = None,
    collection: Any | None = None,
) -> Flask:
```

规则：

- `config is None` 时调用 `Config.from_env()`。
- `collection is None` 时创建 Chroma client 并校验已有 collection。
- `model is None` 时加载 `config.embedding_model`。
- 测试会同时传入 fake model 和 fake collection，此时不得访问真实 Chroma 或下载模型。
- 把三个对象保存在闭包或 Flask app config 中均可，但不要使用可变模块级全局变量。
- 创建 app 时不得调用 Ollama；Ollama 只在提交问题后调用。

### 6.2 GET `/`

访问首页时返回 `templates/index.html`，模板变量固定为：

```python
question=""
answer=None
sources=()
error=None
```

HTTP 状态码为 200。

### 6.3 POST `/ask`

从表单读取：

```python
question = request.form.get("question", "")
```

行为：

1. 空字符串或纯空白：不调用 RAG，重新渲染页面，显示“问题不能为空”，状态码 400。
2. 正常问题：调用 `generate_answer(question, config, model, collection)`。
3. 成功：渲染同一页面，保留用户问题，显示回答和 sources，状态码 200。
4. 捕获 `RagError`：渲染同一页面，保留问题并显示 `str(exc)`，状态码 503。
5. 不要把异常 traceback、完整 prompt 或论文正文放到页面。
6. 不要使用 `flash`、session 或 cookies；直接通过模板变量显示结果。

不要捕获空的 `except:`。未预见的编程错误应由 Flask 记录并返回通用 500，不能伪装成回答。

### 6.4 启动入口

实现：

```python
def main() -> int:
```

规则：

- 创建 app 期间的 `RagError` 输出到 stderr，前缀为 `错误：`，返回 1。
- 成功时在终端打印 `网页问答地址：http://127.0.0.1:8000`。
- 使用第 3.4 节固定参数调用 `app.run()`。
- `KeyboardInterrupt` 可正常结束，不输出 traceback。
- 文件末尾使用：

  ```python
  if __name__ == "__main__":
      raise SystemExit(main())
  ```

不得在模块 import 时直接创建真实 app，否则测试导入 `web_app` 会加载模型和数据库。

---

## 7. 页面契约

### 7.1 `templates/index.html`

页面至少包含：

- `<html lang="zh-CN">`
- `<meta charset="utf-8">`
- 移动端 viewport。
- 标题“科研论文问答”。
- 一句说明：回答基于本地论文的 Top-3 检索片段。
- `<form method="post" action="{{ url_for('ask') }}">`。
- 与输入框关联的 `<label>`。
- `name="question"` 的 `<textarea>`。
- 文本为“提交问题”的 submit 按钮。
- 可选的示例问题纯文本。
- 有错误时显示中文错误区域。
- 有回答时显示“回答”和正文。
- 有 sources 时显示“引用论文”无序列表。

模板应保留已提交问题：

```html
<textarea name="question">{{ question }}</textarea>
```

不得：

- 使用 `|safe`。
- 把回答当 HTML 渲染。
- 引入外部脚本或 CSS。
- 显示完整检索片段、距离或 prompt。
- 声称回答一定正确。

### 7.2 `static/style.css`

CSS 只完成基本可用性：

- 页面内容居中并设置合理最大宽度。
- 输入框宽度 100%，高度足够输入中文问题。
- 按钮有清晰 hover/focus 状态。
- 回答区使用 `white-space: pre-wrap`，保留模型换行。
- 错误区颜色明显但文字可读。
- 引用列表清楚。
- 小屏幕下有合理内边距。

不要追求复杂动画、主题切换、图标库或设计系统。

---

## 8. 逐步实施步骤

### 步骤 0：基线检查并提交本计划

执行第 1 节的检查命令。只检查，不修改业务代码。

检查本计划：

```bash
git diff --check -- plan/plan2.md
git diff --stat -- plan/plan2.md
```

如果负责执行的模型被要求提交计划文档，则只暂存本文件：

```bash
git add plan/plan2.md
git diff --cached --check
git commit -m "docs: add web Q&A implementation plan"
```

不要把业务实现混入计划文档提交。

### 步骤 1：分离问答结果与 CLI 打印

只修改：

- `rag.py`
- `tests/test_rag.py`

实现第 5 节的 `AnswerResult` 和 `generate_answer`，再让 `answer_question` 调用它。

至少补充以下测试：

1. `generate_answer` 返回 `AnswerResult`。
2. 返回的 answer 已移除 `<think>...</think>`。
3. sources 来自检索 metadata，并按 `source_path` 去重。
4. 空问题在访问 collection、Embedding 或 Ollama 前报错。
5. `answer_question` 的终端输出仍包含“回答”和“引用论文”。
6. 原有 98 项测试全部保留并通过。

验证：

```bash
.venv/bin/python -m pytest -q tests/test_rag.py
.venv/bin/python -m py_compile rag.py tests/test_rag.py
git diff --check
```

建议 commit：

```text
refactor: share RAG answer results across interfaces
```

这是为了复用现有能力，不得在此步骤改检索算法、提示词、模型、切块或索引。

### 步骤 2：添加 Flask 依赖与 Web 骨架

修改或新增：

- `requirements.txt`
- `web_app.py`
- `tests/test_web_app.py`

在 `requirements.txt` 增加且只增加：

```text
Flask>=3.0,<4.0
```

不要改变现有依赖范围。

如当前环境没有 Flask，再执行：

```bash
.venv/bin/python -m pip install "Flask>=3.0,<4.0"
```

如果下载因网络限制失败，应申请允许正常安装依赖；不要改用另一个框架规避。

先实现 `create_app`、GET `/` 和 `main()` 骨架，暂时可以使用最小模板，确保 import
`web_app` 不加载真实模型。

验证：

```bash
.venv/bin/python -m py_compile rag.py web_app.py tests/test_rag.py tests/test_web_app.py
.venv/bin/python -m pytest -q tests/test_web_app.py
```

不要在这一小步启动真实 Ollama。

### 步骤 3：实现页面和问答 POST

新增：

- `templates/index.html`
- `static/style.css`

完善：

- `web_app.py`
- `tests/test_web_app.py`

严格实现第 6、7 节契约。页面只显示问题、回答、引用和错误。

至少实现以下测试：

1. GET `/` 返回 200。
2. 首页包含中文标题、表单、`question` textarea 和提交按钮。
3. POST `/ask` 会把原问题传给 `generate_answer`。
4. 成功回答显示 answer 和全部去重 sources。
5. 中文问题、中文回答和中文文件名不会乱码。
6. 空问题返回 400、显示“问题不能为空”，且不调用 RAG。
7. `generate_answer` 抛 `RagError` 时返回 503 并显示中文错误。
8. 页面不出现 traceback、完整 prompt 或 `<think>`。
9. 输入 `<script>alert(1)</script>` 时响应中必须被 HTML 转义，不能原样成为脚本标签。
10. 回答中的 `<b>伪 HTML</b>` 也必须被转义。
11. 向 `create_app` 传 fake model 和 fake collection 时，不加载真实模型、不访问真实 Chroma。

所有测试必须使用 Flask test client 和 fake/monkeypatch。禁止在单元测试中监听真实端口。

验证：

```bash
.venv/bin/python -m pytest -q tests/test_web_app.py
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile rag.py web_app.py tests/test_rag.py tests/test_web_app.py
git diff --check
```

建议 commit：

```text
feat: add local web interface for paper Q&A
```

### 步骤 4：更新 README

只更新 `README.md`，要求：

1. 项目简介改为同时支持 CLI 和本地网页，不再写“首版不包含 Web 界面”。
2. 保留原有索引、CLI 提问、配置和故障说明。
3. 新增“启动网页”章节：

   ```bash
   .venv/bin/python rag.py index
   OLLAMA_BASE_URL=http://127.0.0.1:11436 .venv/bin/python web_app.py
   ```

4. 写明默认 Ollama 在 11434；上面 11436 只是当前机器示例。
5. 写明浏览器访问 `http://127.0.0.1:8000`。
6. 写明必须先建立索引，网页不负责上传或重建索引。
7. 写明网页服务停止方式是终端按 `Ctrl+C`。
8. 同时给出标准虚拟环境激活方式和当前环境可用的直接解释器方式：

   ```bash
   source .venv/bin/activate
   # 如果 activate 不存在：
   .venv/bin/python web_app.py
   ```

9. 测试命令继续使用 `python -m pytest -q`，并可补充直接解释器形式。

验证 README 命令与实际代码一致。建议 commit：

```text
docs: document local web Q&A usage
```

### 步骤 5：完整自动化验证

运行：

```bash
.venv/bin/python -m py_compile rag.py web_app.py tests/test_rag.py tests/test_web_app.py
.venv/bin/python -m pytest -q
.venv/bin/python rag.py --help
.venv/bin/python rag.py index --help
.venv/bin/python rag.py ask --help
```

要求：

- 原有 98 项测试不能减少。
- 新增 Web 测试全部通过。
- 测试不访问真实 Ollama、不下载模型、不修改 `papers/` 和真实 `chroma_db/`。
- `git diff --check` 无输出。
- `git status --short` 不出现 `.venv/`、`chroma_db/`、缓存或模型文件。

额外验证启动失败提示。使用一个不存在的临时 Chroma 路径：

```bash
CHROMA_DIR=/tmp/science_rag_missing_web_index \
  .venv/bin/python web_app.py
```

预期：

- 中文提示先运行 `python rag.py index`。
- 退出码为 1。
- 不启动 Web 服务。
- 不出现难以理解的 traceback。

测试后不要删除用户数据；该临时目录若由 Chroma 自动创建且确需清理，只清理这个明确路径。

### 步骤 6：真实网页验收

真实索引已经存在时，不需要为了网页功能重复建立索引。先确认：

```bash
.venv/bin/python -c "import chromadb; c=chromadb.PersistentClient(path='chroma_db').get_collection('science_papers'); print(c.count(), c.metadata)"
```

当前机器若 Ollama 在 11436，启动：

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11436 .venv/bin/python web_app.py
```

浏览器访问：

```text
http://127.0.0.1:8000
```

至少测试两个问题：

```text
这些论文主要研究了什么问题？
根据检索到的内容，论文采用了哪些实验或评估方法？
```

检查：

1. 首页可以打开。
2. 输入问题后按钮可以提交。
3. 等待期间服务未崩溃。
4. 页面显示非空最终回答。
5. 页面不显示 `<think>` 或独立思考过程。
6. 页面显示引用论文文件名。
7. 同一 `source_path` 只显示一次。
8. 中文文件名正常。
9. 终端日志不打印完整 prompt 或整篇论文。
10. 第二次提问不重新加载 Embedding 模型；日志中不应再次出现模型加载过程。

再验证错误展示。可以在另一个终端用错误 Ollama 地址启动一份服务，但不要停止用户正在使用的
Ollama：

```bash
OLLAMA_BASE_URL=http://127.0.0.1:1 .venv/bin/python web_app.py
```

提交问题后，页面应显示包含 `ollama serve` 的中文连接错误，而不是 traceback。
如果 8000 已被正常服务占用，先用 `Ctrl+C` 正常停止前一个 Web 进程，再测试错误地址；
不要启动多个监听同一端口的实例。

### 步骤 7：最终审查与交付

检查：

```bash
git status --short --branch
git log --oneline --decorate -10
git diff --check
git ls-files
```

确认：

- 未修改或删除 `papers/` 中的 PDF。
- 未提交 `chroma_db/`、`.venv/`、缓存或模型文件。
- 没有引入前端构建产物。
- requirements 只新增 Flask。
- CLI 和网页调用同一个 `generate_answer`。
- 页面引用来自 `cited_sources(hits)`，不是解析模型回答。
- Flask 未开启 debug 和 reloader。
- 没有 `app.run(host="0.0.0.0")`。
- README 启动命令可复制执行。

push 前先告诉用户：

- 将推送 `main`。
- 新增了哪些 commits。
- 自动测试通过数量。
- 真实网页和 Ollama 是否验证成功。

认证失败时保留本地 commit 并报告，不修改远程 URL，不要求用户发送 token。

---

## 9. 自动化测试详细要求

### 9.1 原有测试保护

不得删除或弱化以下原有能力的测试：

- 500/50 切块。
- E5 `passage:` / `query:` 前缀。
- 向量归一化。
- Chroma cosine、metadata 与持久化。
- Top-3 检索。
- Ollama timeout、缺模型、连接失败和非法响应。
- `<think>` 清理。
- 引用按 `source_path` 去重。
- CLI 中文错误和退出码。

### 9.2 Web 测试隔离

Web 单元测试必须满足：

- 使用 Flask `app.test_client()`。
- 给 `create_app` 传 fake model 和 fake collection。
- monkeypatch `generate_answer` 或 `call_ollama`，不得连接真实 Ollama。
- 不加载 SentenceTransformer。
- 不访问仓库真实 `chroma_db/`。
- 不启动真实 HTTP 监听端口。
- 不依赖测试执行顺序。

### 9.3 推荐测试辅助对象

`tests/test_web_app.py` 可以定义最小 fake：

```python
class FakeModel:
    pass


class FakeCollection:
    pass
```

如果 route 层 monkeypatch 了 `generate_answer`，fake 不需要实现无关接口。不要复制
`tests/test_rag.py` 中的大量 fake 逻辑。

成功结果使用：

```python
AnswerResult(
    answer="根据检索片段可以确定某项结论 [1]。",
    sources=("论文1.pdf", "论文4.pdf"),
)
```

测试只断言页面行为，不重新测试 `retrieve`、prompt 或 HTTP payload；这些已经由
`tests/test_rag.py` 覆盖。

---

## 10. 错误处理矩阵

| 场景 | 页面/终端行为 | 不允许的行为 |
| --- | --- | --- |
| 问题为空 | 页面显示“问题不能为空”，HTTP 400 | 调用 Embedding 或 Ollama |
| collection 不存在 | 服务启动失败，终端提示先 index，退出 1 | 启动一个必然不可用的页面 |
| collection 为空 | 同上 | 返回空答案 |
| 索引配置不匹配 | 服务启动失败并提示重建索引 | 用不同模型查询旧向量 |
| Embedding 模型加载失败 | 服务启动失败并显示原有中文提示 | 吞掉异常继续启动 |
| Ollama 无法连接 | 页面显示原有中文错误，HTTP 503 | 浏览器 traceback |
| Ollama 超时 | 页面显示原有超时提示，HTTP 503 | 无限等待或伪答案 |
| 模型不存在 | 页面提示 `ollama pull <模型名>` | 硬编码错误模型名 |
| Ollama 空响应 | 页面显示原有中文错误，HTTP 503 | 显示空白回答 |
| 回答含 HTML | Jinja 转义后显示为文本 | 执行 HTML/脚本 |

---

## 11. 完整验收矩阵

| 编号 | 需求 | 实现证据 | 测试/真实验收 |
| --- | --- | --- | --- |
| W1 | 本地网页可打开 | Flask GET `/` | test client 200 + 浏览器访问 |
| W2 | 网页可提交问题 | POST `/ask` | route 单测 + 真实问题 |
| W3 | 复用现有 RAG | `generate_answer` | CLI 与 Web 共用函数测试 |
| W4 | 只检索 Top-3 | 原 `retrieve` | 原有测试继续通过 |
| W5 | 页面显示最终回答 | `AnswerResult.answer` | 正常 POST 测试 |
| W6 | 页面显示去重论文名 | `AnswerResult.sources` | 中文文件名与去重测试 |
| W7 | 不输出思考过程 | `clean_model_answer` | 原测试 + 页面检查 |
| W8 | 中文错误可见 | route 捕获 `RagError` | 400/503 测试 |
| W9 | 防止 HTML 注入 | Jinja autoescape | script/b 标签转义测试 |
| W10 | 模型只加载一次 | app 启动时加载 | fake 调用次数 + 真实日志 |
| W11 | CLI 无回归 | 保留 `answer_question` | 原有 98 项测试全部通过 |
| W12 | 文档可执行 | README | 按文档启动真实网页 |

W1-W12 全部满足后才能声称网页阶段完成。

---

## 12. 禁止的错误实现

以下实现即使页面“看起来能用”也不合格：

- 在 Web route 中重新实现 Top-3、prompt 或 Ollama payload。
- 通过重定向 stdout 解析 CLI 输出。
- 每次请求重新加载 SentenceTransformer。
- 每次请求重新扫描 PDF 或运行 `build_index`。
- 从模型回答文本中猜测引用论文名。
- 把用户问题拼进 HTML 字符串而不转义。
- 使用 `|safe` 显示模型回答。
- 在 import `web_app` 时加载真实模型。
- Flask `debug=True` 或启用 reloader 导致模型加载两次。
- 监听 `0.0.0.0` 后声称只对本机开放。
- 把完整 prompt、Top-3 原文或 traceback 返回浏览器。
- 测试访问真实 Ollama、真实模型或真实索引。
- 为网页增加上传、历史、登录、OCR、reranker 或流式输出。
- 为了通过 Web 测试删除原 CLI 断言。

---

## 13. 最短执行清单

- [ ] 步骤 0：检查仓库、基线测试和本计划。
- [ ] 步骤 1：增加 `AnswerResult` / `generate_answer`，保持 CLI 不变。
- [ ] 步骤 2：增加 Flask 依赖、`web_app.py` 和测试骨架。
- [ ] 步骤 3：实现模板、CSS、POST 问答与错误展示。
- [ ] 步骤 4：更新 README 网页运行说明。
- [ ] 步骤 5：运行完整自动化验证，原 98 项不得减少。
- [ ] 步骤 6：使用真实 Chroma、Embedding、Ollama 和浏览器验收。
- [ ] 步骤 7：审查 Git、告知用户、提交并 push。

最终交付命令应保持简单：

```bash
# 论文变化后才需要重建索引
.venv/bin/python rag.py index

# 当前机器 Ollama 使用 11436 时
OLLAMA_BASE_URL=http://127.0.0.1:11436 .venv/bin/python web_app.py

# 浏览器打开
http://127.0.0.1:8000
```
