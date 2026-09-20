# 科研 RAG（本地论文问答）

## 1. 项目简介

一个本地运行的极简科研论文问答命令行工具：递归读取 `papers/` 目录下的 PDF，提取文本并按 500 字符切块，使用轻量 Embedding 模型生成向量并持久化到 ChromaDB；提问时用同一个 Embedding 模型编码问题，检索最相关的 3 个文本块，交给本地 Ollama 的 `deepseek-r1:7b` 生成回答，并输出引用论文文件名。首版不包含 Web 界面、API 服务、OCR、多轮对话记忆、BM25 或 reranker。

## 2. 工作流程

```text
PDF -> 文本 -> 500/50 切块 -> E5 Embedding -> ChromaDB
问题 -> E5 Embedding -> Top-3 -> deepseek-r1:7b -> 回答 + 引用
```

索引阶段把向量、原文和来源元数据（论文文件名、相对路径、页码范围、块序号、文件 SHA-256）写入 ChromaDB；问答阶段只使用检索到的片段作为证据，引用论文名取自 Chroma metadata，不依赖模型自行生成。

## 3. 环境要求

- Python 3.10 或更高版本
- Ollama（本机已安装并能启动服务）
- 首次下载 Embedding 模型需要网络
- `deepseek-r1:7b` 需要足够的本机内存/显存（约 8 GB 内存可运行，显存约 6 GB 更快）

## 4. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. 准备模型

```bash
ollama serve
ollama pull deepseek-r1:7b
```

`ollama serve` 可能已由系统服务启动；如果端口已被占用，不要重复启动，直接继续下一步即可。

Embedding 模型 `intfloat/multilingual-e5-small` 会在首次索引时由 `sentence-transformers` 自动下载并缓存，不需要手动安装。

## 6. 准备论文和建立索引

```bash
cp /path/to/paper.pdf papers/
python rag.py index
```

- PDF 放在 `papers/` 目录或其子目录下均可，程序会递归查找（`.pdf` 和 `.PDF` 都识别）。
- 新增、修改、删除或改名 PDF 后必须重新运行 `python rag.py index`；首版是全量重建，会重建 `science_papers` collection。
- 索引只读取 PDF，不会修改或删除 `papers/` 中的文件。

## 7. 提问

```bash
python rag.py ask "这些论文使用了哪些实验方法？"
```

也可以不带参数运行，程序会提示输入一次问题：

```bash
python rag.py ask
```

示例输出（内容为占位说明，不代表真实论文结论）：

```text
回答：
根据检索片段，这些论文主要围绕某个科学问题展开，片段中提到的实验方法包括片段 1 中的某方法和片段 2 中的某方法 [1][2]。

引用论文：
- paper_a.pdf
- paper_b.pdf
```

输出始终包含「回答」和「引用论文」两部分；如果检索片段证据不足，回答会明确说明无法确定，而不是编造结论。

## 8. 配置

默认值集中在 `rag.py` 的配置对象中，可用同名环境变量覆盖，不需要改代码：

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `PAPERS_DIR` | `papers` | PDF 输入目录 |
| `CHROMA_DIR` | `chroma_db` | ChromaDB 持久化目录 |
| `COLLECTION_NAME` | `science_papers` | Chroma collection 名称 |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Embedding 模型 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP 地址 |
| `OLLAMA_MODEL` | `deepseek-r1:7b` | 生成模型 |
| **`CHUNK_SIZE`** | **`500`** | **文本块字符数（不可变约定）** |
| **`CHUNK_OVERLAP`** | **`50`** | **相邻文本块重叠字符数（不可变约定）** |
| **`TOP_K`** | **`3`** | **每次检索返回的文本块数量（不可变约定）** |
| `EMBEDDING_BATCH_SIZE` | `32` | Embedding 批大小 |
| `CHROMA_WRITE_BATCH_SIZE` | `256` | 写入 ChromaDB 的批大小 |
| `OLLAMA_CONNECT_TIMEOUT` | `5` | Ollama 连接超时（秒） |
| `OLLAMA_READ_TIMEOUT` | `300` | Ollama 读取超时（秒） |

整数环境变量必须为正整数；`CHUNK_OVERLAP` 可以为 0，但必须小于 `CHUNK_SIZE`，否则启动时报错。

如果 Ollama 监听在非默认端口（例如 `127.0.0.1:11436`），用环境变量指定，不要修改代码：

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11436 python rag.py ask "这些论文主要研究了什么问题？"
```

## 9. 测试

```bash
python -m pytest -q
```

单元测试不访问真实网络、不下载真实 Embedding 模型，Ollama 调用使用 mock，测试数据库使用临时目录，也不会修改 `papers/` 中的 PDF。

## 10. 常见问题

- **PDF 没有文本层**：首版不做 OCR，扫描件会得到空文本并被跳过。请使用带文本层的 PDF。
- **Ollama 连接失败**：运行 `ollama serve` 或检查服务状态、端口和 `OLLAMA_BASE_URL`。
- **模型不存在**：运行 `ollama pull deepseek-r1:7b`。
- **Embedding 首次加载慢**：首次运行会下载模型并写入本地缓存，之后启动会快很多。
- **修改论文后结果没变化**：重新运行 `python rag.py index` 重建索引。
- **CPU 运行较慢**：`multilingual-e5-small` 可以在 CPU 上运行，但首次索引和提问仍需要等待时间。
