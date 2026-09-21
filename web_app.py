"""本地网页问答入口。

复用 ``rag.py`` 的问答编排：进程启动时加载一次 Embedding 模型并打开一次
ChromaDB collection，之后每次提问只做「问题 Embedding -> Top-3 -> Ollama ->
回答与引用」。服务只监听本机 ``127.0.0.1:8000``，面向单用户，不是公网服务。
"""

from __future__ import annotations

import sys
from typing import Any

from flask import Flask, render_template, request

from rag import (
    AnswerResult,
    Config,
    RagError,
    create_chroma_client,
    generate_answer,
    get_index_collection,
    load_embedding_model,
)

WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
EMPTY_QUESTION_MESSAGE = "问题不能为空，请输入要回答的问题。"


def create_app(
    config: Config | None = None,
    *,
    model: Any | None = None,
    collection: Any | None = None,
) -> Flask:
    """创建 Flask 应用，启动时准备好问答所需的全部资源。

    ``config``、``model``、``collection`` 都可以由调用方注入：测试传入假
    model 和假 collection，从而不加载真实模型、不访问真实 ChromaDB。
    创建应用时不会调用 Ollama，Ollama 只在提交问题后调用。
    """

    if config is None:
        config = Config.from_env()
    if collection is None:
        client = create_chroma_client(config.chroma_dir)
        collection = get_index_collection(client, config)
    if model is None:
        model = load_embedding_model(config.embedding_model)

    app = Flask(__name__)
    app.config["rag_config"] = config
    app.config["rag_model"] = model
    app.config["rag_collection"] = collection

    @app.get("/")
    def index() -> str:
        """首页：空白表单，还没有问题和回答。"""

        return render_template(
            "index.html",
            question="",
            answer=None,
            sources=(),
            error=None,
        )

    @app.post("/ask")
    def ask() -> str | tuple[str, int]:
        """提交问题：复用 generate_answer，错误以中文显示而不暴露 traceback。"""

        question = request.form.get("question", "")
        if not question.strip():
            return (
                render_template(
                    "index.html",
                    question=question,
                    answer=None,
                    sources=(),
                    error=EMPTY_QUESTION_MESSAGE,
                ),
                400,
            )

        try:
            result: AnswerResult = generate_answer(question, config, model, collection)
        except RagError as exc:
            return (
                render_template(
                    "index.html",
                    question=question,
                    answer=None,
                    sources=(),
                    error=str(exc),
                ),
                503,
            )

        return render_template(
            "index.html",
            question=question,
            answer=result.answer,
            sources=result.sources,
            error=None,
        )

    return app


def main() -> int:
    """网页入口：启动失败时输出中文错误并返回 1，不启动不可用的服务。"""

    try:
        app = create_app()
    except RagError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(f"网页问答地址：http://{WEB_HOST}:{WEB_PORT}")
    try:
        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            debug=False,
            use_reloader=False,
            threaded=False,
        )
    except KeyboardInterrupt:
        print("已停止网页服务。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
