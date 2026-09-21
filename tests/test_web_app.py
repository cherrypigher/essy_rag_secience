"""web_app.py 的单元测试。

只使用 Flask test client 和假对象：不访问真实网络、不加载真实 Embedding 模型、
不访问仓库真实 chroma_db/，也不监听真实端口。页面行为在此断言，检索、提示词和
Ollama 调用细节已由 tests/test_rag.py 覆盖。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import rag
import web_app
from rag import AnswerResult, Config, RagError


class FakeModel:
    """假 Embedding 模型：create_app 收到它就不会加载真实模型。"""


class FakeCollection:
    """假 Chroma collection：create_app 收到它就不会访问真实数据库。"""


def make_config() -> Config:
    """构造测试用 Config，字段与生产默认值一致。"""

    return Config(
        papers_dir=Path("papers"),
        chroma_dir=Path("chroma_db"),
        collection_name="science_papers",
        embedding_model="intfloat/multilingual-e5-small",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="deepseek-r1:7b",
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
        embedding_batch_size=32,
        chroma_write_batch_size=256,
        ollama_connect_timeout=5,
        ollama_read_timeout=300,
    )


@pytest.fixture
def app() -> object:
    """注入假 model 和假 collection 的 Flask 应用。"""

    return web_app.create_app(
        make_config(),
        model=FakeModel(),
        collection=FakeCollection(),
    )


def patch_generate_answer(
    monkeypatch: pytest.MonkeyPatch,
    result: AnswerResult | None = None,
    error: Exception | None = None,
) -> dict[str, object]:
    """替换 web_app 使用的问答函数，记录调用参数，不连接真实 Ollama。"""

    captured: dict[str, object] = {}

    def fake_generate_answer(
        question: str,
        config: Config,
        model: object,
        collection: object,
    ) -> AnswerResult:
        captured["question"] = question
        captured["config"] = config
        captured["model"] = model
        captured["collection"] = collection
        if error is not None:
            raise error
        assert result is not None
        return result

    monkeypatch.setattr(web_app, "generate_answer", fake_generate_answer)
    return captured


SUCCESS_RESULT = AnswerResult(
    answer="根据检索片段可以确定某项结论 [1]。",
    sources=("论文1.pdf", "论文4.pdf"),
)


class TestHomePage:
    def test_get_index_returns_200(self, app: object) -> None:
        response = app.test_client().get("/")  # type: ignore[attr-defined]
        assert response.status_code == 200

    def test_home_page_contains_form_elements(self, app: object) -> None:
        response = app.test_client().get("/")  # type: ignore[attr-defined]
        body = response.get_data(as_text=True)

        assert 'lang="zh-CN"' in body
        assert 'charset="utf-8"' in body
        assert "viewport" in body
        assert "科研论文问答" in body
        assert "<form" in body
        assert 'method="post"' in body
        assert 'action="/ask"' in body
        assert 'name="question"' in body
        assert "<textarea" in body
        assert "<label" in body
        assert "提交问题" in body
        assert "Top-3" in body

    def test_home_page_has_no_answer_or_error_area(self, app: object) -> None:
        response = app.test_client().get("/")  # type: ignore[attr-defined]
        body = response.get_data(as_text=True)

        assert "<h2>" not in body
        assert 'class="answer"' not in body
        assert 'class="sources"' not in body
        assert 'role="alert"' not in body

    def test_page_uses_local_stylesheet_without_cdn(self, app: object) -> None:
        response = app.test_client().get("/")  # type: ignore[attr-defined]
        body = response.get_data(as_text=True)

        assert "/static/style.css" in body
        assert 'href="http' not in body
        assert 'src="http' not in body


class TestAskRoute:
    def test_passes_question_and_shared_resources_to_generate_answer(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = patch_generate_answer(monkeypatch, SUCCESS_RESULT)
        client = app.test_client()  # type: ignore[attr-defined]

        response = client.post("/ask", data={"question": "这些论文研究了什么问题？"})

        assert response.status_code == 200
        assert captured["question"] == "这些论文研究了什么问题？"
        assert isinstance(captured["config"], Config)
        assert isinstance(captured["model"], FakeModel)
        assert isinstance(captured["collection"], FakeCollection)

    def test_success_shows_answer_and_deduplicated_sources(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_generate_answer(monkeypatch, SUCCESS_RESULT)
        client = app.test_client()  # type: ignore[attr-defined]

        response = client.post("/ask", data={"question": "问题"})
        body = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "根据检索片段可以确定某项结论 [1]。" in body
        assert "<li>论文1.pdf</li>" in body
        assert "<li>论文4.pdf</li>" in body
        assert "引用论文" in body

    def test_success_preserves_submitted_question(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_generate_answer(monkeypatch, SUCCESS_RESULT)
        client = app.test_client()  # type: ignore[attr-defined]

        response = client.post("/ask", data={"question": "已提交的问题"})
        body = response.get_data(as_text=True)

        assert "已提交的问题" in body

    def test_chinese_question_answer_and_filenames_are_not_mangled(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = AnswerResult(
            answer="这些论文主要研究了某个科学问题，并给出了相应结论 [1][2]。",
            sources=("论文一.pdf", "论文二.pdf"),
        )
        patch_generate_answer(monkeypatch, result)
        client = app.test_client()  # type: ignore[attr-defined]

        response = client.post("/ask", data={"question": "这些论文主要研究了什么问题？"})
        body = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "charset=utf-8" in response.headers["Content-Type"]
        assert "这些论文主要研究了什么问题？" in body
        assert "这些论文主要研究了某个科学问题" in body
        assert "论文一.pdf" in body
        assert "论文二.pdf" in body

    @pytest.mark.parametrize("question", ["", "   ", "\n\t "])
    def test_blank_question_returns_400_without_calling_rag(
        self, app: object, monkeypatch: pytest.MonkeyPatch, question: str
    ) -> None:
        def forbidden(*args: object, **kwargs: object) -> AnswerResult:
            raise AssertionError("问题为空时不得调用 RAG 流程")

        monkeypatch.setattr(web_app, "generate_answer", forbidden)
        client = app.test_client()  # type: ignore[attr-defined]

        response = client.post("/ask", data={"question": question})
        body = response.get_data(as_text=True)

        assert response.status_code == 400
        assert "问题不能为空" in body

    def test_rag_error_returns_503_with_chinese_message(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_generate_answer(
            monkeypatch,
            error=RagError(
                "无法连接 Ollama 服务（http://127.0.0.1:11434/api/chat）：连接被拒绝。"
                "请确认已运行 ollama serve。"
            ),
        )
        client = app.test_client()  # type: ignore[attr-defined]

        response = client.post("/ask", data={"question": "问题"})
        body = response.get_data(as_text=True)

        assert response.status_code == 503
        assert "无法连接 Ollama 服务" in body
        assert "ollama serve" in body
        assert "问题" in body  # 用户输入被保留

    def test_page_hides_traceback_prompt_and_thinking(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_generate_answer(monkeypatch, SUCCESS_RESULT)
        client = app.test_client()  # type: ignore[attr-defined]

        response = client.post("/ask", data={"question": "问题"})
        body = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "Traceback" not in body
        assert "File \"" not in body
        assert "<think>" not in body
        # 页面只显示回答与引用，不得泄露完整 prompt 或检索片段原文
        assert "你是科研论文问答助手" not in body
        assert "用户问题：" not in body
        assert "页码：" not in body
        assert "chunk number" not in body

    def test_script_tag_in_question_is_escaped(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_generate_answer(monkeypatch, SUCCESS_RESULT)
        client = app.test_client()  # type: ignore[attr-defined]

        response = client.post(
            "/ask", data={"question": "<script>alert(1)</script>"}
        )
        body = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "<script>alert(1)</script>" not in body
        assert "<script>" not in body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body

    def test_html_in_answer_is_escaped(
        self, app: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = AnswerResult(
            answer="回答包含 <b>伪 HTML</b> 和 <img src=x onerror=alert(1)>。",
            sources=("论文1.pdf",),
        )
        patch_generate_answer(monkeypatch, result)
        client = app.test_client()  # type: ignore[attr-defined]

        response = client.post("/ask", data={"question": "问题"})
        body = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "<b>伪 HTML</b>" not in body
        assert "<img src=x" not in body
        assert "&lt;b&gt;伪 HTML&lt;/b&gt;" in body


class TestResourceLifecycle:
    def test_create_app_with_fakes_does_not_touch_real_resources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*args: object, **kwargs: object):
            raise AssertionError("注入假对象时不得加载真实模型或访问真实 ChromaDB")

        monkeypatch.setattr(web_app, "load_embedding_model", forbidden)
        monkeypatch.setattr(web_app, "create_chroma_client", forbidden)
        monkeypatch.setattr(web_app, "get_index_collection", forbidden)

        app = web_app.create_app(
            make_config(),
            model=FakeModel(),
            collection=FakeCollection(),
        )

        assert app is not None

    def test_create_app_opens_existing_collection_and_loads_model_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"model": 0, "collection": 0}

        def fake_load_model(name: str) -> FakeModel:
            calls["model"] += 1
            return FakeModel()

        def fake_create_client(chroma_dir: Path) -> object:
            return object()

        def fake_get_collection(client: object, config: Config) -> FakeCollection:
            calls["collection"] += 1
            return FakeCollection()

        monkeypatch.setattr(web_app, "load_embedding_model", fake_load_model)
        monkeypatch.setattr(web_app, "create_chroma_client", fake_create_client)
        monkeypatch.setattr(web_app, "get_index_collection", fake_get_collection)

        app = web_app.create_app(make_config())
        app_config = app.config

        assert calls == {"model": 1, "collection": 1}
        assert isinstance(app_config["rag_model"], FakeModel)
        assert isinstance(app_config["rag_collection"], FakeCollection)

    def test_importing_web_app_does_not_load_resources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """import web_app 只应定义函数，不能创建真实 app 或加载模型。"""

        def forbidden(*args: object, **kwargs: object):
            raise AssertionError("import web_app 不得加载真实模型或数据库")

        monkeypatch.setattr(rag, "load_embedding_model", forbidden)
        monkeypatch.setattr(rag, "create_chroma_client", forbidden)
        monkeypatch.setattr(rag, "get_index_collection", forbidden)

        importlib.reload(web_app)

        monkeypatch.undo()
        importlib.reload(web_app)
