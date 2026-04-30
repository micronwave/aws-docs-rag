import importlib.util
import io
import json
import sys
import types
import uuid
from pathlib import Path

import pytest
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parents[1]


class FakeBody:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return self.payload.encode("utf-8")


class FakeBedrock:
    def __init__(self, *, embed_payload=None, llm_payload=None, error=None):
        self.embed_payload = embed_payload
        self.llm_payload = llm_payload
        self.error = error
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error

        model_id = kwargs["modelId"]
        if "embed" in model_id:
            return {"body": FakeBody(json.dumps({"embedding": self.embed_payload}))}
        return {"body": FakeBody(json.dumps(self.llm_payload))}


def load_script(monkeypatch, module_name: str, relative_path: str, *, bedrock=None, pinecone_module=None):
    boto3_stub = types.ModuleType("boto3")
    boto3_stub.client = lambda service_name, **_kwargs: bedrock
    monkeypatch.setitem(sys.modules, "boto3", boto3_stub)

    if pinecone_module is not None:
        monkeypatch.setitem(sys.modules, "pinecone", pinecone_module)

    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"{module_name}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def rag_module(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    pinecone_stub = types.ModuleType("pinecone")

    class _Pinecone:
        index_calls = []

        def __init__(self, api_key):
            self.api_key = api_key

        def Index(self, name):
            type(self).index_calls.append(name)
            return self

        def query(self, **kwargs):
            self.last_query = kwargs
            return types.SimpleNamespace(matches=[])

    pinecone_stub.Pinecone = _Pinecone

    bedrock = FakeBedrock(
        embed_payload=[0.1, 0.2, 0.3],
        llm_payload={"content": [{"text": "answer"}]},
    )
    module = load_script(
        monkeypatch,
        "rag_local_test",
        "scripts/05_test_rag_local.py",
        bedrock=bedrock,
        pinecone_module=pinecone_stub,
    )
    module._test_bedrock = bedrock
    module._test_pinecone = pinecone_stub.Pinecone
    return module


def test_embed_query_sends_expected_request_and_returns_embedding(rag_module):
    embedding = rag_module.embed_query("How do I create an S3 bucket?")

    assert embedding == [0.1, 0.2, 0.3]
    call = rag_module._test_bedrock.calls[0]
    assert call["modelId"] == rag_module.EMBEDDING_MODEL_ID
    assert call["contentType"] == "application/json"
    assert call["accept"] == "application/json"
    body = json.loads(call["body"])
    assert body == {"inputText": "How do I create an S3 bucket?", "dimensions": 1024, "normalize": True}


def test_search_pinecone_returns_normalized_chunk_dicts(monkeypatch, rag_module):
    class FakeMatch:
        def __init__(self, score, metadata):
            self.score = score
            self.metadata = metadata

    class FakeIndex:
        def __init__(self):
            self.calls = []

        def query(self, **kwargs):
            self.calls.append(kwargs)
            return types.SimpleNamespace(
                matches=[
                    FakeMatch(0.9876, {"content": "chunk", "service": "s3", "source_url": "https://a"}),
                    FakeMatch(0.5, {"service": "lambda"}),
                ]
            )

    class FakePinecone:
        def __init__(self, api_key):
            self.api_key = api_key

        def Index(self, name):
            self.index_name = name
            return FakeIndex()

    monkeypatch.setattr(rag_module, "Pinecone", FakePinecone)

    chunks = rag_module.search_pinecone([1.0, 2.0, 3.0])

    assert chunks == [
        {
            "score": 0.9876,
            "content": "chunk",
            "service": "s3",
            "source_url": "https://a",
        },
        {
            "score": 0.5,
            "content": "",
            "service": "lambda",
            "source_url": "",
        },
    ]


def test_build_prompt_includes_retrieved_context_and_question(rag_module):
    prompt = rag_module.build_prompt(
        "How do I create an S3 bucket?",
        [
            {
                "score": 0.99,
                "content": "S3 bucket docs",
                "service": "s3",
                "source_url": "https://example.com/s3",
            }
        ],
    )

    assert "S3 BUCKET DOCS" not in prompt
    assert "[Source 1: S3 — https://example.com/s3]" in prompt
    assert "S3 bucket docs" in prompt
    assert "How do I create an S3 bucket?" in prompt


def test_main_with_valid_question_prints_answer_and_sources(monkeypatch, capsys, rag_module):
    monkeypatch.setattr(rag_module, "embed_query", lambda question: [1.0, 2.0])
    monkeypatch.setattr(
        rag_module,
        "search_pinecone",
        lambda vec: [
            {"score": 0.9, "content": "chunk 1", "service": "s3", "source_url": "https://one"},
            {"score": 0.8, "content": "chunk 2", "service": "lambda", "source_url": "https://two"},
        ],
    )
    monkeypatch.setattr(rag_module, "build_prompt", lambda question, chunks: "PROMPT")
    monkeypatch.setattr(rag_module, "call_claude", lambda prompt: "final answer")
    monkeypatch.setattr(sys, "argv", ["05_test_rag_local.py", "What is S3?"])

    rag_module.main()

    out = capsys.readouterr().out
    assert "Question: What is S3?" in out
    assert "[OK] Generated 1024-dim vector" in out
    assert "final answer" in out
    assert "https://one" in out
    assert "https://two" in out


def test_main_with_no_relevant_docs_still_prints_answer_and_no_sources(monkeypatch, capsys, rag_module):
    monkeypatch.setattr(rag_module, "embed_query", lambda question: [1.0, 2.0])
    monkeypatch.setattr(rag_module, "search_pinecone", lambda vec: [])
    monkeypatch.setattr(rag_module, "build_prompt", lambda question, chunks: "PROMPT")
    monkeypatch.setattr(rag_module, "call_claude", lambda prompt: "answer with no sources")
    monkeypatch.setattr(sys, "argv", ["05_test_rag_local.py", "No docs?"])

    rag_module.main()

    out = capsys.readouterr().out
    assert "Searching Pinecone (top 5 results)" in out
    assert "Sources:" in out
    assert "answer with no sources" in out
    assert "    1." not in out


def test_main_reports_step_failure_cleanly(monkeypatch, capsys, rag_module):
    monkeypatch.setattr(rag_module, "embed_query", lambda question: (_ for _ in ()).throw(RuntimeError("embed failed")))
    monkeypatch.setattr(sys, "argv", ["05_test_rag_local.py", "fail please"])

    with pytest.raises(SystemExit) as excinfo:
        rag_module.main()

    out = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "[ERR]" in out
    assert "embed failed" in out
    assert "Traceback" not in out


def test_main_requires_question_argument(monkeypatch, capsys, rag_module):
    monkeypatch.setattr(sys, "argv", ["05_test_rag_local.py"])

    with pytest.raises(SystemExit) as excinfo:
        rag_module.main()

    out = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "Usage: python scripts/05_test_rag_local.py" in out

