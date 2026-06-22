import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGIN = "https://d3d0zch3u8ca61.cloudfront.net"


class FakeBody:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return self.payload.encode("utf-8")


class FakeBedrock:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"body": FakeBody(self.payload)}


class FakeIndex:
    def __init__(self, matches=None, error=None):
        self.matches = matches or []
        self.error = error
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(matches=self.matches)


def load_lambda_module(monkeypatch, *, allowed_origin="*", origin_verify_secret="origin-secret", origin_verify_header="x-origin-verify"):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("ORIGIN_VERIFY_SECRET", origin_verify_secret)
    monkeypatch.setenv("ORIGIN_VERIFY_HEADER", origin_verify_header)
    if allowed_origin is None:
        monkeypatch.delenv("ALLOWED_ORIGIN", raising=False)
    else:
        monkeypatch.setenv("ALLOWED_ORIGIN", allowed_origin)

    boto3_stub = types.ModuleType("boto3")
    boto3_stub.client = lambda *_args, **_kwargs: FakeBedrock()
    monkeypatch.setitem(sys.modules, "boto3", boto3_stub)

    pinecone_stub = types.ModuleType("pinecone")

    class _Pinecone:
        def __init__(self, api_key):
            self.api_key = api_key

        def Index(self, name):
            return FakeIndex()

    pinecone_stub.Pinecone = _Pinecone
    monkeypatch.setitem(sys.modules, "pinecone", pinecone_stub)

    module_name = f"lambda_handler_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "lambda_function" / "lambda_handler.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_match(score, content="", service="", source_url=""):
    return SimpleNamespace(score=score, metadata={"content": content, "service": service, "source_url": source_url})


def make_handler_stubs(monkeypatch, module, *, answer="final answer", chunks=None, prompt="prompt"):
    chunks = chunks if chunks is not None else [
        {
            "score": 0.9,
            "content": "chunk text",
            "service": "s3",
            "source_url": "https://example.com",
        }
    ]

    monkeypatch.setattr(module, "embed_query", lambda question: [0.1, 0.2, 0.3])
    monkeypatch.setattr(module, "search_pinecone", lambda vector: chunks)
    monkeypatch.setattr(module, "build_prompt", lambda question, chunk_list: prompt)
    monkeypatch.setattr(module, "call_claude", lambda generated_prompt: answer)


def make_event(body, *, header_name="x-origin-verify", header_value="origin-secret"):
    return {
        "body": body,
        "headers": {
            header_name: header_value,
        },
    }


def test_embed_query_returns_1024_dim_embedding_and_passes_body(monkeypatch):
    module = load_lambda_module(monkeypatch)
    fake_bedrock = FakeBedrock(payload=json.dumps({"embedding": [0.0] * 1024}))
    module.bedrock = fake_bedrock

    embedding = module.embed_query("How do I use AWS?")

    assert embedding == [0.0] * 1024
    assert len(embedding) == 1024
    request = json.loads(fake_bedrock.calls[0]["body"])
    assert request == {"inputText": "How do I use AWS?", "dimensions": 1024, "normalize": True}
    assert fake_bedrock.calls[0]["modelId"] == module.EMBEDDING_MODEL_ID
    assert fake_bedrock.calls[0]["contentType"] == "application/json"
    assert fake_bedrock.calls[0]["accept"] == "application/json"


def test_embed_query_passes_empty_query(monkeypatch):
    module = load_lambda_module(monkeypatch)
    fake_bedrock = FakeBedrock(payload=json.dumps({"embedding": [0.0] * 1024}))
    module.bedrock = fake_bedrock

    module.embed_query("")

    request = json.loads(fake_bedrock.calls[0]["body"])
    assert request["inputText"] == ""


def test_embed_query_surfaces_bedrock_error(monkeypatch):
    module = load_lambda_module(monkeypatch)
    module.bedrock = FakeBedrock(error=RuntimeError("bedrock boom"))

    with pytest.raises(RuntimeError, match="bedrock boom"):
        module.embed_query("question")


def test_embed_query_raises_on_malformed_json(monkeypatch):
    module = load_lambda_module(monkeypatch)
    module.bedrock = FakeBedrock(payload="not-json")

    with pytest.raises(json.JSONDecodeError):
        module.embed_query("question")


def test_embed_query_raises_on_wrong_embedding_dimension(monkeypatch):
    module = load_lambda_module(monkeypatch)
    module.bedrock = FakeBedrock(payload=json.dumps({"embedding": [0.0, 0.1, 0.2]}))

    with pytest.raises(ValueError, match="Expected embedding dimension 1024"):
        module.embed_query("question")


def test_search_pinecone_returns_expected_structure_for_five_matches(monkeypatch):
    module = load_lambda_module(monkeypatch)
    matches = [
        make_match(0.123456, "chunk 1", "s3", "https://example.com/1"),
        make_match(0.2, "chunk 2", "ec2", "https://example.com/2"),
        make_match(0.3, "chunk 3", "iam", "https://example.com/3"),
        make_match(0.4, "chunk 4", "sns", "https://example.com/4"),
        make_match(0.5, "chunk 5", "sqs", "https://example.com/5"),
    ]
    fake_index = FakeIndex(matches=matches)
    module.index = fake_index

    chunks = module.search_pinecone([1.0, 2.0, 3.0])

    assert fake_index.calls[0]["vector"] == [1.0, 2.0, 3.0]
    assert fake_index.calls[0]["top_k"] == 5
    assert fake_index.calls[0]["include_metadata"] is True
    assert chunks == [
        {"score": 0.1235, "content": "chunk 1", "service": "s3", "source_url": "https://example.com/1"},
        {"score": 0.2, "content": "chunk 2", "service": "ec2", "source_url": "https://example.com/2"},
        {"score": 0.3, "content": "chunk 3", "service": "iam", "source_url": "https://example.com/3"},
        {"score": 0.4, "content": "chunk 4", "service": "sns", "source_url": "https://example.com/4"},
        {"score": 0.5, "content": "chunk 5", "service": "sqs", "source_url": "https://example.com/5"},
    ]


def test_search_pinecone_returns_available_subset_for_fewer_matches(monkeypatch):
    module = load_lambda_module(monkeypatch)
    fake_index = FakeIndex(matches=[make_match(0.9, "only chunk", "s3", "https://example.com")])
    module.index = fake_index

    chunks = module.search_pinecone([9.0])

    assert len(chunks) == 1
    assert chunks[0]["content"] == "only chunk"


def test_search_pinecone_returns_empty_list_when_no_matches(monkeypatch):
    module = load_lambda_module(monkeypatch)
    module.index = FakeIndex(matches=[])

    assert module.search_pinecone([1.0]) == []


def test_search_pinecone_defaults_missing_metadata_to_empty_strings(monkeypatch):
    module = load_lambda_module(monkeypatch)
    module.index = FakeIndex(matches=[SimpleNamespace(score=0.75, metadata={})])

    chunks = module.search_pinecone([1.0])

    assert chunks == [{"score": 0.75, "content": "", "service": "", "source_url": ""}]


def test_search_pinecone_surfaces_query_error(monkeypatch):
    module = load_lambda_module(monkeypatch)
    module.index = FakeIndex(error=RuntimeError("pinecone boom"))

    with pytest.raises(RuntimeError, match="pinecone boom"):
        module.search_pinecone([1.0])


def test_build_prompt_formats_single_chunk(monkeypatch):
    module = load_lambda_module(monkeypatch)

    prompt = module.build_prompt("What is S3?", [{"content": "chunk text", "service": "s3", "source_url": "https://example.com"}])

    assert "[Source 1: S3" in prompt
    assert "chunk text" in prompt
    assert "<question>\nWhat is S3?\n</question>" in prompt


def test_build_prompt_formats_multiple_chunks_and_separators(monkeypatch):
    module = load_lambda_module(monkeypatch)

    prompt = module.build_prompt(
        "Question?",
        [
            {"content": "chunk one", "service": "s3", "source_url": "https://example.com/1"},
            {"content": "chunk two", "service": "ec2", "source_url": "https://example.com/2"},
        ],
    )

    assert prompt.count("\n\n---\n\n") == 1
    assert "[Source 1: S3" in prompt
    assert "[Source 2: EC2" in prompt
    assert "https://example.com/1" in prompt
    assert "https://example.com/2" in prompt


def test_build_prompt_handles_empty_chunks(monkeypatch):
    module = load_lambda_module(monkeypatch)

    prompt = module.build_prompt("Question?", [])

    assert "<documentation>\n\n</documentation>" in prompt
    assert "<question>\nQuestion?\n</question>" in prompt


def test_build_prompt_preserves_special_characters_and_newlines(monkeypatch):
    module = load_lambda_module(monkeypatch)

    prompt = module.build_prompt(
        "Line 1\nLine 2 \"quoted\"",
        [
            {
                "content": "first line\nsecond line with {braces} & symbols",
                "service": "s3",
                "source_url": "https://example.com/path?a=1&b=2",
            }
        ],
    )

    assert "Line 1\nLine 2 \"quoted\"" in prompt
    assert "first line\nsecond line with {braces} & symbols" in prompt
    assert "https://example.com/path?a=1&b=2" in prompt


def test_call_claude_returns_first_text_block(monkeypatch):
    module = load_lambda_module(monkeypatch)
    fake_bedrock = FakeBedrock(payload=json.dumps({"content": [{"text": "first"}, {"text": "second"}]}))
    module.bedrock = fake_bedrock

    text = module.call_claude("prompt text")

    assert text == "first"
    request = json.loads(fake_bedrock.calls[0]["body"])
    assert request["anthropic_version"] == "bedrock-2023-05-31"
    assert request["max_tokens"] == 2048
    assert request["messages"] == [{"role": "user", "content": "prompt text"}]


def test_call_claude_sends_empty_prompt(monkeypatch):
    module = load_lambda_module(monkeypatch)
    fake_bedrock = FakeBedrock(payload=json.dumps({"content": [{"text": "answer"}]}))
    module.bedrock = fake_bedrock

    module.call_claude("")

    request = json.loads(fake_bedrock.calls[0]["body"])
    assert request["messages"] == [{"role": "user", "content": ""}]


def test_call_claude_surfaces_bedrock_error(monkeypatch):
    module = load_lambda_module(monkeypatch)
    module.bedrock = FakeBedrock(error=RuntimeError("claude boom"))

    with pytest.raises(RuntimeError, match="claude boom"):
        module.call_claude("prompt")


def test_call_claude_raises_on_malformed_response(monkeypatch):
    module = load_lambda_module(monkeypatch)
    module.bedrock = FakeBedrock(payload=json.dumps({"content": []}))

    with pytest.raises(IndexError):
        module.call_claude("prompt")


@pytest.mark.parametrize("body", [json.dumps({"question": "How do I use AWS?"}), {"question": "How do I use AWS?"}])
def test_lambda_handler_happy_path_parses_body_and_returns_cors_headers(monkeypatch, body):
    module = load_lambda_module(monkeypatch, allowed_origin="https://example.cloudfront.net")
    make_handler_stubs(monkeypatch, module, answer="final answer")

    response = module.lambda_handler(make_event(body), None)
    payload = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert response["headers"]["Access-Control-Allow-Origin"] == "https://example.cloudfront.net"
    assert response["headers"]["Access-Control-Allow-Headers"] == "Content-Type"
    assert response["headers"]["Access-Control-Allow-Methods"] == "POST,OPTIONS"
    assert payload == {
        "answer": "final answer",
        "sources": [
            {"service": "s3", "url": "https://example.com", "score": 0.9},
        ],
        "question": "How do I use AWS?",
    }


def test_lambda_handler_accepts_max_length_question(monkeypatch):
    module = load_lambda_module(monkeypatch, allowed_origin="*")
    make_handler_stubs(monkeypatch, module, answer="final answer")
    question = "x" * 1000

    response = module.lambda_handler(make_event({"question": question}), None)
    payload = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert response["headers"]["Access-Control-Allow-Origin"] == "*"
    assert payload["question"] == question


@pytest.mark.parametrize(
    "failing_stage",
    ["embed_query", "search_pinecone", "build_prompt", "call_claude"],
)
def test_lambda_handler_returns_500_for_pipeline_stage_failures(monkeypatch, capsys, failing_stage):
    module = load_lambda_module(monkeypatch)
    make_handler_stubs(monkeypatch, module)

    def boom(*_args, **_kwargs):
        raise RuntimeError(f"{failing_stage} boom")

    monkeypatch.setattr(module, failing_stage, boom)

    response = module.lambda_handler(make_event(json.dumps({"question": "What is AWS?"})), None)
    payload = json.loads(response["body"])
    captured = capsys.readouterr()

    assert response["statusCode"] == 500
    assert payload == {"error": "Internal server error"}
    assert response["headers"]["Access-Control-Allow-Origin"] == "*"
    assert "Traceback" in captured.out


def test_lambda_handler_returns_generic_500_for_unexpected_exception(monkeypatch, capsys):
    module = load_lambda_module(monkeypatch)

    response = module.lambda_handler(None, None)
    payload = json.loads(response["body"])
    captured = capsys.readouterr()

    assert response["statusCode"] == 500
    assert payload == {"error": "Internal server error"}
    assert response["headers"]["Access-Control-Allow-Origin"] == "*"
    assert "Traceback" in captured.out


@pytest.mark.parametrize(
    "allowed_origin",
    ["https://example.cloudfront.net", "*", None],
)
def test_lambda_handler_uses_configured_cors_origin(monkeypatch, allowed_origin):
    module = load_lambda_module(monkeypatch, allowed_origin=allowed_origin)
    make_handler_stubs(monkeypatch, module)

    response = module.lambda_handler(make_event({"question": "Question?"}), None)

    expected_origin = allowed_origin if allowed_origin is not None else DEFAULT_ORIGIN
    assert response["headers"]["Access-Control-Allow-Origin"] == expected_origin


def test_lambda_handler_rejects_missing_origin_verification_header(monkeypatch):
    module = load_lambda_module(monkeypatch, allowed_origin="https://example.cloudfront.net")
    make_handler_stubs(monkeypatch, module)

    response = module.lambda_handler({"body": {"question": "Question?"}, "headers": {}}, None)

    assert response["statusCode"] == 403
    assert json.loads(response["body"]) == {"error": "Forbidden"}
    assert response["headers"]["Access-Control-Allow-Origin"] == "https://example.cloudfront.net"


def test_lambda_handler_accepts_case_insensitive_origin_header_name(monkeypatch):
    module = load_lambda_module(monkeypatch, origin_verify_secret="shared-secret", origin_verify_header="x-origin-verify")
    make_handler_stubs(monkeypatch, module)

    response = module.lambda_handler(
        {
            "body": {"question": "Question?"},
            "headers": {"X-Origin-Verify": "shared-secret"},
        },
        None,
    )

    assert response["statusCode"] == 200
