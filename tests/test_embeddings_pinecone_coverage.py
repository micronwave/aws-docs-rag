import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(monkeypatch, module_name: str, relative_path: str, stub_pinecone: bool = False):
    if stub_pinecone:
        pinecone_stub = types.ModuleType("pinecone")

        class _Pinecone:
            pass

        class _ServerlessSpec:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        pinecone_stub.Pinecone = _Pinecone
        pinecone_stub.ServerlessSpec = _ServerlessSpec
        monkeypatch.setitem(sys.modules, "pinecone", pinecone_stub)

    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NonClosingStringIO(io.StringIO):
    def close(self):
        pass


@pytest.fixture()
def embed_module(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    return load_script(monkeypatch, "generate_embeddings_coverage_test", "scripts/03_generate_embeddings.py")


@pytest.fixture()
def upload_module(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    return load_script(monkeypatch, "upload_to_pinecone_coverage_test", "scripts/04_upload_to_pinecone.py", stub_pinecone=True)


def make_embedding(module):
    return [0.0] * module.EMBEDDING_DIMENSION


@pytest.mark.parametrize("text", ["hello world", ""])
def test_embed_text_returns_1024_dim_embedding_for_text_and_empty_text(embed_module, monkeypatch, text):
    captured = {}

    class FakeBody:
        def read(self):
            return json.dumps({"embedding": make_embedding(embed_module)}).encode("utf-8")

    class FakeBedrock:
        def invoke_model(self, **kwargs):
            captured.update(kwargs)
            return {"body": FakeBody()}

    monkeypatch.setattr(embed_module, "bedrock", FakeBedrock())

    embedding = embed_module.embed_text(text)

    assert len(embedding) == embed_module.EMBEDDING_DIMENSION
    assert captured["modelId"] == embed_module.EMBEDDING_MODEL_ID
    assert json.loads(captured["body"]) == {
        "inputText": text,
        "dimensions": embed_module.EMBEDDING_DIMENSION,
        "normalize": True,
    }


def test_embed_text_surfaces_bedrock_error(embed_module, monkeypatch):
    class FakeBedrock:
        def invoke_model(self, **_kwargs):
            raise RuntimeError("bedrock unavailable")

    monkeypatch.setattr(embed_module, "bedrock", FakeBedrock())

    with pytest.raises(RuntimeError, match="bedrock unavailable"):
        embed_module.embed_text("hello")


def test_embed_text_rejects_wrong_embedding_dimension(embed_module, monkeypatch):
    class FakeBody:
        def read(self):
            return json.dumps({"embedding": [0.0] * (embed_module.EMBEDDING_DIMENSION - 1)}).encode("utf-8")

    class FakeBedrock:
        def invoke_model(self, **_kwargs):
            return {"body": FakeBody()}

    monkeypatch.setattr(embed_module, "bedrock", FakeBedrock())

    with pytest.raises(ValueError, match="Expected embedding dimension"):
        embed_module.embed_text("hello")


def test_validate_chunk_record_accepts_valid_chunk(embed_module):
    chunk = {
        "chunk_id": "chunk_000001",
        "service": "s3",
        "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "chunk_index": 0,
        "content": "hello",
    }

    assert embed_module.validate_chunk_record(chunk) is None


def test_validate_chunk_record_rejects_missing_required_field(embed_module):
    chunk = {
        "chunk_id": "chunk_000001",
        "service": "s3",
        "chunk_index": 0,
        "content": "hello",
    }

    with pytest.raises(ValueError, match="Missing required chunk fields: source_url"):
        embed_module.validate_chunk_record(chunk)


def test_validate_chunk_record_rejects_empty_content(embed_module):
    chunk = {
        "chunk_id": "chunk_000001",
        "service": "s3",
        "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "chunk_index": 0,
        "content": "   ",
    }

    with pytest.raises(ValueError, match="has empty content"):
        embed_module.validate_chunk_record(chunk)


def test_process_chunks_sets_generation_id_for_successes(embed_module, monkeypatch):
    monkeypatch.setattr(embed_module, "embed_text", lambda text: [0.0] * embed_module.EMBEDDING_DIMENSION)
    sleep_calls = []
    monkeypatch.setattr(embed_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    chunks = [
        {
            "chunk_id": "chunk_000001",
            "service": "s3",
            "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "chunk_index": 0,
            "content": "hello",
        },
        {
            "chunk_id": "chunk_000002",
            "service": "s3",
            "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "chunk_index": 1,
            "content": "world",
        },
    ]

    embedded, failed = embed_module.process_chunks(chunks, "gen-123")

    assert failed == []
    assert len(embedded) == 2
    assert all(record["generation_id"] == "gen-123" for record in embedded)
    assert all(len(record["embedding"]) == embed_module.EMBEDDING_DIMENSION for record in embedded)
    assert sleep_calls == [embed_module.DELAY, embed_module.DELAY]


def test_process_chunks_continues_after_missing_content_and_throttling(embed_module, monkeypatch):
    def fake_embed_text(text):
        if text == "throttle":
            raise Exception("ThrottlingException: rate exceeded")
        return [0.0] * embed_module.EMBEDDING_DIMENSION

    monkeypatch.setattr(embed_module, "embed_text", fake_embed_text)
    sleep_calls = []
    monkeypatch.setattr(embed_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(embed_module.os, "makedirs", lambda *_args, **_kwargs: None)
    failed_file = NonClosingStringIO()
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: failed_file)

    chunks = [
        {
            "chunk_id": "chunk_000001",
            "service": "s3",
            "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "chunk_index": 0,
            "content": "hello",
        },
        {
            "chunk_id": "chunk_000002",
            "service": "s3",
            "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "chunk_index": 1,
        },
        {
            "chunk_id": "chunk_000003",
            "service": "s3",
            "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "chunk_index": 2,
            "content": "throttle",
        },
        {
            "chunk_id": "chunk_000004",
            "service": "s3",
            "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "chunk_index": 3,
            "content": "still ok",
        },
    ]

    embedded, failed = embed_module.process_chunks(chunks, "gen-123")

    assert [record["chunk_id"] for record in embedded] == ["chunk_000001", "chunk_000004"]
    assert failed == ["chunk_000002", "chunk_000003"]
    assert 10 in sleep_calls
    assert json.loads(failed_file.getvalue()) == failed


def test_embedding_main_writes_failure_manifest_for_empty_corpus(embed_module, monkeypatch):
    captured = {}
    monkeypatch.setattr(embed_module, "make_run_id", lambda: "gen-123")
    monkeypatch.setattr(
        embed_module,
        "load_chunks_from_s3",
        lambda: ([], {"run_id": "chunk-run-1", "documents_prefix": "chunks/chunk-run-1/"}),
    )
    monkeypatch.setattr(embed_module, "write_embeddings_manifest", lambda manifest: captured.update(manifest))

    with pytest.raises(SystemExit, match="1"):
        embed_module.main()

    assert captured["status"] == "failed"
    assert captured["reason"] == "empty_corpus"
    assert captured["source_run_id"] == "chunk-run-1"
    assert captured["source_documents_prefix"] == "chunks/chunk-run-1/"


def test_embedding_main_calls_upload_on_success(embed_module, monkeypatch):
    uploaded = {}
    monkeypatch.setattr(embed_module, "make_run_id", lambda: "gen-123")
    monkeypatch.setattr(
        embed_module,
        "load_chunks_from_s3",
        lambda: (
            [
                {
                    "chunk_id": "chunk_000001",
                    "service": "s3",
                    "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
                    "chunk_index": 0,
                    "content": "hello",
                }
            ],
            {"run_id": "chunk-run-1", "documents_prefix": "chunks/chunk-run-1/"},
        ),
    )
    monkeypatch.setattr(
        embed_module,
        "process_chunks",
        lambda chunks, generation_id: (
            [
                {
                    **chunks[0],
                    "generation_id": generation_id,
                    "embedding": [0.0] * embed_module.EMBEDDING_DIMENSION,
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(embed_module, "upload_embeddings_to_s3", lambda embedded, run_id, source_manifest: uploaded.update({
        "embedded": embedded,
        "run_id": run_id,
        "source_manifest": source_manifest,
    }))
    monkeypatch.setattr(embed_module.os, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: io.StringIO())
    monkeypatch.setattr(embed_module.os.path, "getsize", lambda *_args, **_kwargs: 0)

    embed_module.main()

    assert uploaded["run_id"] == "gen-123"
    assert uploaded["source_manifest"]["run_id"] == "chunk-run-1"
    assert uploaded["embedded"][0]["generation_id"] == "gen-123"


def test_upload_embeddings_to_s3_batches_and_manifest_references_source(embed_module, monkeypatch):
    calls = []
    manifest = {}

    class FakeS3:
        def put_object(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(embed_module, "s3_client", FakeS3())
    monkeypatch.setattr(embed_module, "write_embeddings_manifest", lambda payload: manifest.update(payload))

    embedded = []
    for i in range(51):
        embedded.append(
            {
                "chunk_id": f"chunk_{i:06d}",
                "service": "s3",
                "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
                "chunk_index": i,
                "content": f"content-{i}",
                "generation_id": "gen-123",
                "embedding": make_embedding(embed_module),
            }
        )

    source_manifest = {
        "run_id": "chunk-run-1",
        "documents_prefix": "chunks/chunk-run-1/",
    }

    embed_module.upload_embeddings_to_s3(embedded, "gen-123", source_manifest)

    assert [call["Key"] for call in calls] == [
        "embeddings/gen-123/batch_0000.json",
        "embeddings/gen-123/batch_0001.json",
    ]
    assert manifest["run_id"] == "gen-123"
    assert manifest["generation_id"] == "gen-123"
    assert manifest["documents_prefix"] == "embeddings/gen-123/"
    assert manifest["source_run_id"] == "chunk-run-1"
    assert manifest["source_documents_prefix"] == "chunks/chunk-run-1/"


def test_create_index_creates_missing_index_with_expected_shape(upload_module):
    recorded = {}

    class FakePC:
        def list_indexes(self):
            return []

        def create_index(self, **kwargs):
            recorded.update(kwargs)

        def describe_index(self, _name):
            return {"status": {"ready": True}}

    upload_module.create_index(FakePC())

    assert recorded["name"] == upload_module.INDEX_NAME
    assert recorded["dimension"] == upload_module.EMBEDDING_DIMENSION
    assert recorded["metric"] == upload_module.EXPECTED_METRIC
    assert recorded["spec"].kwargs == {"cloud": "aws", "region": "us-east-1"}


def test_create_index_reuses_matching_existing_index(upload_module):
    class FakePC:
        def __init__(self):
            self.created = False

        def list_indexes(self):
            return [types.SimpleNamespace(name=upload_module.INDEX_NAME)]

        def describe_index(self, _name):
            return {"dimension": upload_module.EMBEDDING_DIMENSION, "metric": upload_module.EXPECTED_METRIC}

        def create_index(self, **_kwargs):
            self.created = True

    pc = FakePC()
    upload_module.create_index(pc)

    assert pc.created is False


def test_create_index_rejects_dimension_mismatch(upload_module):
    class FakePC:
        def list_indexes(self):
            return [types.SimpleNamespace(name=upload_module.INDEX_NAME)]

        def describe_index(self, _name):
            return {"dimension": upload_module.EMBEDDING_DIMENSION - 1, "metric": upload_module.EXPECTED_METRIC}

    with pytest.raises(SystemExit, match="expected dimension"):
        upload_module.create_index(FakePC())


def test_create_index_rejects_metric_mismatch(upload_module):
    class FakePC:
        def list_indexes(self):
            return [types.SimpleNamespace(name=upload_module.INDEX_NAME)]

        def describe_index(self, _name):
            return {"dimension": upload_module.EMBEDDING_DIMENSION, "metric": "dotproduct"}

    with pytest.raises(SystemExit, match="expected dimension"):
        upload_module.create_index(FakePC())


def test_validate_and_deduplicate_embeddings_handles_unique_and_empty_lists(upload_module):
    record = {
        "chunk_id": "chunk_000001",
        "embedding": [0.1] * upload_module.EMBEDDING_DIMENSION,
        "service": "s3",
        "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "chunk_index": 0,
        "content": "hello",
        "generation_id": "gen-123",
    }

    deduped, expected_ids = upload_module.validate_and_deduplicate_embeddings([record, dict(record, chunk_id="chunk_000002")])
    assert deduped == [record, dict(record, chunk_id="chunk_000002")]
    assert expected_ids == ["chunk_000001", "chunk_000002"]

    deduped_empty, expected_ids_empty = upload_module.validate_and_deduplicate_embeddings([])
    assert deduped_empty == []
    assert expected_ids_empty == []


def test_verify_index_state_handles_empty_index_and_matching_generation(upload_module):
    class FakeIndex:
        def __init__(self, vectors):
            self.vectors = vectors

        def describe_index_stats(self):
            return {"total_vector_count": len(self.vectors), "dimension": upload_module.EMBEDDING_DIMENSION}

        def fetch(self, ids):
            return {"vectors": {chunk_id: self.vectors[chunk_id] for chunk_id in ids if chunk_id in self.vectors}}

    assert upload_module.verify_index_state(FakeIndex({}), [], "gen-123") == []

    matching_vectors = {
        "chunk_000001": {"metadata": {"generation_id": "gen-123"}},
        "chunk_000002": {"metadata": {"generation_id": "gen-123"}},
    }
    assert upload_module.verify_index_state(FakeIndex(matching_vectors), ["chunk_000001", "chunk_000002"], "gen-123") == []


def test_verify_index_state_reports_stale_generation_ids(upload_module):
    class FakeIndex:
        def describe_index_stats(self):
            return {"total_vector_count": 1, "dimension": upload_module.EMBEDDING_DIMENSION}

        def fetch(self, ids):
            return {
                "vectors": {
                    ids[0]: {
                        "metadata": {"generation_id": "stale-gen"},
                    }
                }
            }

    errors = upload_module.verify_index_state(FakeIndex(), ["chunk_000001"], "gen-123")

    assert any("stale generation_id" in error for error in errors)


def test_verify_index_state_reports_dimension_mismatch(upload_module):
    class FakeIndex:
        def describe_index_stats(self):
            return {"total_vector_count": 1, "dimension": upload_module.EMBEDDING_DIMENSION - 1}

        def fetch(self, ids):
            return {
                "vectors": {
                    ids[0]: {
                        "metadata": {"generation_id": "gen-123"},
                    }
                }
            }

    errors = upload_module.verify_index_state(FakeIndex(), ["chunk_000001"], "gen-123")

    assert any("index dimension is" in error for error in errors)


def test_retry_upsert_succeeds_first_try(upload_module, monkeypatch):
    attempts = []

    class FakeIndex:
        def upsert(self, vectors):
            attempts.append(list(vectors))

    monkeypatch.setattr(upload_module.time, "sleep", lambda *_args, **_kwargs: None)

    ok, error = upload_module.retry_upsert(FakeIndex(), [{"id": "chunk_000001"}], 0)

    assert ok is True
    assert error is None
    assert attempts == [[{"id": "chunk_000001"}]]


def test_retry_upsert_retries_then_succeeds(upload_module, monkeypatch):
    attempts = []
    sleeps = []

    class FakeIndex:
        def __init__(self):
            self.calls = 0

        def upsert(self, vectors):
            attempts.append(list(vectors))
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")

    monkeypatch.setattr(upload_module, "UPSERT_MAX_RETRIES", 3)
    monkeypatch.setattr(upload_module, "UPSERT_BACKOFF_SECONDS", 0.25)
    monkeypatch.setattr(upload_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    ok, error = upload_module.retry_upsert(FakeIndex(), [{"id": "chunk_000001"}], 2)

    assert ok is True
    assert error is None
    assert sleeps == [0.25]
    assert len(attempts) == 2


def test_retry_upsert_returns_failure_after_retries_exhausted(upload_module, monkeypatch):
    sleeps = []

    class FakeIndex:
        def upsert(self, vectors):
            raise RuntimeError("still failing")

    monkeypatch.setattr(upload_module, "UPSERT_MAX_RETRIES", 2)
    monkeypatch.setattr(upload_module, "UPSERT_BACKOFF_SECONDS", 0.5)
    monkeypatch.setattr(upload_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    ok, error = upload_module.retry_upsert(FakeIndex(), [{"id": "chunk_000001"}], 4)

    assert ok is False
    assert error == "still failing"
    assert sleeps == [0.5]


def test_upload_to_pinecone_happy_path_succeeds(upload_module, monkeypatch):
    uploaded_batches = []

    class FakeIndex:
        def describe_index_stats(self):
            return {"total_vector_count": 2, "dimension": upload_module.EMBEDDING_DIMENSION}

        def fetch(self, ids):
            return {
                "vectors": {
                    chunk_id: {"metadata": {"generation_id": "gen-123"}}
                    for chunk_id in ids
                }
            }

        def upsert(self, vectors):
            uploaded_batches.append(vectors)

    class FakePC:
        def Index(self, _name):
            return FakeIndex()

    monkeypatch.setattr(upload_module, "verify_index_state", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(upload_module, "retry_upsert", lambda index, vectors, batch_num: (index.upsert(vectors) or True, None))
    monkeypatch.setattr(upload_module.time, "sleep", lambda *_args, **_kwargs: None)

    chunks = [
        {
            "chunk_id": "chunk_000001",
            "embedding": [0.0] * upload_module.EMBEDDING_DIMENSION,
            "service": "s3",
            "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "chunk_index": 0,
            "content": "hello",
            "generation_id": "gen-123",
        },
        {
            "chunk_id": "chunk_000002",
            "embedding": [0.0] * upload_module.EMBEDDING_DIMENSION,
            "service": "s3",
            "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "chunk_index": 1,
            "content": "world",
            "generation_id": "gen-123",
        },
    ]

    upload_module.upload_to_pinecone(FakePC(), chunks, {"generation_id": "gen-123"})

    assert len(uploaded_batches) == 1
    assert [vector["id"] for vector in uploaded_batches[0]] == ["chunk_000001", "chunk_000002"]


def test_upload_to_pinecone_dedup_conflict_aborts_before_upsert(upload_module, monkeypatch):
    upserted = []

    class FakeIndex:
        def describe_index_stats(self):
            return {"total_vector_count": 0, "dimension": upload_module.EMBEDDING_DIMENSION}

        def fetch(self, _ids):
            return {"vectors": {}}

        def upsert(self, vectors):
            upserted.append(vectors)

    class FakePC:
        def Index(self, _name):
            return FakeIndex()

    record = {
        "chunk_id": "chunk_000001",
        "embedding": [0.0] * upload_module.EMBEDDING_DIMENSION,
        "service": "s3",
        "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "chunk_index": 0,
        "content": "hello",
        "generation_id": "gen-123",
    }
    conflicting = dict(record, content="different")

    with pytest.raises(SystemExit, match="Conflicting duplicate chunk_id"):
        upload_module.upload_to_pinecone(FakePC(), [record, conflicting], {"generation_id": "gen-123"})

    assert upserted == []


def test_upload_to_pinecone_propagates_upsert_failure_through_failure_path(upload_module, monkeypatch):
    failure_artifact = {}

    class FakeIndex:
        def describe_index_stats(self):
            return {"total_vector_count": 1, "dimension": upload_module.EMBEDDING_DIMENSION}

        def fetch(self, ids):
            return {"vectors": {ids[0]: {"metadata": {"generation_id": "gen-123"}}}}

        def upsert(self, vectors):
            raise RuntimeError("upsert boom")

    class FakePC:
        def Index(self, _name):
            return FakeIndex()

    monkeypatch.setattr(upload_module, "UPSERT_MAX_RETRIES", 1)
    monkeypatch.setattr(upload_module, "verify_index_state", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(upload_module, "persist_upload_failure", lambda run_id, failed_batches, verification_errors: failure_artifact.update({
        "run_id": run_id,
        "failed_batches": failed_batches,
        "verification_errors": verification_errors,
    }))
    monkeypatch.setattr(upload_module.time, "sleep", lambda *_args, **_kwargs: None)

    chunk = {
        "chunk_id": "chunk_000001",
        "embedding": [0.0] * upload_module.EMBEDDING_DIMENSION,
        "service": "s3",
        "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "chunk_index": 0,
        "content": "hello",
        "generation_id": "gen-123",
    }

    with pytest.raises(SystemExit, match="1"):
        upload_module.upload_to_pinecone(FakePC(), [chunk], {"generation_id": "gen-123"})

    assert failure_artifact["run_id"] == "gen-123"
    assert failure_artifact["failed_batches"][0]["chunk_ids"] == ["chunk_000001"]
    assert failure_artifact["failed_batches"][0]["error"] == "upsert boom"
    assert failure_artifact["verification_errors"] == []
