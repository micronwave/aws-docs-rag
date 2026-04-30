import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(module_name: str, relative_path: str, stub_pinecone: bool = False):
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
        sys.modules["pinecone"] = pinecone_stub

    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def ingest_module(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    return load_script("ingest_docs_test", "scripts/01_ingest_docs.py")


@pytest.fixture()
def upload_module(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    return load_script("upload_to_pinecone_test", "scripts/04_upload_to_pinecone.py", stub_pinecone=True)


@pytest.fixture()
def embed_module(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    return load_script("generate_embeddings_test", "scripts/03_generate_embeddings.py")


def test_fragment_links_are_normalized(ingest_module):
    html = """
    <html><body>
      <a href="guide/other.html#section-1">Doc link</a>
      <a href="#local-fragment">Local fragment</a>
      <a href="https://docs.aws.amazon.com/AmazonS3/latest/userguide/topics.html#details">Absolute link</a>
      <a href="https://example.com/ignore.html">External</a>
    </body></html>
    """

    links = ingest_module.extract_links(
        html,
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "/AmazonS3/latest/userguide/",
    )

    assert links == [
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/guide/other.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/topics.html",
    ]


def test_embeddings_manifest_prefix_mismatch_fails(upload_module, monkeypatch):
    class FakeBody:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return self._payload.encode("utf-8")

    class FakePaginator:
        def paginate(self, Bucket, Prefix):
            return [{"Contents": [{"Key": f"{Prefix}batch_0000.json"}]}]

    class FakeS3:
        def get_object(self, Bucket, Key):
            if Key == "embeddings/manifest.json":
                manifest = {
                    "run_id": "run-123",
                    "status": "success",
                    "documents_prefix": "embeddings/other-run/",
                }
                return {"Body": FakeBody(json.dumps(manifest))}
            return {"Body": FakeBody("[]")}

        def get_paginator(self, name):
            return FakePaginator()

    monkeypatch.setattr(upload_module, "s3_client", FakeS3())

    with pytest.raises(SystemExit, match="points to"):
        upload_module.load_embeddings_from_s3()


def test_validate_and_verify_catches_conflicts_and_stale_generation(upload_module):
    record = {
        "chunk_id": "chunk_000001",
        "embedding": [0.1] * upload_module.EMBEDDING_DIMENSION,
        "service": "s3",
        "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "chunk_index": 0,
        "content": "hello",
        "generation_id": "run-123",
    }
    deduped, expected_ids = upload_module.validate_and_deduplicate_embeddings([record, dict(record)])
    assert deduped == [record]
    assert expected_ids == ["chunk_000001"]

    class FakeIndex:
        def describe_index_stats(self):
            return {"total_vector_count": 1, "dimension": upload_module.EMBEDDING_DIMENSION}

        def fetch(self, ids):
            return {
                "vectors": {
                    ids[0]: {
                        "metadata": {"generation_id": "stale-run"},
                    }
                }
            }

    errors = upload_module.verify_index_state(FakeIndex(), expected_ids, "run-123")
    assert any("stale generation_id" in error for error in errors)

    conflicting = dict(record)
    conflicting["content"] = "different"
    with pytest.raises(SystemExit, match="Conflicting duplicate chunk_id"):
        upload_module.validate_and_deduplicate_embeddings([record, conflicting])


def test_process_chunks_reports_partial_failures(embed_module, monkeypatch):
    monkeypatch.setattr(embed_module, "embed_text", lambda text: [0.0] * embed_module.EMBEDDING_DIMENSION)
    monkeypatch.setattr(embed_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(embed_module.os, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: io.StringIO())

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
    ]

    embedded, failed = embed_module.process_chunks(chunks, "run-123")
    assert len(embedded) == 1
    assert failed == ["chunk_000002"]
