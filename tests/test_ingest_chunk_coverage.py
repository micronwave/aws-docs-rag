import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]


def load_script(module_name: str, relative_path: str):
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
    return load_script("ingest_docs_coverage_test", "scripts/01_ingest_docs.py")


@pytest.fixture()
def chunk_module(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    splitter_stub = types.ModuleType("langchain_text_splitters")

    class _RecursiveCharacterTextSplitter:
        instances = []

        def __init__(self, *, chunk_size, chunk_overlap, length_function, separators):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function
            self.separators = separators
            type(self).instances.append(self)

        def split_text(self, text):
            if not text:
                return []
            if len(text) <= self.chunk_size:
                return [text]

            step = self.chunk_size - self.chunk_overlap
            pieces = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                pieces.append(text[start:end])
                if end >= len(text):
                    break
                start += step
            return pieces

    splitter_stub.RecursiveCharacterTextSplitter = _RecursiveCharacterTextSplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", splitter_stub)
    return load_script("chunk_docs_coverage_test", "scripts/02_chunk_docs.py")


class _FakeResponse:
    def __init__(self, text: str, raise_exc: Exception | None = None):
        self.text = text
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc


def _html_with_text(text: str, links: str = "") -> str:
    return f"""
    <html>
      <body>
        <div id="main-col-body">
          <h1>{text}</h1>
          <p>{text}</p>
          {links}
        </div>
      </body>
    </html>
    """


def test_get_page_returns_html_on_first_success(ingest_module, monkeypatch):
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, headers, timeout))
        return _FakeResponse("<html>ok</html>")

    sleep_calls = []
    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    html = ingest_module.get_page("https://example.com/page.html")

    assert html == "<html>ok</html>"
    assert calls == [
        (
            "https://example.com/page.html",
            {"User-Agent": "AWS-RAG-Project-Educational/1.0"},
            15,
        )
    ]
    assert sleep_calls == []


@pytest.mark.parametrize(
    "first_outcome",
    [
        requests.exceptions.Timeout("timeout"),
        requests.exceptions.ConnectionError("connection reset"),
        _FakeResponse("<html>ok</html>", raise_exc=requests.exceptions.HTTPError("http error")),
    ],
)
def test_get_page_retries_on_request_errors_then_succeeds(ingest_module, monkeypatch, first_outcome):
    outcomes = [first_outcome, _FakeResponse("<html>ok</html>")]
    sleep_calls = []

    def fake_get(url, headers, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    html = ingest_module.get_page("https://example.com/page.html")

    assert html == "<html>ok</html>"
    assert sleep_calls == [1]


def test_get_page_returns_none_after_all_attempts_fail(ingest_module, monkeypatch):
    calls = []
    sleep_calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, headers, timeout))
        raise requests.exceptions.Timeout("timeout")

    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    html = ingest_module.get_page("https://example.com/page.html", max_retries=3)

    assert html is None
    assert len(calls) == 3
    assert sleep_calls == [1, 2]


def test_clean_html_extracts_main_content_and_removes_chrome(ingest_module):
    html = """
    <html>
      <body>
        <div id="main-col-body">
          <header>Docs header</header>
          <nav>Navigation</nav>
          <style>.x { color: red; }</style>
          <script>alert('x')</script>
          <div class="awsdocs-page-header">Page chrome</div>
          <div class="feedback">Feedback</div>
          <h1>Title</h1>
          <p>Body copy</p>
          <footer>Footer</footer>
        </div>
      </body>
    </html>
    """

    text = ingest_module.clean_html(html)

    assert text == "Title\nBody copy"


def test_clean_html_falls_back_to_main_element(ingest_module):
    html = """
    <html>
      <body>
        <main>
          <p>Fallback text</p>
        </main>
      </body>
    </html>
    """

    assert ingest_module.clean_html(html) == "Fallback text"


def test_clean_html_returns_empty_text_for_minimal_document(ingest_module):
    assert ingest_module.clean_html("<html><body></body></html>") == ""


def test_scrape_service_follows_internal_links_and_ignores_external(ingest_module, monkeypatch):
    monkeypatch.setattr(ingest_module, "MIN_TEXT_CHARS", 20)
    monkeypatch.setattr(ingest_module, "REQUEST_DELAY", 0)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda *_args, **_kwargs: None)

    seed = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html"
    page1 = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/page1.html"
    page2 = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/page2.html"
    html_map = {
        seed: _html_with_text(
            "Seed page text",
            links=f'<a href="page1.html">Next</a><a href="https://example.com/outside.html">Outside</a>',
        ),
        page1: _html_with_text(
            "Page one text",
            links=f'<a href="page2.html">Next</a><a href="https://example.com/ignore.html">Ignore</a>',
        ),
        page2: _html_with_text("Page two text"),
    }

    def fake_get_page(url):
        assert url in html_map
        return html_map[url]

    monkeypatch.setattr(ingest_module, "get_page", fake_get_page)

    docs, report = ingest_module.scrape_service("s3", seed)

    assert [doc["url"] for doc in docs] == [seed, page1, page2]
    assert report["pages_attempted"] == 3
    assert report["pages_failed"] == 0
    assert report["failed_urls"] == []
    assert report["seed_failed"] is False
    assert report["skipped_pages"] == []


def test_scrape_service_records_failed_urls(ingest_module, monkeypatch):
    monkeypatch.setattr(ingest_module, "MIN_TEXT_CHARS", 20)
    monkeypatch.setattr(ingest_module, "REQUEST_DELAY", 0)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda *_args, **_kwargs: None)

    seed = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html"
    page1 = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/page1.html"
    html_map = {
        seed: _html_with_text("Seed page text", links='<a href="page1.html">Next</a>'),
        page1: None,
    }

    monkeypatch.setattr(ingest_module, "get_page", lambda url: html_map[url])

    docs, report = ingest_module.scrape_service("s3", seed)

    assert [doc["url"] for doc in docs] == [seed]
    assert report["pages_attempted"] == 2
    assert report["pages_failed"] == 1
    assert report["failed_urls"] == [page1]
    assert report["seed_failed"] is False


def test_scrape_service_skips_short_pages(ingest_module, monkeypatch):
    monkeypatch.setattr(ingest_module, "MIN_TEXT_CHARS", 50)
    monkeypatch.setattr(ingest_module, "REQUEST_DELAY", 0)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda *_args, **_kwargs: None)

    seed = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html"
    short_page = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/short.html"
    html_map = {
        seed: _html_with_text(
            "Seed page text " + ("A" * 60),
            links='<a href="short.html">Short</a>',
        ),
        short_page: _html_with_text("tiny"),
    }

    monkeypatch.setattr(ingest_module, "get_page", lambda url: html_map[url])

    docs, report = ingest_module.scrape_service("s3", seed)

    assert [doc["url"] for doc in docs] == [seed]
    assert report["pages_attempted"] == 2
    assert report["pages_failed"] == 0
    assert report["skipped_pages"] == [
        {
            "service": "s3",
            "url": short_page,
            "reason": "short_content",
            "char_count": len("tiny\ntiny"),
            "minimum_char_count": 50,
        }
    ]


def test_scrape_service_marks_seed_failure(ingest_module, monkeypatch):
    monkeypatch.setattr(ingest_module, "REQUEST_DELAY", 0)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda *_args, **_kwargs: None)

    seed = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html"
    monkeypatch.setattr(ingest_module, "get_page", lambda url: None)

    docs, report = ingest_module.scrape_service("s3", seed)

    assert docs == []
    assert report["pages_attempted"] == 1
    assert report["pages_failed"] == 1
    assert report["failed_urls"] == [seed]
    assert report["seed_failed"] is True


def test_summarize_crawl_reports_aggregates_mixed_results(ingest_module):
    service_reports = [
        {
            "service": "s3",
            "seed_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "pages_attempted": 3,
            "pages_failed": 0,
            "failed_urls": [],
            "seed_failed": False,
            "skipped_pages": [
                {
                    "service": "s3",
                    "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/short-1.html",
                    "reason": "short_content",
                    "char_count": 12,
                    "minimum_char_count": ingest_module.MIN_TEXT_CHARS,
                },
                {
                    "service": "s3",
                    "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/short-2.html",
                    "reason": "short_content",
                    "char_count": 15,
                    "minimum_char_count": ingest_module.MIN_TEXT_CHARS,
                },
            ],
        },
        {
            "service": "ec2",
            "seed_url": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
            "pages_attempted": 2,
            "pages_failed": 0,
            "failed_urls": [],
            "seed_failed": False,
            "skipped_pages": [
                {
                    "service": "ec2",
                    "url": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/nav.html",
                    "reason": "nav_only",
                    "char_count": 8,
                    "minimum_char_count": ingest_module.MIN_TEXT_CHARS,
                }
            ],
        },
    ]

    summary = ingest_module.summarize_crawl_reports(service_reports)

    assert summary["total_attempts"] == 5
    assert summary["total_failures"] == 0
    assert summary["failure_rate"] == 0.0
    assert summary["run_failed"] is False
    assert summary["skipped_pages_total"] == 3
    assert summary["skipped_pages_by_reason"] == {"short_content": 2, "nav_only": 1}
    assert summary["skipped_pages_by_service"] == {"s3": 2, "ec2": 1}
    assert summary["services"]["s3"]["skipped_pages"] == 2
    assert summary["services"]["ec2"]["skipped_pages"] == 1


def test_summarize_crawl_reports_all_success(ingest_module):
    service_reports = [
        {
            "service": "lambda",
            "seed_url": "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
            "pages_attempted": 4,
            "pages_failed": 0,
            "failed_urls": [],
            "seed_failed": False,
            "skipped_pages": [],
        }
    ]

    summary = ingest_module.summarize_crawl_reports(service_reports)

    assert summary["run_failed"] is False
    assert summary["services"]["lambda"]["failure_rate"] == 0.0
    assert summary["services"]["lambda"]["seed_failed"] is False


def test_summarize_crawl_reports_marks_one_service_failed(ingest_module):
    service_reports = [
        {
            "service": "s3",
            "seed_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "pages_attempted": 2,
            "pages_failed": 0,
            "failed_urls": [],
            "seed_failed": False,
            "skipped_pages": [],
        },
        {
            "service": "ec2",
            "seed_url": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
            "pages_attempted": 1,
            "pages_failed": 0,
            "failed_urls": [],
            "seed_failed": True,
            "skipped_pages": [],
        },
    ]

    summary = ingest_module.summarize_crawl_reports(service_reports)

    assert summary["run_failed"] is True
    assert summary["services"]["ec2"]["seed_failed"] is True


def test_summarize_crawl_reports_handles_empty_input(ingest_module):
    summary = ingest_module.summarize_crawl_reports([])

    assert summary == {
        "total_attempts": 0,
        "total_failures": 0,
        "failure_rate": 0.0,
        "failure_threshold": ingest_module.CRAWL_FAILURE_RATE_THRESHOLD,
        "run_failed": False,
        "services": {},
        "skipped_pages_total": 0,
        "skipped_pages_by_reason": {},
        "skipped_pages_by_service": {},
        "skipped_pages": [],
    }


def test_chunk_documents_splits_preserves_metadata_and_assigns_ids(chunk_module):
    small_doc = {
        "service": "s3",
        "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "content": "short content",
    }
    large_text = "L" * 1200
    large_doc = {
        "service": "ec2",
        "url": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
        "content": large_text,
    }
    empty_doc = {
        "service": "lambda",
        "url": "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
        "content": "",
    }

    chunks = chunk_module.chunk_documents([small_doc, large_doc, empty_doc])

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "chunk_000000",
        "chunk_000001",
        "chunk_000002",
    ]
    assert [chunk["service"] for chunk in chunks] == ["s3", "ec2", "ec2"]
    assert [chunk["source_url"] for chunk in chunks] == [
        small_doc["url"],
        large_doc["url"],
        large_doc["url"],
    ]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 0, 1]
    assert [chunk["total_chunks_in_doc"] for chunk in chunks] == [1, 2, 2]
    assert chunks[0]["content"] == "short content"
    assert chunks[0]["char_count"] == len("short content")
    assert chunks[1]["content"] == large_text[:1000]
    assert chunks[2]["content"] == large_text[800:]
    assert chunks[2]["content"][:200] == chunks[1]["content"][-200:]
    assert chunk_module.RecursiveCharacterTextSplitter.instances[0].chunk_size == 1000
    assert chunk_module.RecursiveCharacterTextSplitter.instances[0].chunk_overlap == 200


def test_chunk_documents_skips_empty_documents_cleanly(chunk_module):
    chunks = chunk_module.chunk_documents([
        {
            "service": "s3",
            "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "content": "",
        }
    ])

    assert chunks == []


def test_chunk_main_exits_when_no_chunks_are_produced(chunk_module, monkeypatch):
    monkeypatch.setattr(chunk_module, "load_documents_from_s3", lambda: (
        [
            {
                "service": "s3",
                "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
                "content": "seed",
            }
        ],
        {
            "run_id": "raw-run-123",
            "documents_prefix": "raw-docs/raw-run-123/",
        },
    ))
    monkeypatch.setattr(chunk_module, "chunk_documents", lambda _documents: [])
    monkeypatch.setattr(chunk_module, "write_chunks_manifest", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit, match="1"):
        chunk_module.main()


def test_chunk_main_success_carries_source_run_id_and_writes_local_output(chunk_module, monkeypatch):
    written = {}
    uploaded = {}

    class RecordingFile(io.StringIO):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def __exit__(self, exc_type, exc, tb):
            written[self.path] = self.getvalue()
            return super().__exit__(exc_type, exc, tb)

    def fake_open(path, mode="r", *args, **kwargs):
        return RecordingFile(path)

    monkeypatch.setattr(chunk_module, "load_documents_from_s3", lambda: (
        [
            {
                "service": "s3",
                "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
                "content": "seed",
            }
        ],
        {
            "run_id": "raw-run-123",
            "documents_prefix": "raw-docs/raw-run-123/",
        },
    ))
    monkeypatch.setattr(
        chunk_module,
        "chunk_documents",
        lambda _documents: [
            {
                "chunk_id": "chunk_000000",
                "service": "s3",
                "source_url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
                "chunk_index": 0,
                "total_chunks_in_doc": 1,
                "content": "seed",
                "char_count": 4,
            }
        ],
    )
    monkeypatch.setattr(
        chunk_module,
        "upload_chunks_to_s3",
        lambda chunks, run_id, source_manifest: uploaded.update(
            {"chunks": chunks, "run_id": run_id, "source_manifest": source_manifest}
        ),
    )
    monkeypatch.setattr(chunk_module.os, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("builtins.open", fake_open)

    chunk_module.main()

    assert uploaded["source_manifest"] == {
        "run_id": "raw-run-123",
        "documents_prefix": "raw-docs/raw-run-123/",
    }
    assert uploaded["chunks"][0]["chunk_id"] == "chunk_000000"
    assert "local-data/chunks/all_chunks.json" in written
    assert json.loads(written["local-data/chunks/all_chunks.json"]) == uploaded["chunks"]
