import os
import shutil
import stat
import uuid
from pathlib import Path

import pytest

from scripts.deploy_config import (
    DEFAULT_ALLOWED_ORIGIN,
    DEFAULT_ORIGIN_VERIFY_HEADER,
    get_allowed_origin,
    get_origin_verify_header,
    get_origin_verify_secret,
)


def test_get_allowed_origin_defaults_to_demo_domain(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGIN", raising=False)

    assert get_allowed_origin() == DEFAULT_ALLOWED_ORIGIN


def test_get_allowed_origin_uses_environment_override(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGIN", "https://example.cloudfront.net")

    assert get_allowed_origin() == "https://example.cloudfront.net"


def test_get_origin_verify_header_defaults(monkeypatch):
    monkeypatch.delenv("ORIGIN_VERIFY_HEADER", raising=False)

    assert get_origin_verify_header() == DEFAULT_ORIGIN_VERIFY_HEADER


def test_get_origin_verify_header_uses_environment_override(monkeypatch):
    monkeypatch.setenv("ORIGIN_VERIFY_HEADER", "x-custom-origin-secret")

    assert get_origin_verify_header() == "x-custom-origin-secret"


def test_get_origin_verify_header_rejects_invalid_override(monkeypatch):
    monkeypatch.setenv("ORIGIN_VERIFY_HEADER", "bad header")

    with pytest.raises(ValueError):
        get_origin_verify_header()


def test_get_origin_verify_secret_uses_environment_override(monkeypatch):
    monkeypatch.setenv("ORIGIN_VERIFY_SECRET", "env-secret")

    assert get_origin_verify_secret() == "env-secret"


def test_get_origin_verify_secret_uses_existing_file(monkeypatch):
    artifacts_dir = Path(__file__).resolve().parents[1] / ".test-artifacts" / f"deploy-config-{uuid.uuid4().hex}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    secret_file = artifacts_dir / "origin_verify_secret.txt"
    secret_file.write_text("file-secret", encoding="utf-8")
    monkeypatch.delenv("ORIGIN_VERIFY_SECRET", raising=False)
    monkeypatch.setattr("scripts.deploy_config.ORIGIN_VERIFY_SECRET_FILE", secret_file)
    try:
        assert get_origin_verify_secret() == "file-secret"
    finally:
        shutil.rmtree(artifacts_dir, ignore_errors=True)


def test_get_origin_verify_secret_generates_and_persists(monkeypatch):
    artifacts_dir = Path(__file__).resolve().parents[1] / ".test-artifacts" / f"deploy-config-{uuid.uuid4().hex}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    secret_file = artifacts_dir / "origin_verify_secret.txt"
    monkeypatch.delenv("ORIGIN_VERIFY_SECRET", raising=False)
    monkeypatch.setattr("scripts.deploy_config.ORIGIN_VERIFY_SECRET_FILE", secret_file)
    try:
        secret = get_origin_verify_secret()

        assert secret
        assert secret_file.read_text(encoding="utf-8") == secret
        if os.name != "nt":
            assert stat.S_IMODE(secret_file.stat().st_mode) & 0o077 == 0
    finally:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
