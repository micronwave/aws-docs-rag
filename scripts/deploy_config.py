import os
import re
import secrets
from pathlib import Path


DEFAULT_ALLOWED_ORIGIN = "https://d3d0zch3u8ca61.cloudfront.net"
DEFAULT_ORIGIN_VERIFY_HEADER = "x-origin-verify"
ORIGIN_VERIFY_SECRET_FILE = Path(__file__).resolve().parents[1] / "origin_verify_secret.txt"


def get_allowed_origin() -> str:
    """Return the frontend origin allowed by Lambda and API Gateway."""
    return os.environ.get("ALLOWED_ORIGIN", DEFAULT_ALLOWED_ORIGIN)


def get_origin_verify_header() -> str:
    """Return the header name CloudFront uses to prove requests came from the edge."""
    header = os.environ.get("ORIGIN_VERIFY_HEADER", DEFAULT_ORIGIN_VERIFY_HEADER).strip()
    if not header or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", header):
        raise ValueError("ORIGIN_VERIFY_HEADER must be a valid HTTP header name")
    return header


def _read_secret_file() -> str:
    if ORIGIN_VERIFY_SECRET_FILE.is_symlink():
        raise RuntimeError(f"Refusing to read symlinked secret file: {ORIGIN_VERIFY_SECRET_FILE}")
    os.chmod(ORIGIN_VERIFY_SECRET_FILE, 0o600)
    return ORIGIN_VERIFY_SECRET_FILE.read_text(encoding="utf-8").strip()


def get_origin_verify_secret() -> str:
    """Return a stable secret shared between CloudFront and the Lambda backend."""
    secret = os.environ.get("ORIGIN_VERIFY_SECRET", "").strip()
    if secret:
        return secret

    if ORIGIN_VERIFY_SECRET_FILE.exists():
        file_secret = _read_secret_file()
        if file_secret:
            return file_secret

    secret = secrets.token_urlsafe(48)
    try:
        fd = os.open(ORIGIN_VERIFY_SECRET_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        file_secret = _read_secret_file()
        if file_secret:
            return file_secret
        raise RuntimeError(f"Origin verification secret file is empty: {ORIGIN_VERIFY_SECRET_FILE}")

    with os.fdopen(fd, "w", encoding="utf-8") as secret_file:
        secret_file.write(secret)
    return secret
