import os


DEFAULT_ALLOWED_ORIGIN = "https://d3d0zch3u8ca61.cloudfront.net"


def get_allowed_origin() -> str:
    """Return the frontend origin allowed by Lambda and API Gateway."""
    return os.environ.get("ALLOWED_ORIGIN", DEFAULT_ALLOWED_ORIGIN)
