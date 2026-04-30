from scripts.deploy_config import DEFAULT_ALLOWED_ORIGIN, get_allowed_origin


def test_get_allowed_origin_defaults_to_demo_domain(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGIN", raising=False)

    assert get_allowed_origin() == DEFAULT_ALLOWED_ORIGIN


def test_get_allowed_origin_uses_environment_override(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGIN", "https://example.cloudfront.net")

    assert get_allowed_origin() == "https://example.cloudfront.net"
