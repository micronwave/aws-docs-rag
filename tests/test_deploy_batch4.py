import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import types
import uuid
import zipfile
from pathlib import Path

import pytest
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parents[1]


def load_script(monkeypatch, module_name: str, relative_path: str, *, boto3_client_map):
    boto3_stub = types.ModuleType("boto3")
    boto3_stub.client = lambda service_name, **_kwargs: boto3_client_map[service_name]
    monkeypatch.setitem(sys.modules, "boto3", boto3_stub)

    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"{module_name}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeWaiter:
    def __init__(self):
        self.calls = []

    def wait(self, **kwargs):
        self.calls.append(kwargs)


class FakeIAM:
    def __init__(self, *, existing_role=None, get_role_error=None):
        self.existing_role = existing_role
        self.get_role_error = get_role_error
        self.calls = []

    def get_role(self, **kwargs):
        self.calls.append(("get_role", kwargs))
        if self.get_role_error is not None:
            raise self.get_role_error
        return {"Role": {"Arn": self.existing_role}}

    def put_role_policy(self, **kwargs):
        self.calls.append(("put_role_policy", kwargs))

    def detach_role_policy(self, **kwargs):
        self.calls.append(("detach_role_policy", kwargs))

    def create_role(self, **kwargs):
        self.calls.append(("create_role", kwargs))
        return {"Role": {"Arn": "arn:aws:iam::123456789012:role/new-role"}}

    def attach_role_policy(self, **kwargs):
        self.calls.append(("attach_role_policy", kwargs))


class FakeLambdaClient:
    def __init__(self, *, update_error=None, create_error=None, function_arn="arn:aws:lambda:us-east-1:123:function:aws-rag-query"):
        self.update_error = update_error
        self.create_error = create_error
        self.function_arn = function_arn
        self.calls = []
        self.waiters = {}

    def get_waiter(self, name):
        waiter = self.waiters.get(name)
        if waiter is None:
            waiter = FakeWaiter()
            self.waiters[name] = waiter
        return waiter

    def update_function_code(self, **kwargs):
        self.calls.append(("update_function_code", kwargs))
        if self.update_error is not None:
            raise self.update_error

    def update_function_configuration(self, **kwargs):
        self.calls.append(("update_function_configuration", kwargs))

    def get_function(self, **kwargs):
        self.calls.append(("get_function", kwargs))
        return {"Configuration": {"FunctionArn": self.function_arn}}

    def create_function(self, **kwargs):
        self.calls.append(("create_function", kwargs))
        if self.create_error is not None:
            raise self.create_error
        return {"FunctionArn": self.function_arn}


class FakeSTS:
    def __init__(self, account="123456789012"):
        self.account = account
        self.calls = []

    def get_caller_identity(self):
        self.calls.append(("get_caller_identity", {}))
        return {"Account": self.account}


class FakeAPIGW:
    def __init__(self):
        self.calls = []
        self.pages = []
        self.resources = []

    def get_paginator(self, name):
        self.calls.append(("get_paginator", {"name": name}))

        class _Paginator:
            def __init__(self, pages):
                self.pages = pages

            def paginate(self):
                return self.pages

        return _Paginator(self.pages)

    def create_rest_api(self, **kwargs):
        self.calls.append(("create_rest_api", kwargs))
        return {"id": "api-new"}

    def get_resources(self, **kwargs):
        self.calls.append(("get_resources", kwargs))
        return {"items": self.resources}

    def create_resource(self, **kwargs):
        self.calls.append(("create_resource", kwargs))
        return {"id": "res-new"}

    def delete_method(self, **kwargs):
        self.calls.append(("delete_method", kwargs))

    def put_method(self, **kwargs):
        self.calls.append(("put_method", kwargs))

    def put_integration(self, **kwargs):
        self.calls.append(("put_integration", kwargs))

    def put_method_response(self, **kwargs):
        self.calls.append(("put_method_response", kwargs))

    def put_integration_response(self, **kwargs):
        self.calls.append(("put_integration_response", kwargs))

    def put_gateway_response(self, **kwargs):
        self.calls.append(("put_gateway_response", kwargs))

    def create_deployment(self, **kwargs):
        self.calls.append(("create_deployment", kwargs))

    def update_stage(self, **kwargs):
        self.calls.append(("update_stage", kwargs))


class FakeS3:
    def __init__(self, *, head_error=None):
        self.head_error = head_error
        self.calls = []

    def head_bucket(self, **kwargs):
        self.calls.append(("head_bucket", kwargs))
        if self.head_error is not None:
            raise self.head_error

    def create_bucket(self, **kwargs):
        self.calls.append(("create_bucket", kwargs))

    def put_bucket_website(self, **kwargs):
        self.calls.append(("put_bucket_website", kwargs))

    def put_public_access_block(self, **kwargs):
        self.calls.append(("put_public_access_block", kwargs))

    def put_object(self, **kwargs):
        self.calls.append(("put_object", kwargs))

    def put_bucket_policy(self, **kwargs):
        self.calls.append(("put_bucket_policy", kwargs))


class FakeCloudFront:
    def __init__(self):
        self.calls = []
        self.origin_access_controls = []
        self.distributions = []
        self.distribution_config = {
            "Origins": {"Items": [{
                "Id": "s3-frontend",
                "DomainName": "bucket.s3.us-east-1.amazonaws.com",
                "CustomOriginConfig": {"HTTPPort": 80},
            }]},
            "DefaultCacheBehavior": {"TargetOriginId": "s3-frontend"},
        }

    def list_origin_access_controls(self):
        self.calls.append(("list_origin_access_controls", {}))
        return {"OriginAccessControlList": {"Items": self.origin_access_controls}}

    def create_origin_access_control(self, **kwargs):
        self.calls.append(("create_origin_access_control", kwargs))
        return {"OriginAccessControl": {"Id": "oac-new"}}

    def get_paginator(self, name):
        self.calls.append(("get_paginator", {"name": name}))

        class _Paginator:
            def __init__(self, pages):
                self.pages = pages

            def paginate(self):
                return self.pages

        return _Paginator(self.distributions)

    def get_distribution_config(self, **kwargs):
        self.calls.append(("get_distribution_config", kwargs))
        return {
            "DistributionConfig": self.distribution_config,
            "ETag": "etag-1",
        }

    def update_distribution(self, **kwargs):
        self.calls.append(("update_distribution", kwargs))

    def create_distribution(self, **kwargs):
        self.calls.append(("create_distribution", kwargs))
        return {"Distribution": {"DomainName": "d111.cloudfront.net", "Id": "dist-new", "ARN": "arn:aws:cloudfront::123:distribution/dist-new"}}


@pytest.fixture()
def lambda_module(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    iam = FakeIAM(existing_role="arn:aws:iam::123456789012:role/existing")
    lambda_client = FakeLambdaClient()
    sts = FakeSTS()
    module = load_script(
        monkeypatch,
        "deploy_lambda_test",
        "scripts/06_deploy_lambda.py",
        boto3_client_map={"iam": iam, "lambda": lambda_client, "sts": sts},
    )
    module._test_iam = iam
    module._test_lambda = lambda_client
    module._test_sts = sts
    return module


@pytest.fixture()
def api_module(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    iam = FakeIAM(existing_role="arn:aws:iam::123456789012:role/existing")
    lambda_client = FakeLambdaClient()
    sts = FakeSTS()
    apigw = FakeAPIGW()
    module = load_script(
        monkeypatch,
        "deploy_api_test",
        "scripts/07_deploy_api_gateway.py",
        boto3_client_map={"apigateway": apigw, "lambda": lambda_client, "sts": sts},
    )
    module._test_apigw = apigw
    module._test_lambda = lambda_client
    module._test_sts = sts
    return module


@pytest.fixture()
def frontend_module(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    sts = FakeSTS()
    s3 = FakeS3()
    cf = FakeCloudFront()
    module = load_script(
        monkeypatch,
        "deploy_frontend_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": sts, "s3": s3, "cloudfront": cf},
    )
    module._test_s3 = s3
    module._test_cf = cf
    module._test_sts = sts
    return module


def test_create_role_uses_existing_role_and_attaches_scoped_policy(lambda_module):
    role_arn = lambda_module.create_role()

    assert role_arn == "arn:aws:iam::123456789012:role/existing"
    names = [name for name, _kwargs in lambda_module._test_iam.calls]
    assert "get_role" in names
    assert "put_role_policy" in names
    assert "detach_role_policy" in names


def test_create_role_creates_new_role_with_basic_and_scoped_policies(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    not_found = ClientError({"Error": {"Code": "NoSuchEntity", "Message": "missing"}}, "GetRole")
    iam = FakeIAM(get_role_error=not_found)
    lambda_client = FakeLambdaClient()
    sts = FakeSTS()
    module = load_script(
        monkeypatch,
        "deploy_lambda_new_role_test",
        "scripts/06_deploy_lambda.py",
        boto3_client_map={"iam": iam, "lambda": lambda_client, "sts": sts},
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    role_arn = module.create_role()

    assert role_arn == "arn:aws:iam::123456789012:role/new-role"
    calls = {name for name, _kwargs in iam.calls}
    assert {"create_role", "attach_role_policy", "put_role_policy"}.issubset(calls)


def test_create_role_attaches_scoped_bedrock_policy(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    not_found = ClientError({"Error": {"Code": "NoSuchEntity", "Message": "missing"}}, "GetRole")
    iam = FakeIAM(get_role_error=not_found)
    lambda_client = FakeLambdaClient()
    sts = FakeSTS()
    module = load_script(
        monkeypatch,
        "deploy_lambda_policy_test",
        "scripts/06_deploy_lambda.py",
        boto3_client_map={"iam": iam, "lambda": lambda_client, "sts": sts},
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    module.create_role()

    put_call = next(kwargs for name, kwargs in iam.calls if name == "put_role_policy")
    policy = json.loads(put_call["PolicyDocument"])
    assert put_call["PolicyName"] == "BedrockInvokeOnly"
    assert policy["Statement"][0]["Action"] == "bedrock:InvokeModel"
    assert len(policy["Statement"][0]["Resource"]) == 2


def test_package_lambda_includes_handler_and_installed_dependencies(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    module = load_script(
        monkeypatch,
        "deploy_lambda_package_test",
        "scripts/06_deploy_lambda.py",
        boto3_client_map={"iam": FakeIAM(existing_role="arn"), "lambda": FakeLambdaClient(), "sts": FakeSTS()},
    )
    artifacts_dir = ROOT / ".test-artifacts" / f"lambda-package-{uuid.uuid4().hex}"
    build_dir = artifacts_dir / "lambda-build"
    zip_path = artifacts_dir / "lambda-deployment.zip"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(artifacts_dir))

        def fake_run(*_args, **_kwargs):
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "orjson").mkdir(exist_ok=True)
            (build_dir / "orjson" / "__init__.py").write_text("x=1", encoding="utf-8")
            (build_dir / "pinecone").mkdir(exist_ok=True)
            (build_dir / "pinecone" / "__init__.py").write_text("y=2", encoding="utf-8")

        monkeypatch.setattr(module.subprocess, "run", fake_run)

        created_zip = Path(module.package_lambda())

        assert created_zip == zip_path
        assert created_zip.exists()
        with zipfile.ZipFile(created_zip, "r") as zf:
            names = set(zf.namelist())
        assert "lambda_handler.py" in names
        assert "orjson/__init__.py" in names
        assert "pinecone/__init__.py" in names
    finally:
        shutil.rmtree(artifacts_dir, ignore_errors=True)


def test_deploy_function_updates_existing_function_with_expected_env_vars(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    iam = FakeIAM(existing_role="arn")
    lambda_client = FakeLambdaClient()
    module = load_script(
        monkeypatch,
        "deploy_lambda_update_test",
        "scripts/06_deploy_lambda.py",
        boto3_client_map={"iam": iam, "lambda": lambda_client, "sts": FakeSTS()},
    )
    artifacts_dir = ROOT / ".test-artifacts" / f"lambda-update-{uuid.uuid4().hex}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        zip_path = artifacts_dir / "deployment.zip"
        zip_path.write_bytes(b"zip")

        monkeypatch.setattr(module, "get_allowed_origin", lambda: "https://frontend.example")
        monkeypatch.setattr(module, "get_origin_verify_header", lambda: "x-origin-verify")
        monkeypatch.setattr(module, "get_origin_verify_secret", lambda: "shared-secret")

        func_arn = module.deploy_function("arn:role", str(zip_path))

        assert func_arn == "arn:aws:lambda:us-east-1:123:function:aws-rag-query"
        update_config_call = next(kwargs for name, kwargs in lambda_client.calls if name == "update_function_configuration")
        env_vars = update_config_call["Environment"]["Variables"]
        assert env_vars["PINECONE_API_KEY"] == "test-key"
        assert env_vars["ALLOWED_ORIGIN"] == "https://frontend.example"
        assert env_vars["ORIGIN_VERIFY_HEADER"] == "x-origin-verify"
        assert env_vars["ORIGIN_VERIFY_SECRET"] == "shared-secret"
    finally:
        shutil.rmtree(artifacts_dir, ignore_errors=True)


def test_deploy_function_creates_new_function_when_missing(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    not_found = ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}}, "UpdateFunctionCode")
    lambda_client = FakeLambdaClient(update_error=not_found)
    module = load_script(
        monkeypatch,
        "deploy_lambda_create_test",
        "scripts/06_deploy_lambda.py",
        boto3_client_map={"iam": FakeIAM(existing_role="arn"), "lambda": lambda_client, "sts": FakeSTS()},
    )
    artifacts_dir = ROOT / ".test-artifacts" / f"lambda-create-{uuid.uuid4().hex}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        zip_path = artifacts_dir / "deployment.zip"
        zip_path.write_bytes(b"zip")
        monkeypatch.setattr(module, "get_allowed_origin", lambda: "https://frontend.example")
        monkeypatch.setattr(module, "get_origin_verify_header", lambda: "x-origin-verify")
        monkeypatch.setattr(module, "get_origin_verify_secret", lambda: "shared-secret")

        func_arn = module.deploy_function("arn:role", str(zip_path))

        assert func_arn == "arn:aws:lambda:us-east-1:123:function:aws-rag-query"
        create_call = next(kwargs for name, kwargs in lambda_client.calls if name == "create_function")
        assert create_call["Role"] == "arn:role"
        assert create_call["Environment"]["Variables"]["ALLOWED_ORIGIN"] == "https://frontend.example"
        assert create_call["Environment"]["Variables"]["ORIGIN_VERIFY_HEADER"] == "x-origin-verify"
        assert create_call["Environment"]["Variables"]["ORIGIN_VERIFY_SECRET"] == "shared-secret"
    finally:
        shutil.rmtree(artifacts_dir, ignore_errors=True)


def test_get_or_create_api_returns_existing_api_id(api_module):
    api_module._test_apigw.pages = [{"items": [{"name": api_module.API_NAME, "id": "api-existing"}]}]

    assert api_module.get_or_create_api() == "api-existing"


def test_get_or_create_api_creates_api_when_missing(monkeypatch):
    apigw = FakeAPIGW()
    apigw.pages = [{"items": []}]
    module = load_script(
        monkeypatch,
        "deploy_api_create_test",
        "scripts/07_deploy_api_gateway.py",
        boto3_client_map={"apigateway": apigw, "lambda": FakeLambdaClient(), "sts": FakeSTS()},
    )

    assert module.get_or_create_api() == "api-new"
    assert any(name == "create_rest_api" for name, _kwargs in apigw.calls)


def test_get_or_create_resource_returns_existing_resource_id(api_module):
    api_module._test_apigw.resources = [{"path": "/", "id": "root"}, {"pathPart": "query", "id": "res-existing"}]

    assert api_module.get_or_create_resource("api-id", "root", "query") == "res-existing"


def test_get_or_create_resource_creates_resource_when_missing(monkeypatch):
    apigw = FakeAPIGW()
    apigw.resources = [{"path": "/", "id": "root"}]
    module = load_script(
        monkeypatch,
        "deploy_api_resource_create_test",
        "scripts/07_deploy_api_gateway.py",
        boto3_client_map={"apigateway": apigw, "lambda": FakeLambdaClient(), "sts": FakeSTS()},
    )

    assert module.get_or_create_resource("api-id", "root", "query") == "res-new"
    assert any(name == "create_resource" for name, _kwargs in apigw.calls)


def test_setup_method_configures_post_lambda_proxy(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    apigw = FakeAPIGW()
    module = load_script(
        monkeypatch,
        "deploy_api_post_method_test",
        "scripts/07_deploy_api_gateway.py",
        boto3_client_map={"apigateway": apigw, "lambda": FakeLambdaClient(), "sts": FakeSTS()},
    )

    module.setup_method("api-id", "res-id", "POST", "arn:aws:lambda:us-east-1:123:function:fn")

    integration_call = next(kwargs for name, kwargs in apigw.calls if name == "put_integration")
    assert integration_call["type"] == "AWS_PROXY"
    assert integration_call["integrationHttpMethod"] == "POST"
    assert f"arn:aws:apigateway:{module.REGION}:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:123:function:fn/invocations" == integration_call["uri"]


def test_setup_method_configures_options_cors_headers(monkeypatch):
    apigw = FakeAPIGW()
    module = load_script(
        monkeypatch,
        "deploy_api_options_method_test",
        "scripts/07_deploy_api_gateway.py",
        boto3_client_map={"apigateway": apigw, "lambda": FakeLambdaClient(), "sts": FakeSTS()},
    )
    monkeypatch.setattr(module, "get_allowed_origin", lambda: "https://frontend.example")

    module.setup_method("api-id", "res-id", "OPTIONS", "arn:aws:lambda:us-east-1:123:function:fn", is_cors=True)

    integration_call = next(kwargs for name, kwargs in apigw.calls if name == "put_integration")
    assert integration_call["type"] == "MOCK"
    response_call = next(kwargs for name, kwargs in apigw.calls if name == "put_integration_response")
    assert response_call["responseParameters"]["method.response.header.Access-Control-Allow-Origin"] == "'https://frontend.example'"


def test_setup_gateway_responses_sets_cors_headers_on_error_types(monkeypatch):
    apigw = FakeAPIGW()
    module = load_script(
        monkeypatch,
        "deploy_api_gateway_response_test",
        "scripts/07_deploy_api_gateway.py",
        boto3_client_map={"apigateway": apigw, "lambda": FakeLambdaClient(), "sts": FakeSTS()},
    )
    monkeypatch.setattr(module, "get_allowed_origin", lambda: "https://frontend.example")

    module.setup_gateway_responses("api-id")

    gateway_calls = [kwargs for name, kwargs in apigw.calls if name == "put_gateway_response"]
    assert len(gateway_calls) == len(module.GATEWAY_ERROR_TYPES)
    assert gateway_calls[0]["responseParameters"]["gatewayresponse.header.Access-Control-Allow-Origin"] == "'https://frontend.example'"


def test_deploy_api_restores_post_throttling_without_api_key(api_module):
    endpoint = api_module.deploy_api("api-id")

    assert endpoint.endswith("/prod")
    update_call = next(
        kwargs for name, kwargs in api_module._test_apigw.calls if name == "update_stage"
    )
    operations = {item["path"]: item["value"] for item in update_call["patchOperations"]}
    assert operations["/methodSettings/~1query~1POST/throttling/rateLimit"] == "5"
    assert operations["/methodSettings/~1query~1POST/throttling/burstLimit"] == "10"


def test_create_frontend_bucket_creates_or_reuses_bucket_and_sets_website_and_public_access_block(frontend_module):
    frontend_module.create_frontend_bucket()

    names = [name for name, _kwargs in frontend_module._test_s3.calls]
    assert "put_bucket_website" in names
    assert "put_public_access_block" in names


def test_upload_frontend_injects_api_endpoint_before_upload(monkeypatch):
    s3 = FakeS3()
    cf = FakeCloudFront()
    sts = FakeSTS()
    module = load_script(
        monkeypatch,
        "deploy_frontend_upload_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": sts, "s3": s3, "cloudfront": cf},
    )
    monkeypatch.chdir(ROOT)
    module.upload_frontend("/query")

    put_object_call = next(kwargs for name, kwargs in s3.calls if name == "put_object")
    body = put_object_call["Body"].decode("utf-8")
    assert "/query" in body
    assert "%%API_ENDPOINT%%" not in body
    assert "%%API_KEY%%" not in body


def test_build_api_origin_uses_server_side_secret(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    cf = FakeCloudFront()
    module = load_script(
        monkeypatch,
        "deploy_frontend_api_origin_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": FakeSTS(), "s3": FakeS3(), "cloudfront": cf},
    )
    monkeypatch.setattr(module, "get_origin_verify_header", lambda: "x-origin-verify")
    monkeypatch.setattr(module, "get_origin_verify_secret", lambda: "shared-secret")

    origin = module.build_api_origin("https://abc123.execute-api.us-east-1.amazonaws.com/prod/query")

    assert origin["DomainName"] == "abc123.execute-api.us-east-1.amazonaws.com"
    assert origin["OriginPath"] == "/prod"
    assert origin["CustomHeaders"]["Items"][0] == {
        "HeaderName": "x-origin-verify",
        "HeaderValue": "shared-secret",
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://abc123.execute-api.us-east-1.amazonaws.com/prod/query",
        "https://attacker.example/prod/query",
        "https://abc123.execute-api.us-west-2.amazonaws.com/prod/query",
        "https://abc123.execute-api.us-east-1.amazonaws.com/prod/not-query",
        "https://abc123.execute-api.us-east-1.amazonaws.com/prod/query?redirect=attacker",
    ],
)
def test_parse_api_gateway_origin_rejects_untrusted_endpoints(monkeypatch, endpoint):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    module = load_script(
        monkeypatch,
        "deploy_frontend_untrusted_origin_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": FakeSTS(), "s3": FakeS3(), "cloudfront": FakeCloudFront()},
    )

    with pytest.raises(ValueError):
        module.parse_api_gateway_origin(endpoint)


def test_ensure_api_cache_behavior_uses_exact_query_path(monkeypatch):
    module = load_script(
        monkeypatch,
        "deploy_frontend_exact_query_path_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": FakeSTS(), "s3": FakeS3(), "cloudfront": FakeCloudFront()},
    )
    config = {"CacheBehaviors": {"Quantity": 1, "Items": [{"PathPattern": "query*"}]}}

    assert module.ensure_api_cache_behavior(config) is True
    assert config["CacheBehaviors"]["Items"][0]["PathPattern"] == "query"


def test_get_or_create_oac_returns_existing_oac(monkeypatch):
    cf = FakeCloudFront()
    cf.origin_access_controls = [{"Name": "aws-rag-frontend-oac", "Id": "oac-existing"}]
    module = load_script(
        monkeypatch,
        "deploy_frontend_oac_existing_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": FakeSTS(), "s3": FakeS3(), "cloudfront": cf},
    )

    assert module.get_or_create_oac() == "oac-existing"


def test_get_or_create_oac_creates_new_oac_when_missing(monkeypatch):
    cf = FakeCloudFront()
    module = load_script(
        monkeypatch,
        "deploy_frontend_oac_create_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": FakeSTS(), "s3": FakeS3(), "cloudfront": cf},
    )

    assert module.get_or_create_oac() == "oac-new"
    assert any(name == "create_origin_access_control" for name, _kwargs in cf.calls)


def test_create_cloudfront_distribution_reuses_existing_distribution_with_oac(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    cf = FakeCloudFront()
    module = load_script(
        monkeypatch,
        "deploy_frontend_cf_reuse_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": FakeSTS(), "s3": FakeS3(), "cloudfront": cf},
    )
    cf.distributions = [{
        "DistributionList": {
            "Items": [{
                "Id": "dist-existing",
                "ARN": "arn:aws:cloudfront::123:distribution/dist-existing",
                "DomainName": "d111.cloudfront.net",
                "Origins": {"Items": [{"DomainName": f"{module.FRONTEND_BUCKET}.s3.{module.REGION}.amazonaws.com", "S3OriginConfig": {"OriginAccessIdentity": ""}}]},
            }]
        }
    }]

    monkeypatch.setattr(module, "get_origin_verify_header", lambda: "x-origin-verify")
    monkeypatch.setattr(module, "get_origin_verify_secret", lambda: "shared-secret")

    url, arn = module.create_cloudfront_distribution("https://abc123.execute-api.us-east-1.amazonaws.com/prod/query")

    assert url == "https://d111.cloudfront.net"
    assert arn == "arn:aws:cloudfront::123:distribution/dist-existing"
    update_call = next(kwargs for name, kwargs in cf.calls if name == "update_distribution")
    assert any(item.get("Id") == "api-gateway-query" for item in update_call["DistributionConfig"]["Origins"]["Items"])


def test_create_cloudfront_distribution_updates_custom_origin_distribution_to_oac(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    cf = FakeCloudFront()
    module = load_script(
        monkeypatch,
        "deploy_frontend_cf_update_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": FakeSTS(), "s3": FakeS3(), "cloudfront": cf},
    )
    cf.distributions = [{
        "DistributionList": {
            "Items": [{
                "Id": "dist-existing",
                "ARN": "arn:aws:cloudfront::123:distribution/dist-existing",
                "DomainName": "d111.cloudfront.net",
                "Origins": {"Items": [{"DomainName": f"{module.FRONTEND_BUCKET}.s3.{module.REGION}.amazonaws.com", "CustomOriginConfig": {"HTTPPort": 80}}]},
            }]
        }
    }]
    cf.distribution_config["DefaultCacheBehavior"]["TargetOriginId"] = "legacy-s3-origin"
    cf.distribution_config["Origins"]["Items"][0]["Id"] = "legacy-s3-origin"
    cf.distribution_config["Origins"]["Items"][0]["DomainName"] = (
        f"{module.FRONTEND_BUCKET}.s3.{module.REGION}.amazonaws.com"
    )

    monkeypatch.setattr(module, "get_origin_verify_header", lambda: "x-origin-verify")
    monkeypatch.setattr(module, "get_origin_verify_secret", lambda: "shared-secret")

    url, arn = module.create_cloudfront_distribution("https://abc123.execute-api.us-east-1.amazonaws.com/prod/query")

    assert url == "https://d111.cloudfront.net"
    assert arn == "arn:aws:cloudfront::123:distribution/dist-existing"
    update_call = next(kwargs for name, kwargs in cf.calls if name == "update_distribution")
    assert update_call["DistributionConfig"]["Origins"]["Items"][0]["OriginAccessControlId"] == "oac-new"
    assert update_call["DistributionConfig"]["DefaultCacheBehavior"]["TargetOriginId"] == "s3-frontend"


def test_create_cloudfront_distribution_creates_new_distribution_and_returns_arn(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    cf = FakeCloudFront()
    module = load_script(
        monkeypatch,
        "deploy_frontend_cf_create_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": FakeSTS(), "s3": FakeS3(), "cloudfront": cf},
    )

    monkeypatch.setattr(module, "get_origin_verify_header", lambda: "x-origin-verify")
    monkeypatch.setattr(module, "get_origin_verify_secret", lambda: "shared-secret")

    url, arn = module.create_cloudfront_distribution("https://abc123.execute-api.us-east-1.amazonaws.com/prod/query")

    assert url == "https://d111.cloudfront.net"
    assert arn == "arn:aws:cloudfront::123:distribution/dist-new"
    assert any(name == "create_distribution" for name, _kwargs in cf.calls)
    create_call = next(kwargs for name, kwargs in cf.calls if name == "create_distribution")
    assert create_call["DistributionConfig"]["Origins"]["Quantity"] == 2


def test_main_sets_bucket_policy_using_distribution_arn(monkeypatch, capsys):
    s3 = FakeS3()
    cf = FakeCloudFront()
    sts = FakeSTS()
    module = load_script(
        monkeypatch,
        "deploy_frontend_main_test",
        "scripts/08_deploy_frontend.py",
        boto3_client_map={"sts": sts, "s3": s3, "cloudfront": cf},
    )
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["08_deploy_frontend.py", "https://api.example/query"])
    monkeypatch.setattr(module, "create_frontend_bucket", lambda: None)
    monkeypatch.setattr(module, "upload_frontend", lambda endpoint: None)
    monkeypatch.setattr(module, "create_cloudfront_distribution", lambda endpoint: ("https://d111.cloudfront.net", "arn:aws:cloudfront::123:distribution/dist-arn"))

    module.main()

    out = capsys.readouterr().out
    assert "Frontend deployed!" in out
    put_policy_call = next(kwargs for name, kwargs in s3.calls if name == "put_bucket_policy")
    policy = json.loads(put_policy_call["Policy"])
    assert policy["Statement"][0]["Condition"]["StringEquals"]["AWS:SourceArn"] == "arn:aws:cloudfront::123:distribution/dist-arn"
