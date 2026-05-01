"""
07_deploy_api_gateway.py
Creates an API Gateway REST API with a /query POST endpoint,
connects it to the Lambda function, enables CORS, and deploys.

Run: python scripts/07_deploy_api_gateway.py
"""

import os
import json
import sys
import boto3
from botocore.exceptions import ClientError

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from deploy_config import get_allowed_origin

# ─── Configuration ────────────────────────────────────────────────────
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
FUNCTION_NAME = "aws-rag-query"
API_NAME = "aws-rag-api"
STAGE_NAME = "prod"

apigw = boto3.client("apigateway", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)
sts = boto3.client("sts", region_name=REGION)

ACCOUNT_ID = sts.get_caller_identity()["Account"]


def get_or_create_api() -> str:
    """Get existing API or create new one. Returns API ID."""
    # Check if API already exists (paginated)
    paginator = apigw.get_paginator("get_rest_apis")
    for page in paginator.paginate():
        for api in page["items"]:
            if api["name"] == API_NAME:
                print(f"  API already exists: {api['id']}")
                return api["id"]

    # Create new API
    print(f"  Creating REST API '{API_NAME}'...")
    resp = apigw.create_rest_api(
        name=API_NAME,
        description="RAG system query API",
        endpointConfiguration={"types": ["REGIONAL"]},
    )
    api_id = resp["id"]
    print(f"  [OK] Created API: {api_id}")
    return api_id


def get_root_resource_id(api_id: str) -> str:
    """Get the root (/) resource ID."""
    resources = apigw.get_resources(restApiId=api_id)["items"]
    for r in resources:
        if r["path"] == "/":
            return r["id"]
    raise Exception("Root resource not found")


def get_or_create_resource(api_id: str, parent_id: str, path_part: str) -> str:
    """Get existing resource or create it. Returns resource ID."""
    resources = apigw.get_resources(restApiId=api_id)["items"]
    for r in resources:
        if r.get("pathPart") == path_part:
            print(f"  Resource /{path_part} already exists: {r['id']}")
            return r["id"]

    resp = apigw.create_resource(
        restApiId=api_id,
        parentId=parent_id,
        pathPart=path_part,
    )
    print(f"  [OK] Created resource /{path_part}: {resp['id']}")
    return resp["id"]


def setup_method(api_id: str, resource_id: str, http_method: str, lambda_arn: str, is_cors: bool = False):
    """Set up a method on the resource with Lambda integration."""
    # Delete existing method if it exists
    try:
        apigw.delete_method(restApiId=api_id, resourceId=resource_id, httpMethod=http_method)
    except ClientError:
        pass

    # Create method
    apigw.put_method(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod=http_method,
        authorizationType="NONE",
        apiKeyRequired=not is_cors,
    )

    if is_cors:
        # OPTIONS method returns CORS headers directly (mock integration)
        allowed_origin = get_allowed_origin()
        apigw.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod=http_method,
            type="MOCK",
            requestTemplates={"application/json": '{"statusCode": 200}'},
        )
        apigw.put_method_response(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod=http_method,
            statusCode="200",
            responseParameters={
                "method.response.header.Access-Control-Allow-Headers": False,
                "method.response.header.Access-Control-Allow-Methods": False,
                "method.response.header.Access-Control-Allow-Origin": False,
            },
        )
        apigw.put_integration_response(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod=http_method,
            statusCode="200",
            responseParameters={
                "method.response.header.Access-Control-Allow-Headers": "'Content-Type,x-api-key'",
                "method.response.header.Access-Control-Allow-Methods": "'POST,OPTIONS'",
                "method.response.header.Access-Control-Allow-Origin": f"'{allowed_origin}'",
            },
            responseTemplates={"application/json": ""},
        )
        print(f"  [OK] OPTIONS method configured (CORS)")
    else:
        # POST method -> Lambda proxy integration
        lambda_uri = (
            f"arn:aws:apigateway:{REGION}:lambda:path"
            f"/2015-03-31/functions/{lambda_arn}/invocations"
        )
        apigw.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod=http_method,
            type="AWS_PROXY",
            integrationHttpMethod="POST",
            uri=lambda_uri,
        )
        print(f"  [OK] POST method configured -> Lambda proxy")


def add_lambda_permission(api_id: str, lambda_arn: str):
    """Allow API Gateway to invoke the Lambda function."""
    statement_id = "apigateway-invoke-lambda"

    try:
        lambda_client.remove_permission(
            FunctionName=FUNCTION_NAME,
            StatementId=statement_id,
        )
    except ClientError:
        pass

    source_arn = f"arn:aws:execute-api:{REGION}:{ACCOUNT_ID}:{api_id}/*"

    lambda_client.add_permission(
        FunctionName=FUNCTION_NAME,
        StatementId=statement_id,
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=source_arn,
    )
    print(f"  [OK] Lambda invoke permission granted to API Gateway")


# API Gateway error response types that need CORS headers so the browser
# can read the error instead of showing a generic "Failed to fetch".
GATEWAY_ERROR_TYPES = [
    "MISSING_AUTHENTICATION_TOKEN",
    "ACCESS_DENIED",
    "QUOTA_EXCEEDED",
    "THROTTLED",
    "API_CONFIGURATION_ERROR",
    "DEFAULT_4XX",
    "DEFAULT_5XX",
]


def setup_gateway_responses(api_id: str) -> None:
    """Add CORS headers to API Gateway-generated error responses."""
    cors_origin = get_allowed_origin()
    for response_type in GATEWAY_ERROR_TYPES:
        apigw.put_gateway_response(
            restApiId=api_id,
            responseType=response_type,
            responseParameters={
                "gatewayresponse.header.Access-Control-Allow-Origin": f"'{cors_origin}'",
                "gatewayresponse.header.Access-Control-Allow-Headers": "'Content-Type'",
            },
        )
    print(f"  [OK] CORS headers added to {len(GATEWAY_ERROR_TYPES)} gateway error responses")


def deploy_api(api_id: str) -> str:
    """Deploy the API to a stage and return the invoke URL."""
    apigw.create_deployment(
        restApiId=api_id,
        stageName=STAGE_NAME,
        description="RAG API deployment",
    )

    invoke_url = f"https://{api_id}.execute-api.{REGION}.amazonaws.com/{STAGE_NAME}"
    print(f"  [OK] Deployed to stage '{STAGE_NAME}'")
    return invoke_url


def create_usage_plan(api_id: str) -> str:
    """Create/repair usage plan + API key wiring and return key value."""
    try:
        keys = apigw.get_api_keys(nameQuery="aws-rag-key", includeValues=True).get("items", [])
        if keys:
            api_key_id = keys[0]["id"]
            api_key_value = keys[0]["value"]
            print(f"  API key already exists: {api_key_id}")
        else:
            raise KeyError("aws-rag-key not found")
    except (ClientError, KeyError):
        resp = apigw.create_api_key(
            name="aws-rag-key",
            description="API key for RAG frontend",
            enabled=True,
        )
        api_key_id = resp["id"]
        api_key_value = resp["value"]
        print(f"  [OK] Created API key: {api_key_id}")

    plans = apigw.get_usage_plans().get("items", [])
    plan_id = None
    for plan in plans:
        if plan["name"] == "aws-rag-plan":
            plan_id = plan["id"]
            print(f"  Usage plan already exists: {plan_id}")
            api_stages = plan.get("apiStages", [])
            has_stage = any(
                stage.get("apiId") == api_id and stage.get("stage") == STAGE_NAME
                for stage in api_stages
            )
            if not has_stage:
                apigw.update_usage_plan(
                    usagePlanId=plan_id,
                    patchOperations=[
                        {
                            "op": "add",
                            "path": "/apiStages",
                            "value": f"{api_id}:{STAGE_NAME}",
                        }
                    ],
                )
                print(f"  [OK] Re-attached usage plan to {api_id}:{STAGE_NAME}")
            break

    if not plan_id:
        resp = apigw.create_usage_plan(
            name="aws-rag-plan",
            description="Rate-limited plan for RAG API",
            throttle={"rateLimit": 5, "burstLimit": 10},
            quota={"limit": 1000, "period": "DAY"},
            apiStages=[{"apiId": api_id, "stage": STAGE_NAME}],
        )
        plan_id = resp["id"]
        print(f"  [OK] Created usage plan: {plan_id}")

    usage_plan_keys = apigw.get_usage_plan_keys(usagePlanId=plan_id).get("items", [])
    if any(item.get("id") == api_key_id for item in usage_plan_keys):
        print("  API key already linked to usage plan")
    else:
        apigw.create_usage_plan_key(
            usagePlanId=plan_id,
            keyId=api_key_id,
            keyType="API_KEY",
        )
        print("  [OK] Linked API key to usage plan")

    return api_key_value


def main():
    print(f"\n{'='*60}")
    print(f"  Deploying API Gateway: {API_NAME}")
    print(f"{'='*60}")

    # Get Lambda ARN
    print("\n[1/8] Getting Lambda function ARN...")
    func = lambda_client.get_function(FunctionName=FUNCTION_NAME)
    lambda_arn = func["Configuration"]["FunctionArn"]
    print(f"  Lambda ARN: {lambda_arn}")

    # Create/get API
    print("\n[2/8] Creating REST API...")
    api_id = get_or_create_api()

    # Create /query resource
    print("\n[3/8] Creating /query resource...")
    root_id = get_root_resource_id(api_id)
    query_resource_id = get_or_create_resource(api_id, root_id, "query")

    # Set up POST and OPTIONS methods
    print("\n[4/8] Configuring methods...")
    setup_method(api_id, query_resource_id, "POST", lambda_arn, is_cors=False)
    setup_method(api_id, query_resource_id, "OPTIONS", lambda_arn, is_cors=True)

    # Grant API Gateway permission to invoke Lambda
    print("\n[5/8] Granting permissions...")
    add_lambda_permission(api_id, lambda_arn)

    # Add CORS headers to gateway error responses (403, 429, etc.)
    print("\n[6/8] Configuring gateway error responses...")
    setup_gateway_responses(api_id)

    # Deploy
    print("\n[7/8] Deploying API...")
    invoke_url = deploy_api(api_id)

    print("\n[8/8] Creating usage plan and API key...")
    api_key_value = create_usage_plan(api_id)

    endpoint = f"{invoke_url}/query"

    print(f"\n{'='*60}")
    print(f"  [OK] API Gateway deployed successfully!")
    print(f"  Endpoint: {endpoint}")
    print(f"{'='*60}")
    print(f"\n  Test with curl:")
    print(f"  curl -X POST {endpoint} \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -H 'x-api-key: {api_key_value}' \\")
    print(f"    -d '{{\"question\": \"How do I create an S3 bucket?\"}}'")
    print()

    # Save the endpoint URL for the frontend deployment script
    with open("api_endpoint.txt", "w") as f:
        f.write(endpoint)
    print(f"  Endpoint saved to api_endpoint.txt\n")

    with open("api_key.txt", "w") as f:
        f.write(api_key_value)
    print("  API key saved to api_key.txt\n")


if __name__ == "__main__":
    main()
