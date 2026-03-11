"""
06_deploy_lambda.py
Packages the Lambda function with dependencies and deploys it to AWS.
Also creates the IAM execution role if it doesn't exist.

Run: python scripts/06_deploy_lambda.py
"""

import os
import json
import time
import subprocess
import shutil
import zipfile
import boto3
from botocore.exceptions import ClientError

# ─── Configuration ────────────────────────────────────────────────────
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
FUNCTION_NAME = "aws-rag-query"
ROLE_NAME = "aws-rag-lambda-role"
HANDLER = "lambda_handler.lambda_handler"
RUNTIME = "python3.11"
TIMEOUT = 60       # seconds — Claude can take a moment to respond
MEMORY = 512       # MB
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "aws-rag-index")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")

iam = boto3.client("iam", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)
sts = boto3.client("sts", region_name=REGION)

ACCOUNT_ID = sts.get_caller_identity()["Account"]

# Trust policy — allows Lambda service to assume this role
TRUST_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
})


def create_role() -> str:
    """Create the Lambda execution role with Bedrock + CloudWatch access."""
    try:
        resp = iam.get_role(RoleName=ROLE_NAME)
        role_arn = resp["Role"]["Arn"]
        print(f"  Role already exists: {role_arn}")
        return role_arn
    except ClientError:
        pass

    print(f"  Creating IAM role '{ROLE_NAME}'...")
    resp = iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=TRUST_POLICY,
        Description="Execution role for RAG Lambda function",
    )
    role_arn = resp["Role"]["Arn"]

    # Attach policies
    policies = [
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",  # CloudWatch logs
        "arn:aws:iam::aws:policy/AmazonBedrockFullAccess",                   # Bedrock calls
    ]
    for policy_arn in policies:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
        print(f"    Attached: {policy_arn.split('/')[-1]}")

    # IAM roles need a few seconds to propagate
    print("  Waiting 10s for role to propagate...")
    time.sleep(10)

    return role_arn


def package_lambda() -> str:
    """
    Package the Lambda function + dependencies into a ZIP file.
    Installs pinecone-client into a temp directory and bundles it.
    """
    build_dir = "/tmp/lambda-build"
    zip_path = "/tmp/lambda-deployment.zip"

    # Clean up any previous build
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    os.makedirs(build_dir)

    # Install orjson first (needs Linux binary)
    subprocess.run(
        [
            "pip", "install",
            "orjson",
            "--platform", "manylinux2014_x86_64",
            "--implementation", "cp",
            "--python-version", "3.11",
            "--only-binary=:all:",
            "-t", build_dir,
            "--quiet",
            "--no-cache-dir",
        ],
        check=True,
    )

    print("  Installing dependencies into build directory...")
    subprocess.run(
        [
            "pip", "install",
            "pinecone",
            "--platform", "manylinux2014_x86_64",
            "--implementation", "cp",
            "--python-version", "3.11",
            "--only-binary=:all:",
            "-t", build_dir,
            "--quiet",
            "--no-cache-dir",
        ],
        check=True,
    )

    # Copy the Lambda handler into the build directory
    shutil.copy("lambda_function/lambda_handler.py", build_dir)

    # Create the ZIP
    print("  Creating deployment ZIP...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(build_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, build_dir)
                zf.write(filepath, arcname)

    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"  ✓ Package size: {size_mb:.1f} MB")

    if size_mb > 50:
        print("  ⚠ Warning: Package > 50MB. Consider using a Lambda Layer instead.")

    return zip_path


def deploy_function(role_arn: str, zip_path: str) -> str:
    """Create or update the Lambda function."""
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    env_vars = {
        "APP_REGION": REGION,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "PINECONE_INDEX_NAME": INDEX_NAME,
        "EMBEDDING_MODEL_ID": EMBEDDING_MODEL_ID,
        "LLM_MODEL_ID": LLM_MODEL_ID,
    }

    try:
        # Try to update existing function
        lambda_client.update_function_code(
            FunctionName=FUNCTION_NAME,
            ZipFile=zip_bytes,
        )
        # Wait for update to complete
        print(f"  Updating existing function '{FUNCTION_NAME}'...")
        waiter = lambda_client.get_waiter("function_updated_v2")
        waiter.wait(FunctionName=FUNCTION_NAME)

        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Handler=HANDLER,
            Runtime=RUNTIME,
            Timeout=TIMEOUT,
            MemorySize=MEMORY,
            Environment={"Variables": env_vars},
        )
        waiter.wait(FunctionName=FUNCTION_NAME)

        resp = lambda_client.get_function(FunctionName=FUNCTION_NAME)
        func_arn = resp["Configuration"]["FunctionArn"]
        print(f"  ✓ Updated: {func_arn}")

    except ClientError as e:
        if "ResourceNotFoundException" in str(e):
            # Create new function
            print(f"  Creating function '{FUNCTION_NAME}'...")
            resp = lambda_client.create_function(
                FunctionName=FUNCTION_NAME,
                Runtime=RUNTIME,
                Role=role_arn,
                Handler=HANDLER,
                Code={"ZipFile": zip_bytes},
                Timeout=TIMEOUT,
                MemorySize=MEMORY,
                Environment={"Variables": env_vars},
                Description="RAG query handler - embeds question, searches Pinecone, calls Claude",
            )
            func_arn = resp["FunctionArn"]

            # Wait for function to be active
            print("  Waiting for function to become active...")
            waiter = lambda_client.get_waiter("function_active_v2")
            waiter.wait(FunctionName=FUNCTION_NAME)
            print(f"  ✓ Created: {func_arn}")
        else:
            raise

    return func_arn


def main():
    print(f"\n{'='*60}")
    print(f"  Deploying Lambda: {FUNCTION_NAME}")
    print(f"{'='*60}")

    print("\n[1/3] Creating IAM execution role...")
    role_arn = create_role()

    print("\n[2/3] Packaging Lambda function...")
    zip_path = package_lambda()

    print("\n[3/3] Deploying to AWS Lambda...")
    func_arn = deploy_function(role_arn, zip_path)

    print(f"\n{'='*60}")
    print(f"  ✓ Lambda deployed successfully!")
    print(f"  Function: {FUNCTION_NAME}")
    print(f"  ARN:      {func_arn}")
    print(f"  Region:   {REGION}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
