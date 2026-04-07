"""
08_deploy_frontend.py
Deploys the static HTML/JS frontend to S3 and creates a CloudFront distribution.
Reads the API endpoint from api_endpoint.txt and injects it into the HTML.

Run: python scripts/08_deploy_frontend.py
"""

import os
import sys
import json
import time
import boto3
from botocore.exceptions import ClientError

# ─── Configuration ────────────────────────────────────────────────────
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
FRONTEND_BUCKET = f"aws-rag-frontend-{ACCOUNT_ID}"

s3 = boto3.client("s3", region_name=REGION)
cf = boto3.client("cloudfront", region_name=REGION)


def create_frontend_bucket() -> None:
    """Create S3 bucket for static website hosting."""
    try:
        s3.head_bucket(Bucket=FRONTEND_BUCKET)
        print(f"  Bucket already exists: {FRONTEND_BUCKET}")
    except ClientError:
        print(f"  Creating bucket: {FRONTEND_BUCKET}")
        # us-east-1 doesn't need LocationConstraint
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=FRONTEND_BUCKET)
        else:
            s3.create_bucket(
                Bucket=FRONTEND_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )

    # Enable static website hosting
    s3.put_bucket_website(
        Bucket=FRONTEND_BUCKET,
        WebsiteConfiguration={
            "IndexDocument": {"Suffix": "index.html"},
            "ErrorDocument": {"Key": "index.html"},
        },
    )

    # Block all public access — only CloudFront OAC can read
    s3.put_public_access_block(
        Bucket=FRONTEND_BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print(f"  Public access blocked (CloudFront OAC will be used)")


def upload_frontend(api_endpoint: str) -> None:
    """Read index.html, inject the API endpoint, and upload to S3."""
    print("  Reading frontend/index.html...")
    with open("frontend/index.html", "r") as f:
        html = f.read()

    # Replace placeholder with actual endpoint
    html = html.replace("%%API_ENDPOINT%%", api_endpoint)
    print(f"  Injected API endpoint: {api_endpoint}")

    # Replace API key placeholder
    if os.path.exists("api_key.txt"):
        with open("api_key.txt") as f:
            api_key = f.read().strip()
        html = html.replace("%%API_KEY%%", api_key)
        print(f"  Injected API key")
    else:
        print("  WARNING: api_key.txt not found — API key placeholder will not be replaced.")
        print("    Run 07_deploy_api_gateway.py first to generate the API key.")

    s3.put_object(
        Bucket=FRONTEND_BUCKET,
        Key="index.html",
        Body=html.encode("utf-8"),
        ContentType="text/html",
    )
    print(f"  ✓ Uploaded index.html to s3://{FRONTEND_BUCKET}/")


def get_or_create_oac() -> str:
    """Create a CloudFront Origin Access Control for S3."""
    oac_name = "aws-rag-frontend-oac"

    # Check existing OACs
    try:
        oacs = cf.list_origin_access_controls()
        for item in oacs.get("OriginAccessControlList", {}).get("Items", []):
            if item["Name"] == oac_name:
                print(f"  OAC already exists: {item['Id']}")
                return item["Id"]
    except Exception:
        pass

    resp = cf.create_origin_access_control(
        OriginAccessControlConfig={
            "Name": oac_name,
            "Description": "OAC for RAG frontend S3 bucket",
            "SigningProtocol": "sigv4",
            "SigningBehavior": "always",
            "OriginAccessControlOriginType": "s3",
        }
    )
    oac_id = resp["OriginAccessControl"]["Id"]
    print(f"  Created OAC: {oac_id}")
    return oac_id


def create_cloudfront_distribution() -> tuple[str, str]:
    """Create or update a CloudFront distribution with OAC for S3.

    Returns (cloudfront_url, distribution_arn).
    """
    s3_origin_domain = f"{FRONTEND_BUCKET}.s3.{REGION}.amazonaws.com"
    oac_id = get_or_create_oac()

    # Check if distribution already exists for this bucket
    paginator = cf.get_paginator("list_distributions")
    for page in paginator.paginate():
        dist_list = page.get("DistributionList", {})
        for dist in dist_list.get("Items", []):
            for origin in dist["Origins"]["Items"]:
                if FRONTEND_BUCKET in origin.get("DomainName", ""):
                    dist_id = dist["Id"]
                    dist_arn = dist["ARN"]
                    url = f"https://{dist['DomainName']}"

                    # Already using OAC — nothing to do
                    if origin.get("S3OriginConfig") is not None:
                        print(f"  Distribution already using OAC: {url}")
                        return url, dist_arn

                    # Existing distribution still on CustomOriginConfig — update it
                    print(f"  Updating distribution {dist_id} to use OAC...")
                    config_resp = cf.get_distribution_config(Id=dist_id)
                    dist_config = config_resp["DistributionConfig"]
                    etag = config_resp["ETag"]

                    # Swap origin to S3OriginConfig + OAC
                    origin_item = dist_config["Origins"]["Items"][0]
                    origin_item["DomainName"] = s3_origin_domain
                    origin_item.pop("CustomOriginConfig", None)
                    origin_item["S3OriginConfig"] = {
                        "OriginAccessIdentity": "",  # Empty for OAC (not OAI)
                    }
                    origin_item["OriginAccessControlId"] = oac_id

                    cf.update_distribution(
                        Id=dist_id,
                        DistributionConfig=dist_config,
                        IfMatch=etag,
                    )
                    print(f"  Distribution updated: {dist_id}")
                    return url, dist_arn

    # No existing distribution — create a new one
    print("  Creating CloudFront distribution...")
    caller_ref = str(int(time.time()))

    resp = cf.create_distribution(
        DistributionConfig={
            "CallerReference": caller_ref,
            "Comment": "AWS RAG Frontend",
            "DefaultCacheBehavior": {
                "TargetOriginId": "s3-frontend",
                "ViewerProtocolPolicy": "redirect-to-https",
                "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
                "ForwardedValues": {
                    "QueryString": False,
                    "Cookies": {"Forward": "none"},
                },
                "MinTTL": 0,
                "DefaultTTL": 86400,
                "MaxTTL": 31536000,
                "Compress": True,
            },
            "Origins": {
                "Quantity": 1,
                "Items": [{
                    "Id": "s3-frontend",
                    "DomainName": s3_origin_domain,
                    "S3OriginConfig": {
                        "OriginAccessIdentity": "",  # Empty for OAC (not OAI)
                    },
                    "OriginAccessControlId": oac_id,
                }],
            },
            "Enabled": True,
            "DefaultRootObject": "index.html",
            "PriceClass": "PriceClass_100",  # US, Canada, Europe only (cheapest)
        },
    )

    domain = resp["Distribution"]["DomainName"]
    dist_id = resp["Distribution"]["Id"]
    dist_arn = resp["Distribution"]["ARN"]
    url = f"https://{domain}"

    print(f"  Distribution created: {dist_id}")
    print(f"  CloudFront takes 5-15 minutes to fully deploy.")
    return url, dist_arn


def main():
    print(f"\n{'='*60}")
    print(f"  Deploying Frontend")
    print(f"{'='*60}")

    # Read API endpoint from argument or file
    if len(sys.argv) > 1:
        api_endpoint = sys.argv[1]
    elif os.path.exists("api_endpoint.txt"):
        with open("api_endpoint.txt") as f:
            api_endpoint = f.read().strip()
    else:
        print("\n  ✗ No API endpoint provided!")
        print("  Usage: python scripts/08_deploy_frontend.py [endpoint_url]")
        print("  Or run 07_deploy_api_gateway.py first to create api_endpoint.txt")
        return
    print(f"\n  API endpoint: {api_endpoint}")

    # Create bucket
    print("\n[1/3] Setting up S3 bucket...")
    create_frontend_bucket()

    # Upload HTML
    print("\n[2/3] Uploading frontend...")
    upload_frontend(api_endpoint)

    # CloudFront
    print("\n[3/3] Setting up CloudFront...")
    cloudfront_url, dist_arn = create_cloudfront_distribution()

    # Set bucket policy — allows only this CloudFront distribution via OAC
    oac_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowCloudFrontOAC",
            "Effect": "Allow",
            "Principal": {"Service": "cloudfront.amazonaws.com"},
            "Action": "s3:GetObject",
            "Resource": f"arn:aws:s3:::{FRONTEND_BUCKET}/*",
            "Condition": {
                "StringEquals": {
                    "AWS:SourceArn": dist_arn
                }
            }
        }]
    })
    s3.put_bucket_policy(Bucket=FRONTEND_BUCKET, Policy=oac_policy)
    print(f"  Bucket policy set for CloudFront OAC")

    print(f"\n{'='*60}")
    print(f"  Frontend deployed!")
    print(f"")
    print(f"  CloudFront (HTTPS):")
    print(f"    {cloudfront_url}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
