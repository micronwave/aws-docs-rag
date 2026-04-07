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

# Disable block public access (required for public bucket policy)
    s3.put_public_access_block(
        Bucket=FRONTEND_BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": False,
            "IgnorePublicAcls": False,
            "BlockPublicPolicy": False,
            "RestrictPublicBuckets": False,
        },
    )
    print(f"  ✓ Bucket configured for static hosting")

    # Set bucket policy for public read (CloudFront OAC is better but more complex)
    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "PublicRead",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": f"arn:aws:s3:::{FRONTEND_BUCKET}/*",
        }],
    })
    s3.put_bucket_policy(Bucket=FRONTEND_BUCKET, Policy=policy)


def upload_frontend(api_endpoint: str) -> None:
    """Read index.html, inject the API endpoint, and upload to S3."""
    print("  Reading frontend/index.html...")
    with open("frontend/index.html", "r") as f:
        html = f.read()

    # Replace placeholder with actual endpoint
    html = html.replace("%%API_ENDPOINT%%", api_endpoint)

    print(f"  Injected API endpoint: {api_endpoint}")

    s3.put_object(
        Bucket=FRONTEND_BUCKET,
        Key="index.html",
        Body=html.encode("utf-8"),
        ContentType="text/html",
    )
    print(f"  ✓ Uploaded index.html to s3://{FRONTEND_BUCKET}/")


def create_cloudfront_distribution() -> str:
    """Create a CloudFront distribution pointing to the S3 website."""
    # Check if distribution already exists for this bucket
    origin_domain = f"{FRONTEND_BUCKET}.s3-website-{REGION}.amazonaws.com"

    paginator = cf.get_paginator("list_distributions")
    for page in paginator.paginate():
        dist_list = page.get("DistributionList", {})
        for dist in dist_list.get("Items", []):
            for origin in dist["Origins"]["Items"]:
                if FRONTEND_BUCKET in origin.get("DomainName", ""):
                    url = f"https://{dist['DomainName']}"
                    print(f"  CloudFront distribution already exists: {url}")
                    return url

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
                    "DomainName": origin_domain,
                    "CustomOriginConfig": {
                        "HTTPPort": 80,
                        "HTTPSPort": 443,
                        "OriginProtocolPolicy": "http-only",
                    },
                }],
            },
            "Enabled": True,
            "DefaultRootObject": "index.html",
            "PriceClass": "PriceClass_100",  # US, Canada, Europe only (cheapest)
        },
    )

    domain = resp["Distribution"]["DomainName"]
    dist_id = resp["Distribution"]["Id"]
    url = f"https://{domain}"

    print(f"  ✓ Distribution created: {dist_id}")
    print(f"  ⏳ CloudFront takes 5-15 minutes to fully deploy.")
    return url


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
    cloudfront_url = create_cloudfront_distribution()

    # Also show direct S3 website URL (works immediately)
    s3_url = f"http://{FRONTEND_BUCKET}.s3-website-{REGION}.amazonaws.com"

    print(f"\n{'='*60}")
    print(f"  ✓ Frontend deployed!")
    print(f"")
    print(f"  S3 Website (available now):")
    print(f"    {s3_url}")
    print(f"")
    print(f"  CloudFront (HTTPS, available in 5-15 min):")
    print(f"    {cloudfront_url}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
