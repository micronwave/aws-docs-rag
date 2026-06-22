"""
08_deploy_frontend.py
Deploys the static HTML/JS frontend to S3 and creates a CloudFront distribution.
The browser uses same-origin /query while CloudFront forwards that path to
API Gateway with a private origin-verification header.

Run: python scripts/08_deploy_frontend.py
"""

import os
import sys
import json
import re
import time
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from deploy_config import get_origin_verify_header, get_origin_verify_secret

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
    """Read index.html, inject same-origin API path, and upload to S3."""
    print("  Reading frontend/index.html...")
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # The browser should only call the same-origin CloudFront path.
    html = html.replace("%%API_ENDPOINT%%", api_endpoint)
    print(f"  Injected API endpoint: {api_endpoint}")

    s3.put_object(
        Bucket=FRONTEND_BUCKET,
        Key="index.html",
        Body=html.encode("utf-8"),
        ContentType="text/html",
        CacheControl="max-age=300, must-revalidate",
    )
    print(f"  [OK] Uploaded index.html to s3://{FRONTEND_BUCKET}/")


def parse_api_gateway_origin(api_endpoint: str) -> tuple[str, str]:
    """Return (origin domain, origin path) for an API Gateway invoke URL."""
    parsed = urlparse(api_endpoint)
    expected_host = rf"^[a-z0-9]+\.execute-api\.{re.escape(REGION)}\.amazonaws\.com$"
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(expected_host, parsed.hostname, flags=re.IGNORECASE)
    ):
        raise ValueError(f"Expected an https API endpoint, got: {api_endpoint}")

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) != 2 or path_parts[1] != "query":
        raise ValueError(f"Expected API endpoint ending in /<stage>/query, got: {api_endpoint}")

    return parsed.hostname, f"/{path_parts[0]}"


def build_api_origin(api_endpoint: str) -> dict:
    domain_name, origin_path = parse_api_gateway_origin(api_endpoint)
    return {
        "Id": "api-gateway-query",
        "DomainName": domain_name,
        "OriginPath": origin_path,
        "CustomHeaders": {
            "Quantity": 1,
            "Items": [{
                "HeaderName": get_origin_verify_header(),
                "HeaderValue": get_origin_verify_secret(),
            }],
        },
        "CustomOriginConfig": {
            "HTTPPort": 80,
            "HTTPSPort": 443,
            "OriginProtocolPolicy": "https-only",
            "OriginReadTimeout": 30,
            "OriginKeepaliveTimeout": 5,
            "OriginSslProtocols": {
                "Quantity": 1,
                "Items": ["TLSv1.2"],
            },
        },
    }


def ensure_api_cache_behavior(dist_config: dict) -> bool:
    """Ensure /query routes to API Gateway without caching."""
    cache_behaviors = dist_config.setdefault("CacheBehaviors", {"Quantity": 0, "Items": []})
    items = cache_behaviors.setdefault("Items", [])
    for item in items:
        if item.get("PathPattern") in {"query", "query*"}:
            changed = False
            if item.get("PathPattern") != "query":
                item["PathPattern"] = "query"
                changed = True
            if item.get("TargetOriginId") != "api-gateway-query":
                item["TargetOriginId"] = "api-gateway-query"
                changed = True
            desired_methods = {
                "Quantity": 7,
                "Items": ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
                "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
            }
            if item.get("AllowedMethods") != desired_methods:
                item["AllowedMethods"] = desired_methods
                changed = True
            if item.get("ViewerProtocolPolicy") != "redirect-to-https":
                item["ViewerProtocolPolicy"] = "redirect-to-https"
                changed = True
            desired_forwarded_values = {
                "QueryString": False,
                "Cookies": {"Forward": "none"},
            }
            if item.get("ForwardedValues") != desired_forwarded_values:
                item["ForwardedValues"] = desired_forwarded_values
                changed = True
            for key, value in [("MinTTL", 0), ("DefaultTTL", 0), ("MaxTTL", 0), ("Compress", True)]:
                if item.get(key) != value:
                    item[key] = value
                    changed = True
            if cache_behaviors.get("Quantity") != len(items):
                cache_behaviors["Quantity"] = len(items)
                changed = True
            return changed

    items.append({
        "PathPattern": "query",
        "TargetOriginId": "api-gateway-query",
        "ViewerProtocolPolicy": "redirect-to-https",
        "AllowedMethods": {
            "Quantity": 7,
            "Items": ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
            "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
        },
        "ForwardedValues": {
            "QueryString": False,
            "Cookies": {"Forward": "none"},
        },
        "MinTTL": 0,
        "DefaultTTL": 0,
        "MaxTTL": 0,
        "Compress": True,
    })
    cache_behaviors["Quantity"] = len(items)
    return True


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


def create_cloudfront_distribution(api_endpoint: str) -> tuple[str, str]:
    """Create or update a CloudFront distribution with OAC for S3.

    Returns (cloudfront_url, distribution_arn).
    """
    s3_origin_domain = f"{FRONTEND_BUCKET}.s3.{REGION}.amazonaws.com"
    oac_id = get_or_create_oac()
    api_origin = build_api_origin(api_endpoint)

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
                    config_resp = cf.get_distribution_config(Id=dist_id)
                    dist_config = config_resp["DistributionConfig"]
                    etag = config_resp["ETag"]
                    changed = False

                    # Swap origin to S3OriginConfig + OAC
                    origins = dist_config.setdefault("Origins", {"Quantity": 0, "Items": []})
                    origin_items = origins.setdefault("Items", [])
                    s3_origin = None
                    for origin_item in origin_items:
                        if FRONTEND_BUCKET in origin_item.get("DomainName", "") or origin_item.get("Id") == "s3-frontend":
                            s3_origin = origin_item
                            break

                    if s3_origin is None:
                        s3_origin = {
                            "Id": "s3-frontend",
                            "DomainName": s3_origin_domain,
                            "S3OriginConfig": {"OriginAccessIdentity": ""},
                            "OriginAccessControlId": oac_id,
                        }
                        origin_items.append(s3_origin)
                        changed = True
                    else:
                        previous_s3_origin_id = s3_origin.get("Id")
                        if s3_origin.get("Id") != "s3-frontend":
                            s3_origin["Id"] = "s3-frontend"
                            changed = True
                        if s3_origin.get("DomainName") != s3_origin_domain:
                            s3_origin["DomainName"] = s3_origin_domain
                            changed = True
                        if s3_origin.get("S3OriginConfig") != {"OriginAccessIdentity": ""}:
                            s3_origin["S3OriginConfig"] = {"OriginAccessIdentity": ""}
                            changed = True
                        if s3_origin.get("OriginAccessControlId") != oac_id:
                            s3_origin["OriginAccessControlId"] = oac_id
                            changed = True
                        if "CustomOriginConfig" in s3_origin:
                            s3_origin.pop("CustomOriginConfig", None)
                            changed = True

                        default_behavior = dist_config.get("DefaultCacheBehavior", {})
                        if default_behavior.get("TargetOriginId") == previous_s3_origin_id:
                            default_behavior["TargetOriginId"] = "s3-frontend"
                            changed = True

                    existing_api_origin = None
                    for origin_item in origin_items:
                        if origin_item.get("Id") == "api-gateway-query":
                            existing_api_origin = origin_item
                            break
                    if existing_api_origin != api_origin:
                        if existing_api_origin is None:
                            origin_items.append(api_origin)
                        else:
                            existing_api_origin.clear()
                            existing_api_origin.update(api_origin)
                        changed = True

                    origins["Quantity"] = len(origin_items)
                    if ensure_api_cache_behavior(dist_config):
                        changed = True

                    if changed:
                        print(f"  Updating distribution {dist_id} to enforce OAC and /query proxy...")
                        cf.update_distribution(
                            Id=dist_id,
                            DistributionConfig=dist_config,
                            IfMatch=etag,
                        )
                        print(f"  Distribution updated: {dist_id}")
                    else:
                        print(f"  Distribution already configured: {url}")
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
                "Quantity": 2,
                "Items": [{
                    "Id": "s3-frontend",
                    "DomainName": s3_origin_domain,
                    "S3OriginConfig": {
                        "OriginAccessIdentity": "",  # Empty for OAC (not OAI)
                    },
                    "OriginAccessControlId": oac_id,
                }, api_origin],
            },
            "CacheBehaviors": {
                "Quantity": 1,
                "Items": [{
                    "PathPattern": "query",
                    "TargetOriginId": "api-gateway-query",
                    "ViewerProtocolPolicy": "redirect-to-https",
                    "AllowedMethods": {
                        "Quantity": 7,
                        "Items": ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
                        "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
                    },
                    "ForwardedValues": {
                        "QueryString": False,
                        "Cookies": {"Forward": "none"},
                    },
                    "MinTTL": 0,
                    "DefaultTTL": 0,
                    "MaxTTL": 0,
                    "Compress": True,
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
        print("\n  [ERR] No API endpoint provided!")
        print("  Usage: python scripts/08_deploy_frontend.py [endpoint_url]")
        print("  Or run 07_deploy_api_gateway.py first to create api_endpoint.txt")
        return
    print(f"\n  API endpoint: {api_endpoint}")

    # Create bucket
    print("\n[1/3] Setting up S3 bucket...")
    create_frontend_bucket()

    # Upload HTML
    print("\n[2/3] Uploading frontend...")
    upload_frontend("/query")

    # CloudFront
    print("\n[3/3] Setting up CloudFront...")
    cloudfront_url, dist_arn = create_cloudfront_distribution(api_endpoint)

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
