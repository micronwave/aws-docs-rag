# AWS Docs RAG Deployment Guide

This guide covers the full deployment path for the repository cloned as
`aws-docs-rag`.

## Prerequisites

- AWS account with Bedrock model access
- Pinecone account and API key
- AWS CLI v2
- Python 3.11+
- A frontend origin URL that will be allowed by CORS

## Clone the repo

```bash
git clone https://github.com/micronwave/aws-docs-rag.git
cd aws-docs-rag
```

## Set environment variables

Set the project variables before running the deploy scripts. `ALLOWED_ORIGIN`
must match the CloudFront URL that serves the frontend.

`ORIGIN_VERIFY_SECRET` is optional. If you do not set it, the deploy helpers
generate a strong secret and store it in `origin_verify_secret.txt` so Lambda
and CloudFront share the same value without exposing it to the browser.

Example:

```bash
export ALLOWED_ORIGIN=https://your-cloudfront-domain.example.com
source set_env.sh
```

Windows users can set the same value in `set_env.template.ps1` or their local
copy of that file before running the deployment scripts.

## Deploy order

Run these scripts in order:

1. `python scripts/01_ingest_docs.py`
2. `python scripts/02_chunk_docs.py`
3. `python scripts/03_generate_embeddings.py`
4. `python scripts/04_upload_to_pinecone.py`
5. `python scripts/05_test_rag_local.py`
6. `python scripts/06_deploy_lambda.py`
7. `python scripts/07_deploy_api_gateway.py`
8. `python scripts/08_deploy_frontend.py`

## CORS behavior

`scripts/06_deploy_lambda.py` injects `ALLOWED_ORIGIN` into the Lambda
environment, and `scripts/07_deploy_api_gateway.py` uses the same value for
OPTIONS and gateway error responses. `scripts/08_deploy_frontend.py` publishes
the frontend with a same-origin `/query` endpoint and configures CloudFront to
forward that path to API Gateway with a private verification header. If the
frontend URL changes, update the environment variable and rerun the Lambda and
API Gateway scripts.
