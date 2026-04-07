# AWS Documentation RAG System

A Retrieval Augmented Generation (RAG) system that answers questions about AWS services by searching real AWS documentation and generates responses using Claude.

**Live demo:** [https://d3d0zch3u8ca61.cloudfront.net](https://d3d0zch3u8ca61.cloudfront.net)

---

## What It Does

Ask a question about AWS → the system searches through indexed AWS documentation → retrieves the most relevant sections → sends them to Claude → returns an answer grounded in real documentation with source links.

Every answer is backed by retrieved documentation.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   INGESTION (one-time)                       │
│                                                              │
│    AWS Docs → Scrape & Clean → Chunk → Embed (Titan v2) →    |                               
│                Upload to S3 → Upload to Pinecone             │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   QUERY (per question)                       │
│                                                              │
│  User → CloudFront → API Gateway → Lambda                    │
│                                      ├─ Embed question       │
│                                      ├─ Search Pinecone      │
│                                      ├─ Build prompt         │
│                                      ├─ Call Claude          │
│                                      └─ Return answer        │
└──────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Service |
|-----------|---------|
| **LLM** | Claude Sonnet 4.6 via Amazon Bedrock |
| **Embeddings** | Amazon Titan Embeddings v2 (1024-dim) |
| **Vector DB** | Pinecone (free tier) |
| **Backend** | AWS Lambda + API Gateway (REST) |
| **Frontend** | Static HTML/JS on S3 + CloudFront |
| **Storage** | Amazon S3 |
| **Monitoring** | Amazon CloudWatch |
| **Language** | Python 3.11 |

---

## Project Structure

```
aws-rag-project/
├── scripts/
│   ├── 01_ingest_docs.py          # Scrape & clean AWS documentation
│   ├── 02_chunk_docs.py           # Split docs into overlapping chunks
│   ├── 03_generate_embeddings.py  # Generate vectors via Titan v2
│   ├── 04_upload_to_pinecone.py   # Create index & upload vectors
│   ├── 05_test_rag_local.py       # Test full RAG pipeline locally
│   ├── 06_deploy_lambda.py        # Package & deploy Lambda function
│   ├── 07_deploy_api_gateway.py   # Create REST API endpoint
│   └── 08_deploy_frontend.py      # Deploy static site to S3 + CloudFront
├── lambda_function/
│   └── lambda_handler.py          # Lambda handler (embed → search → generate)
├── frontend/
│   └── index.html                 # Chat UI (single file, no framework)
├── set_env.sh                     # Environment variables (Linux/macOS)
├── set_env.ps1                    # Environment variables (Windows)
├── requirements.txt               # Python dependencies
├── GUIDE.md                       # Complete step-by-step build guide
└── README.md
```

---

### Ingestion Pipeline (run once)

1. **Scrape** — Downloads user guide pages for S3, EC2, Lambda, DynamoDB, and VPC from docs.aws.amazon.com
2. **Clean** — Strips HTML chrome (nav, footers, scripts), keeps documentation content
3. **Chunk** — Splits documents into ~1000-character pieces with 200-character overlap using LangChain's RecursiveCharacterTextSplitter
4. **Embed** — Sends each chunk to Amazon Titan Embeddings v2 → 1024-dimensional vector
5. **Store** — Uploads vectors + metadata to Pinecone with cosine similarity indexing

### Query Pipeline (every question)

1. **Embed** the user's question using the same Titan v2 model
2. **Search** Pinecone for the 5 most semantically similar document chunks
3. **Build** a prompt with the question + retrieved chunks + anti-hallucination instructions
4. **Generate** an answer using Claude via Bedrock
5. **Return** the answer with source URLs

---

## Setup & Deployment

### Prerequisites

- AWS account with Bedrock model access
- Pinecone account (free tier)
- Python 3.11+
- AWS CLI v2

### Quick Start

```bash
# Clone the repo
git clone https://github.com/micronwave/aws-rag-project.git
cd aws-rag-project

# Set up Python environment
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Configure environment variables
cp set_env.sh set_env.local.sh    # Edit with your Pinecone key + AWS account ID
source set_env.local.sh

# Create S3 bucket
aws s3 mb s3://$S3_BUCKET_NAME --region us-east-2

# Run the pipeline
python scripts/01_ingest_docs.py          # ~5 min
python scripts/02_chunk_docs.py           # ~30 sec
python scripts/03_generate_embeddings.py  # ~10 min
python scripts/04_upload_to_pinecone.py   # ~1 min

# Test locally
python scripts/05_test_rag_local.py "How do I create an S3 bucket?"

# Deploy
python scripts/06_deploy_lambda.py
python scripts/07_deploy_api_gateway.py
python scripts/08_deploy_frontend.py

---

## Cost

| Service | Monthly Cost |
|---------|-------------|
| Pinecone | $0 (free tier) |
| Lambda | $0 (free tier) |
| API Gateway | $0 (free tier, first 12 months) |
| S3 | ~$0.03 |
| CloudFront | ~$0.50 |
| Bedrock (Claude) | ~$2–10 depending on usage |
| **Total** | **~$3–11/month** |

Compared to ~$700+/month if using OpenSearch Serverless as the vector database.

---

## Security

- Lambda runs with a **scoped IAM policy** — only `bedrock:InvokeModel` and CloudWatch log permissions
- Pinecone API key stored as a **Lambda environment variable**
- API Gateway handles CORS and request validation
- S3 frontend bucket served through **CloudFront with HTTPS**
- No user data is stored — queries are stateless

---

## Scaling Path

| Need | Solution |
|------|----------|
| More documentation | Add service URLs to `01_ingest_docs.py`, re-run pipeline |
| Faster cold starts | Lambda Provisioned Concurrency or move to ECS Fargate |
| Repeated query caching | Add DynamoDB TTL cache in front of Bedrock |
| Better retrieval | Add reranking step (retrieve top 20, rerank to top 5) |
| Cheaper generation | Swap Claude Sonnet for Claude Haiku on simple queries |
| User authentication | Add Amazon Cognito |

---
