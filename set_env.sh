#!/bin/bash
# ============================================================
# set_env.sh — Set all environment variables for the RAG project
# 
# USAGE:
#   1. Edit the PINECONE_API_KEY line below with your real key
#   2. Run: source set_env.sh
# ============================================================

# AWS region — us-east-1 has best Bedrock model availability
export AWS_DEFAULT_REGION=us-east-1

# S3 bucket — includes your account ID for global uniqueness
export S3_BUCKET_NAME=aws-rag-docs-$(aws sts get-caller-identity --query Account --output text)

# Pinecone — replace with your actual API key from https://app.pinecone.io
export PINECONE_API_KEY=your-pinecone-api-key-here
export PINECONE_INDEX_NAME=aws-rag-index

# Bedrock model IDs
export EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
export LLM_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

# Print confirmation
echo "Environment variables set:"
echo "  AWS_DEFAULT_REGION  = $AWS_DEFAULT_REGION"
echo "  S3_BUCKET_NAME      = $S3_BUCKET_NAME"
echo "  PINECONE_INDEX_NAME = $PINECONE_INDEX_NAME"
echo "  EMBEDDING_MODEL_ID  = $EMBEDDING_MODEL_ID"
echo "  LLM_MODEL_ID        = $LLM_MODEL_ID"
echo ""
if [ "$PINECONE_API_KEY" = "your-pinecone-api-key-here" ]; then
    echo "⚠️  WARNING: You still need to replace PINECONE_API_KEY with your real key!"
    echo "   Edit this file and run 'source set_env.sh' again."
fi
