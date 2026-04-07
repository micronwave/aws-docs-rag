$env:AWS_DEFAULT_REGION = "us-east-2"
$env:S3_BUCKET_NAME = "aws-rag-docs-YOUR_ACCOUNT_ID_HERE"
$env:PINECONE_API_KEY = "your-pinecone-api-key-here"
$env:PINECONE_INDEX_NAME = "aws-rag-index"
$env:EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
$env:LLM_MODEL_ID = "us.anthropic.claude-sonnet-4-6"

# Add AWS CLI to path if needed inside venv
# $env:PATH = $env:PATH + ";C:\Program Files\Amazon\AWSCLIV2"

Write-Host "Environment variables set:"
Write-Host "  S3_BUCKET_NAME      = $env:S3_BUCKET_NAME"
Write-Host "  PINECONE_INDEX_NAME = $env:PINECONE_INDEX_NAME"
Write-Host "  EMBEDDING_MODEL_ID  = $env:EMBEDDING_MODEL_ID"
Write-Host "  LLM_MODEL_ID        = $env:LLM_MODEL_ID"

if ($env:PINECONE_API_KEY -eq "your-pinecone-api-key-here") {
    Write-Host "WARNING: Replace PINECONE_API_KEY with your real key!" -ForegroundColor Yellow
}
