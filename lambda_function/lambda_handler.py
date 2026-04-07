"""
lambda_handler.py
AWS Lambda function that handles RAG queries.
Receives a question via API Gateway, retrieves relevant docs from Pinecone,
sends them to Claude via Bedrock, and returns the answer.
"""

import os
import json
import traceback
import boto3
from pinecone import Pinecone

# ─── Configuration (set as Lambda environment variables) ──────────────
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "aws-rag-index")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
TOP_K = 5

# Initialize clients outside the handler (reused across invocations)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def embed_query(text: str) -> list[float]:
    """Generate embedding for user's question."""
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    resp = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    return json.loads(resp["body"].read())["embedding"]


def search_pinecone(query_vector: list[float]) -> list[dict]:
    """Search Pinecone for top-K similar chunks."""
    results = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)
    chunks = []
    for match in results.matches:
        chunks.append({
            "score": round(match.score, 4),
            "content": match.metadata.get("content", ""),
            "service": match.metadata.get("service", ""),
            "source_url": match.metadata.get("source_url", ""),
        })
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build RAG prompt with context and anti-hallucination instructions."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['service'].upper()} — {chunk['source_url']}]\n"
            f"{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are an AWS documentation assistant. Answer the user's question
using ONLY the documentation provided below. If the documentation does not contain
enough information to answer the question, say "I don't have enough information in
the available documentation to answer that question."

Do NOT make up information. Do NOT use knowledge outside of the provided documentation.
When possible, reference which AWS service documentation your answer comes from.

<documentation>
{context}
</documentation>

<question>
{question}
</question>

Provide a clear, helpful answer based strictly on the documentation above."""


def call_claude(prompt: str) -> str:
    """Call Claude 3.5 Sonnet via Bedrock."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"]


def lambda_handler(event, context):
    """
    Main Lambda entry point.
    Expects: POST body with {"question": "your question here"}
    Returns: {"answer": "...", "sources": [...]}
    """
    try:
        # Parse the incoming request
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event.get("body", event)

        question = body.get("question", "").strip()

        if not question:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "POST,OPTIONS",
                },
                "body": json.dumps({"error": "No question provided"}),
            }

        if len(question) > 1000:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "POST,OPTIONS",
                },
                "body": json.dumps({"error": "Question too long. Maximum 1000 characters."}),
            }

        # RAG pipeline
        query_vec = embed_query(question)
        chunks = search_pinecone(query_vec)
        prompt = build_prompt(question, chunks)
        answer = call_claude(prompt)

        # Build sources list for the frontend
        sources = [
            {"service": c["service"], "url": c["source_url"], "score": c["score"]}
            for c in chunks
        ]

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST,OPTIONS",
            },
            "body": json.dumps({
                "answer": answer,
                "sources": sources,
                "question": question,
            }),
        }

    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST,OPTIONS",
            },
            "body": json.dumps({"error": "Internal server error"}),
        }
