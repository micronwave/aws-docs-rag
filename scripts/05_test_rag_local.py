"""
05_test_rag_local.py
Tests the full RAG pipeline locally: embed question -> search Pinecone ->
build prompt -> call Claude -> print answer.

Run: python scripts/05_test_rag_local.py "How do I create an S3 bucket?"
"""

import os
import sys
import json
import boto3
from pinecone import Pinecone

# ─── Configuration ────────────────────────────────────────────────────
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "aws-rag-index")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
TOP_K = 5  # number of chunks to retrieve

bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def embed_query(text: str) -> list[float]:
    """Generate embedding for the user's question."""
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    resp = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    return json.loads(resp["body"].read())["embedding"]


def search_pinecone(query_vector: list[float]) -> list[dict]:
    """Search Pinecone for the top-K most similar chunks."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,  # we need the chunk text back
    )

    # Extract the matches
    chunks = []
    for match in results.matches:
        chunks.append({
            "score": match.score,
            "content": match.metadata.get("content", ""),
            "service": match.metadata.get("service", ""),
            "source_url": match.metadata.get("source_url", ""),
        })
    return chunks


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the prompt that we send to Claude.
    Includes the retrieved documentation chunks and anti-hallucination instructions.
    """
    # Format the context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['service'].upper()} — {chunk['source_url']}]\n"
            f"{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an AWS documentation assistant. Answer the user's question
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

    return prompt


def call_claude(prompt: str) -> str:
    """Send the prompt to Claude 3.5 Sonnet via Bedrock and return the response."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    })

    resp = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(resp["body"].read())
    # Claude's response is in content[0].text
    return result["content"][0]["text"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/05_test_rag_local.py \"Your question here\"")
        print("\nExample:")
        print("  python scripts/05_test_rag_local.py \"How do I create an S3 bucket?\"")
        sys.exit(1)

    question = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"  Question: {question}")
    print(f"{'='*60}")

    # Step 1: Embed the question
    print("\n[1/4] Generating question embedding...")
    query_vec = embed_query(question)
    print(f"  [OK] Generated 1024-dim vector")

    # Step 2: Search Pinecone
    print(f"\n[2/4] Searching Pinecone (top {TOP_K} results)...")
    chunks = search_pinecone(query_vec)
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. [{c['service']}] score={c['score']:.4f} — {c['source_url']}")

    # Step 3: Build prompt
    print(f"\n[3/4] Building prompt with {len(chunks)} context chunks...")
    prompt = build_prompt(question, chunks)
    print(f"  [OK] Prompt length: {len(prompt)} characters")

    # Step 4: Call Claude
    print(f"\n[4/4] Calling Claude 3.5 Sonnet via Bedrock...")
    answer = call_claude(prompt)

    print(f"\n{'='*60}")
    print(f"  ANSWER")
    print(f"{'='*60}")
    print(answer)
    print(f"\n{'='*60}")
    print(f"  Sources:")
    for i, c in enumerate(chunks, 1):
        print(f"    {i}. {c['source_url']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
