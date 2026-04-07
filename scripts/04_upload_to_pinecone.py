"""
04_upload_to_pinecone.py
Creates a Pinecone index (if it doesn't exist) and uploads all
embedded chunks from S3 into it.

Run: python scripts/04_upload_to_pinecone.py
"""

import os
import json
import time
import boto3
from pinecone import Pinecone, ServerlessSpec

# ─── Configuration ────────────────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "aws-rag-index")

EMBEDDING_DIMENSION = 1024  # must match Titan v2 output
BATCH_SIZE = 100            # Pinecone recommends batches of 100

s3_client = boto3.client("s3", region_name=REGION)


def load_embeddings_from_s3() -> list[dict]:
    """Download all embedded chunks from S3."""
    print("Loading embeddings from S3...")
    chunks = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="embeddings/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json") or key == "embeddings/manifest.json":
                continue
            resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            chunks.extend(json.loads(resp["Body"].read().decode("utf-8")))
    print(f"  Loaded {len(chunks)} embedded chunks")
    return chunks


def create_index(pc: Pinecone) -> None:
    """Create Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME in existing:
        print(f"  Index '{INDEX_NAME}' already exists — skipping creation.")
        return

    print(f"  Creating index '{INDEX_NAME}'...")
    print(f"    Dimension: {EMBEDDING_DIMENSION}")
    print(f"    Metric:    cosine")
    print(f"    Cloud:     aws / us-east-1")

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",  # Pinecone free tier supports us-east-1
        ),
    )

    # Wait for index to be ready
    print("  Waiting for index to be ready...", end="", flush=True)
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
        print(".", end="", flush=True)
    print(" ready!")


def upload_to_pinecone(pc: Pinecone, chunks: list[dict]) -> None:
    """Upload embedded chunks to Pinecone in batches."""
    index = pc.Index(INDEX_NAME)
    total = len(chunks)
    print(f"\nUploading {total} vectors to Pinecone index '{INDEX_NAME}'...")

    failed_batches = []
    for bn in range(0, total, BATCH_SIZE):
        batch = chunks[bn:bn + BATCH_SIZE]

        vectors = []
        for chunk in batch:
            vectors.append({
                "id": chunk["chunk_id"],
                "values": chunk["embedding"],
                "metadata": {
                    "service": chunk["service"],
                    "source_url": chunk["source_url"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                },
            })

        try:
            index.upsert(vectors=vectors)
            print(f"  Uploaded batch {bn // BATCH_SIZE} ({len(batch)} vectors)")
        except Exception as e:
            batch_num = bn // BATCH_SIZE
            failed_batches.append(batch_num)
            print(f"  ✗ Batch {batch_num} failed: {e}")

    if failed_batches:
        print(f"\n  WARNING: {len(failed_batches)} batches failed: {failed_batches}")
        print(f"  Re-run the script to retry (upsert is idempotent).")
        failed_path = "local-data/failed_batches.json"
        os.makedirs("local-data", exist_ok=True)
        with open(failed_path, "w") as f:
            json.dump(failed_batches, f, indent=2)
        print(f"  Failed batch numbers saved to {failed_path}")

    # Verify
    time.sleep(3)  # give Pinecone a moment to index
    stats = index.describe_index_stats()
    print(f"\n  ✓ Pinecone index stats:")
    print(f"    Total vectors: {stats.total_vector_count}")
    print(f"    Dimension:     {stats.dimension}")


def main():
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index
    create_index(pc)

    # Load embeddings from S3
    chunks = load_embeddings_from_s3()
    if not chunks:
        print("No embeddings found. Run 03_generate_embeddings.py first.")
        return

    # Upload
    upload_to_pinecone(pc, chunks)
    print("\n  ✓ All vectors uploaded. Ready for queries!")


if __name__ == "__main__":
    main()
