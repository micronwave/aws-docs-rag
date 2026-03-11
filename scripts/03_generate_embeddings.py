"""
03_generate_embeddings.py
Reads chunks from S3, generates embeddings via Amazon Titan Embeddings v2,
saves embedded chunks back to S3.

Run: python scripts/03_generate_embeddings.py
"""

import os
import json
import time
import boto3

# ─── Configuration ────────────────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
DELAY = 0.1  # seconds between Bedrock calls; increase if throttled

s3_client = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def load_chunks_from_s3() -> list[dict]:
    """Download all chunks from S3."""
    print("Loading chunks from S3...")
    chunks = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="chunks/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json") or key == "chunks/manifest.json":
                continue
            resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            chunks.extend(json.loads(resp["Body"].read().decode("utf-8")))
    print(f"  Loaded {len(chunks)} chunks")
    return chunks


def embed_text(text: str) -> list[float]:
    """Call Titan Embeddings v2 for a single text → 1024-dim vector."""
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    resp = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    return json.loads(resp["body"].read())["embedding"]


def process_chunks(chunks: list[dict]) -> list[dict]:
    """Generate embeddings for all chunks with progress tracking."""
    total = len(chunks)
    print(f"\nGenerating embeddings for {total} chunks...")
    print(f"  Model: {EMBEDDING_MODEL_ID}")
    print(f"  Estimated time: {total * DELAY / 60:.1f}–{total * DELAY * 2 / 60:.1f} minutes\n")

    embedded = []
    failed = 0

    for i, chunk in enumerate(chunks):
        try:
            vec = embed_text(chunk["content"])
            embedded.append({**chunk, "embedding": vec})

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

            time.sleep(DELAY)

        except Exception as e:
            failed += 1
            print(f"  ✗ ERROR chunk {chunk['chunk_id']}: {e}")
            if "ThrottlingException" in str(e):
                print("    Throttled — waiting 10s...")
                time.sleep(10)

    print(f"\n  ✓ Done: {len(embedded)} succeeded, {failed} failed")
    return embedded


def upload_embeddings_to_s3(embedded: list[dict]) -> None:
    """Upload embedded chunks to S3 in batches."""
    print(f"Uploading {len(embedded)} embedded chunks to S3...")

    batch_size = 50
    for bn in range(0, len(embedded), batch_size):
        batch = embedded[bn:bn + batch_size]
        key = f"embeddings/batch_{bn // batch_size:04d}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET, Key=key,
            Body=json.dumps(batch),
            ContentType="application/json",
        )
        print(f"  Uploaded batch {bn // batch_size} ({len(batch)} chunks)")

    manifest = {
        "total_embedded_chunks": len(embedded),
        "embedding_model": EMBEDDING_MODEL_ID,
        "embedding_dimensions": 1024,
    }
    s3_client.put_object(
        Bucket=S3_BUCKET, Key="embeddings/manifest.json",
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json",
    )
    print(f"  ✓ Manifest: {json.dumps(manifest, indent=2)}")


def main():
    chunks = load_chunks_from_s3()
    if not chunks:
        print("No chunks found. Run 02_chunk_docs.py first.")
        return

    embedded = process_chunks(chunks)
    if not embedded:
        print("No embeddings generated. Check errors above.")
        return

    upload_embeddings_to_s3(embedded)

    os.makedirs("local-data/embeddings", exist_ok=True)
    with open("local-data/embeddings/all_embeddings.json", "w") as f:
        json.dump(embedded, f)
    size_mb = os.path.getsize("local-data/embeddings/all_embeddings.json") / 1024 / 1024
    print(f"  Also saved locally → local-data/embeddings/all_embeddings.json ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
