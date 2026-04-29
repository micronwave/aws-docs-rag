"""
03_generate_embeddings.py
Reads chunks from S3, generates embeddings via Amazon Titan Embeddings v2,
saves embedded chunks back to S3.

Run: python scripts/03_generate_embeddings.py
"""

import os
import json
import time
from datetime import datetime, timezone
import uuid
import boto3

# ─── Configuration ────────────────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
DELAY = 0.1  # seconds between Bedrock calls; increase if throttled

s3_client = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def make_run_id() -> str:
    """Create a run identifier for versioned S3 prefixes."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def load_chunks_from_s3() -> tuple[list[dict], dict]:
    """Download all chunks from the latest chunks manifest."""
    print("Loading chunks from S3...")
    manifest_resp = s3_client.get_object(Bucket=S3_BUCKET, Key="chunks/manifest.json")
    manifest = json.loads(manifest_resp["Body"].read().decode("utf-8"))
    run_id = manifest.get("run_id")
    if not run_id:
        raise SystemExit("chunks/manifest.json is missing run_id; rerun 02_chunk_docs.py.")
    if manifest.get("status") != "success":
        raise SystemExit("chunks/manifest.json does not mark a successful chunking run; rerun 02_chunk_docs.py.")

    prefix = manifest.get("documents_prefix") or f"chunks/{run_id}/"
    chunks = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            chunks.extend(json.loads(resp["Body"].read().decode("utf-8")))
    print(f"  Loaded {len(chunks)} chunks from {prefix}")
    return chunks, manifest


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
    failed_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            vec = embed_text(chunk["content"])
            embedded.append({**chunk, "embedding": vec})

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

            time.sleep(DELAY)

        except Exception as e:
            failed += 1
            failed_chunks.append(chunk["chunk_id"])
            print(f"  [ERR] ERROR chunk {chunk['chunk_id']}: {e}")
            if "ThrottlingException" in str(e):
                print("    Throttled — waiting 10s...")
                time.sleep(10)

    if failed_chunks:
        failed_path = "local-data/failed_chunks.json"
        os.makedirs("local-data", exist_ok=True)
        with open(failed_path, "w") as f:
            json.dump(failed_chunks, f, indent=2)
        print(f"  Failed chunk IDs saved to {failed_path}")

    print(f"\n  [OK] Done: {len(embedded)} succeeded, {failed} failed")
    return embedded


def write_embeddings_manifest(manifest: dict) -> None:
    """Write the embeddings manifest pointer."""
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key="embeddings/manifest.json",
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json",
    )


def upload_embeddings_to_s3(embedded: list[dict], run_id: str, source_manifest: dict) -> None:
    """Upload embedded chunks to S3 in batches."""
    output_prefix = f"embeddings/{run_id}/"
    print(f"Uploading {len(embedded)} embedded chunks to s3://{S3_BUCKET}/{output_prefix} ...")

    batch_size = 50
    for bn in range(0, len(embedded), batch_size):
        batch = embedded[bn:bn + batch_size]
        key = f"{output_prefix}batch_{bn // batch_size:04d}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET, Key=key,
            Body=json.dumps(batch),
            ContentType="application/json",
        )
        print(f"  Uploaded batch {bn // batch_size} ({len(batch)} chunks)")

    manifest = {
        "run_id": run_id,
        "status": "success",
        "documents_prefix": output_prefix,
        "total_embedded_chunks": len(embedded),
        "embedding_model": EMBEDDING_MODEL_ID,
        "embedding_dimensions": 1024,
        "source_run_id": source_manifest.get("run_id"),
        "source_documents_prefix": source_manifest.get("documents_prefix"),
    }
    write_embeddings_manifest(manifest)
    print(f"  [OK] Manifest: {json.dumps(manifest, indent=2)}")


def main():
    run_id = make_run_id()
    chunks, source_manifest = load_chunks_from_s3()
    if not chunks:
        failure_manifest = {
            "run_id": run_id,
            "status": "failed",
            "reason": "empty_corpus",
            "documents_prefix": None,
            "source_run_id": source_manifest.get("run_id"),
            "source_documents_prefix": source_manifest.get("documents_prefix"),
        }
        write_embeddings_manifest(failure_manifest)
        print("No chunks found in the current chunks run. Run 02_chunk_docs.py first.")
        raise SystemExit(1)

    embedded = process_chunks(chunks)
    if not embedded:
        failure_manifest = {
            "run_id": run_id,
            "status": "failed",
            "reason": "empty_embeddings",
            "documents_prefix": None,
            "source_run_id": source_manifest.get("run_id"),
            "source_documents_prefix": source_manifest.get("documents_prefix"),
        }
        write_embeddings_manifest(failure_manifest)
        print("No embeddings generated. Check errors above.")
        raise SystemExit(1)

    upload_embeddings_to_s3(embedded, run_id, source_manifest)

    os.makedirs("local-data/embeddings", exist_ok=True)
    with open("local-data/embeddings/all_embeddings.json", "w") as f:
        json.dump(embedded, f)
    size_mb = os.path.getsize("local-data/embeddings/all_embeddings.json") / 1024 / 1024
    print(f"  Also saved locally -> local-data/embeddings/all_embeddings.json ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
