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
EMBEDDING_DIMENSION = 1024
DELAY = 0.1  # seconds between Bedrock calls; increase if throttled

s3_client = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def make_run_id() -> str:
    """Create a run identifier for versioned S3 prefixes."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def expected_chunks_prefix(run_id: str) -> str:
    return f"chunks/{run_id}/"


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

    prefix = manifest.get("documents_prefix") or expected_chunks_prefix(run_id)
    if prefix != expected_chunks_prefix(run_id):
        raise SystemExit(
            f"chunks/manifest.json points to {prefix!r}, but the manifest run_id is {run_id!r}; rerun 02_chunk_docs.py."
        )
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
    embedding = json.loads(resp["body"].read())["embedding"]
    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIMENSION}, got {len(embedding)} from {EMBEDDING_MODEL_ID}."
        )
    return embedding


def validate_chunk_record(chunk: dict) -> None:
    """Fail fast on malformed chunk records before calling Bedrock."""
    required_fields = ("chunk_id", "service", "source_url", "chunk_index", "content")
    missing = [field for field in required_fields if field not in chunk]
    if missing:
        raise ValueError(f"Missing required chunk fields: {', '.join(missing)}")
    if not isinstance(chunk["content"], str) or not chunk["content"].strip():
        raise ValueError(f"Chunk {chunk.get('chunk_id', '<unknown>')} has empty content.")


def process_chunks(chunks: list[dict], generation_id: str) -> tuple[list[dict], list[str]]:
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
            validate_chunk_record(chunk)
            vec = embed_text(chunk["content"])
            embedded.append({**chunk, "generation_id": generation_id, "embedding": vec})

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

            time.sleep(DELAY)

        except Exception as e:
            failed += 1
            chunk_id = chunk.get("chunk_id", f"chunk_{i:06d}")
            failed_chunks.append(chunk_id)
            print(f"  [ERR] ERROR chunk {chunk_id}: {e}")
            if "ThrottlingException" in str(e):
                print("    Throttled — waiting 10s...")
                time.sleep(10)

    if failed_chunks:
        failed_path = "local-data/failed_chunks.json"
        os.makedirs("local-data", exist_ok=True)
        with open(failed_path, "w") as f:
            json.dump(failed_chunks, f, indent=2)
        print(f"  Failed chunk IDs saved to {failed_path}")

    status = "[OK]" if not failed else "[WARN]"
    print(f"\n  {status} Done: {len(embedded)} succeeded, {failed} failed")
    return embedded, failed_chunks


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
        "generation_id": run_id,
        "status": "success",
        "documents_prefix": output_prefix,
        "total_embedded_chunks": len(embedded),
        "embedding_model": EMBEDDING_MODEL_ID,
        "embedding_dimensions": EMBEDDING_DIMENSION,
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
            "generation_id": run_id,
            "status": "failed",
            "reason": "empty_corpus",
            "documents_prefix": None,
            "source_run_id": source_manifest.get("run_id"),
            "source_documents_prefix": source_manifest.get("documents_prefix"),
        }
        write_embeddings_manifest(failure_manifest)
        print("No chunks found in the current chunks run. Run 02_chunk_docs.py first.")
        raise SystemExit(1)

    embedded, failed_chunks = process_chunks(chunks, run_id)
    if failed_chunks:
        failure_manifest = {
            "run_id": run_id,
            "generation_id": run_id,
            "status": "failed",
            "reason": "embedding_generation_failure",
            "documents_prefix": None,
            "failed_chunk_ids": failed_chunks,
            "source_run_id": source_manifest.get("run_id"),
            "source_documents_prefix": source_manifest.get("documents_prefix"),
        }
        write_embeddings_manifest(failure_manifest)
        print("Embedding generation failed for one or more chunks. Check errors above.")
        raise SystemExit(1)

    if not embedded:
        failure_manifest = {
            "run_id": run_id,
            "generation_id": run_id,
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
