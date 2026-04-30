"""
02_chunk_docs.py
Reads raw documents from S3, splits into chunks using LangChain,
saves chunks back to S3.

Run: python scripts/02_chunk_docs.py
"""

import os
import json
from datetime import datetime, timezone
import uuid
import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Configuration ────────────────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 200   # overlap between consecutive chunks

s3_client = boto3.client("s3", region_name=REGION)


def make_run_id() -> str:
    """Create a run identifier for versioned S3 prefixes."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def expected_raw_docs_prefix(run_id: str) -> str:
    return f"raw-docs/{run_id}/"


def load_documents_from_s3() -> tuple[list[dict], dict]:
    """Download all raw documents from the latest raw-docs manifest."""
    print("Loading documents from S3...")
    manifest_resp = s3_client.get_object(Bucket=S3_BUCKET, Key="raw-docs/manifest.json")
    manifest = json.loads(manifest_resp["Body"].read().decode("utf-8"))
    run_id = manifest.get("run_id")
    if not run_id:
        raise SystemExit("raw-docs/manifest.json is missing run_id; rerun 01_ingest_docs.py.")
    if manifest.get("status") != "success":
        raise SystemExit("raw-docs/manifest.json does not mark a successful ingest run; rerun 01_ingest_docs.py.")

    prefix = manifest.get("documents_prefix") or expected_raw_docs_prefix(run_id)
    if prefix != expected_raw_docs_prefix(run_id):
        raise SystemExit(
            f"raw-docs/manifest.json points to {prefix!r}, but the manifest run_id is {run_id!r}; rerun 01_ingest_docs.py."
        )
    documents = []

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            doc = json.loads(resp["Body"].read().decode("utf-8"))
            documents.append(doc)

    print(f"  Loaded {len(documents)} documents from {prefix}")
    return documents, manifest


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    cid = 0

    for doc in documents:
        pieces = splitter.split_text(doc["content"])
        for i, text in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"chunk_{cid:06d}",
                "service": doc["service"],
                "source_url": doc["url"],
                "chunk_index": i,
                "total_chunks_in_doc": len(pieces),
                "content": text,
                "char_count": len(text),
            })
            cid += 1

    return all_chunks


def write_chunks_manifest(manifest: dict) -> None:
    """Write the chunks manifest pointer."""
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key="chunks/manifest.json",
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json",
    )


def upload_chunks_to_s3(chunks: list[dict], run_id: str, source_manifest: dict) -> None:
    """Upload chunks to S3 in batches."""
    output_prefix = f"chunks/{run_id}/"
    print(f"Uploading {len(chunks)} chunks to s3://{S3_BUCKET}/{output_prefix} ...")

    batch_size = 100
    for bn in range(0, len(chunks), batch_size):
        batch = chunks[bn:bn + batch_size]
        key = f"{output_prefix}batch_{bn // batch_size:04d}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET, Key=key,
            Body=json.dumps(batch, indent=2),
            ContentType="application/json",
        )
        print(f"  Uploaded batch {bn // batch_size} ({len(batch)} chunks)")

    manifest = {
        "run_id": run_id,
        "status": "success",
        "documents_prefix": output_prefix,
        "total_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "services": sorted(set(c["service"] for c in chunks)),
        "avg_chunk_chars": sum(c["char_count"] for c in chunks) // max(len(chunks), 1),
        "source_run_id": source_manifest.get("run_id"),
        "source_documents_prefix": source_manifest.get("documents_prefix"),
    }
    write_chunks_manifest(manifest)
    print(f"  [OK] Chunk manifest: {json.dumps(manifest, indent=2)}")


def main():
    run_id = make_run_id()
    documents, source_manifest = load_documents_from_s3()
    if not documents:
        failure_manifest = {
            "run_id": run_id,
            "status": "failed",
            "reason": "empty_corpus",
            "documents_prefix": None,
            "source_run_id": source_manifest.get("run_id"),
            "source_documents_prefix": source_manifest.get("documents_prefix"),
        }
        write_chunks_manifest(failure_manifest)
        print("No documents found in the current raw-docs run. Run 01_ingest_docs.py first.")
        raise SystemExit(1)

    chunks = chunk_documents(documents)
    if not chunks:
        failure_manifest = {
            "run_id": run_id,
            "status": "failed",
            "reason": "empty_chunks",
            "documents_prefix": None,
            "source_run_id": source_manifest.get("run_id"),
            "source_documents_prefix": source_manifest.get("documents_prefix"),
        }
        write_chunks_manifest(failure_manifest)
        print("No chunks were generated from the current documents. Check the minimum length filter and source content.")
        raise SystemExit(1)

    print(f"\nChunking results:")
    print(f"  Input documents: {len(documents)}")
    print(f"  Output chunks:   {len(chunks)}")
    print(f"  Avg chunk size:  {sum(c['char_count'] for c in chunks) // max(len(chunks),1)} chars")

    upload_chunks_to_s3(chunks, run_id, source_manifest)

    os.makedirs("local-data/chunks", exist_ok=True)
    with open("local-data/chunks/all_chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("  Also saved locally -> local-data/chunks/all_chunks.json")


if __name__ == "__main__":
    main()
