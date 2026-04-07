"""
02_chunk_docs.py
Reads raw documents from S3, splits into chunks using LangChain,
saves chunks back to S3.

Run: python scripts/02_chunk_docs.py
"""

import os
import json
import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Configuration ────────────────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 200   # overlap between consecutive chunks

s3_client = boto3.client("s3", region_name=REGION)


def load_documents_from_s3() -> list[dict]:
    """Download all raw documents from S3."""
    print("Loading documents from S3...")
    documents = []

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="raw-docs/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json") or key == "raw-docs/manifest.json":
                continue
            resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            doc = json.loads(resp["Body"].read().decode("utf-8"))
            documents.append(doc)

    print(f"  Loaded {len(documents)} documents")
    return documents


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


def upload_chunks_to_s3(chunks: list[dict]) -> None:
    """Upload chunks to S3 in batches."""
    print(f"Uploading {len(chunks)} chunks to S3...")

    batch_size = 100
    for bn in range(0, len(chunks), batch_size):
        batch = chunks[bn:bn + batch_size]
        key = f"chunks/batch_{bn // batch_size:04d}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET, Key=key,
            Body=json.dumps(batch, indent=2),
            ContentType="application/json",
        )
        print(f"  Uploaded batch {bn // batch_size} ({len(batch)} chunks)")

    manifest = {
        "total_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "services": list(set(c["service"] for c in chunks)),
        "avg_chunk_chars": sum(c["char_count"] for c in chunks) // max(len(chunks), 1),
    }
    s3_client.put_object(
        Bucket=S3_BUCKET, Key="chunks/manifest.json",
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json",
    )
    print(f"  [OK] Chunk manifest: {json.dumps(manifest, indent=2)}")


def main():
    documents = load_documents_from_s3()
    if not documents:
        print("No documents found. Run 01_ingest_docs.py first.")
        return

    chunks = chunk_documents(documents)
    print(f"\nChunking results:")
    print(f"  Input documents: {len(documents)}")
    print(f"  Output chunks:   {len(chunks)}")
    print(f"  Avg chunk size:  {sum(c['char_count'] for c in chunks) // max(len(chunks),1)} chars")

    upload_chunks_to_s3(chunks)

    os.makedirs("local-data/chunks", exist_ok=True)
    with open("local-data/chunks/all_chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("  Also saved locally -> local-data/chunks/all_chunks.json")


if __name__ == "__main__":
    main()
