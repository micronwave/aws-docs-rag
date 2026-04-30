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
UPSERT_MAX_RETRIES = int(os.environ.get("PINECONE_UPSERT_MAX_RETRIES", "3"))
UPSERT_BACKOFF_SECONDS = float(os.environ.get("PINECONE_UPSERT_BACKOFF_SECONDS", "1.0"))
EXPECTED_METRIC = "cosine"

s3_client = boto3.client("s3", region_name=REGION)


def expected_embeddings_prefix(run_id: str) -> str:
    return f"embeddings/{run_id}/"


def get_manifest_prefix(manifest: dict, run_id: str) -> str:
    prefix = manifest.get("documents_prefix") or expected_embeddings_prefix(run_id)
    if prefix != expected_embeddings_prefix(run_id):
        raise SystemExit(
            f"embeddings/manifest.json points to {prefix!r}, but the manifest run_id is {run_id!r}; rerun 03_generate_embeddings.py."
        )
    return prefix


def load_embeddings_from_s3() -> tuple[list[dict], dict]:
    """Download all embedded chunks from the latest embeddings manifest."""
    print("Loading embeddings from S3...")
    manifest_resp = s3_client.get_object(Bucket=S3_BUCKET, Key="embeddings/manifest.json")
    manifest = json.loads(manifest_resp["Body"].read().decode("utf-8"))
    run_id = manifest.get("run_id")
    if not run_id:
        raise SystemExit("embeddings/manifest.json is missing run_id; rerun 03_generate_embeddings.py.")
    if manifest.get("status") != "success":
        raise SystemExit("embeddings/manifest.json does not mark a successful embedding run; rerun 03_generate_embeddings.py.")

    prefix = get_manifest_prefix(manifest, run_id)
    chunks = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            resp = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            chunks.extend(json.loads(resp["Body"].read().decode("utf-8")))
    print(f"  Loaded {len(chunks)} embedded chunks from {prefix}")
    return chunks, manifest


def normalize_index_names(indexes) -> list[str]:
    names = []
    for idx in indexes:
        name = getattr(idx, "name", None)
        names.append(name if name is not None else str(idx))
    return names


def get_index_field(value, name: str):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def create_index(pc: Pinecone) -> None:
    """Create Pinecone index if it doesn't already exist."""
    existing = normalize_index_names(pc.list_indexes())

    if INDEX_NAME in existing:
        index_desc = pc.describe_index(INDEX_NAME)
        existing_dimension = get_index_field(index_desc, "dimension")
        existing_metric = get_index_field(index_desc, "metric")
        if existing_dimension != EMBEDDING_DIMENSION or existing_metric != EXPECTED_METRIC:
            raise SystemExit(
                f"Index '{INDEX_NAME}' exists with dimension={existing_dimension!r} metric={existing_metric!r}, "
                f"expected dimension={EMBEDDING_DIMENSION!r} metric={EXPECTED_METRIC!r}."
            )
        print(f"  Index '{INDEX_NAME}' already exists and matches the expected shape.")
        return

    print(f"  Creating index '{INDEX_NAME}'...")
    print(f"    Dimension: {EMBEDDING_DIMENSION}")
    print(f"    Metric:    {EXPECTED_METRIC}")
    print(f"    Cloud:     aws / us-east-1")

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric=EXPECTED_METRIC,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",  # Pinecone free tier supports us-east-1
        ),
    )

    print("  Waiting for index to be ready...", end="", flush=True)
    while True:
        status = get_index_field(pc.describe_index(INDEX_NAME), "status")
        if (isinstance(status, dict) and status.get("ready")) or getattr(status, "ready", False):
            break
        time.sleep(2)
        print(".", end="", flush=True)
    print(" ready!")


def required_embedding_fields(chunk: dict) -> list[str]:
    required = ["chunk_id", "embedding", "service", "source_url", "chunk_index", "content"]
    return [field for field in required if field not in chunk]


def validate_and_deduplicate_embeddings(chunks: list[dict]) -> tuple[list[dict], list[str]]:
    """Validate embedding records and collapse identical duplicate chunk IDs."""
    deduped = []
    seen_signatures = {}
    expected_ids = []

    for chunk in chunks:
        missing = required_embedding_fields(chunk)
        if missing:
            raise SystemExit(f"Embedding record {chunk.get('chunk_id', '<unknown>')} is missing required fields: {', '.join(missing)}")

        embedding = chunk["embedding"]
        if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMENSION:
            raise SystemExit(
                f"Embedding record {chunk['chunk_id']} has dimension {len(embedding) if isinstance(embedding, list) else '<non-list>'}, "
                f"expected {EMBEDDING_DIMENSION}."
            )

        signature = json.dumps(
            {
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding,
                "service": chunk["service"],
                "source_url": chunk["source_url"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "generation_id": chunk.get("generation_id"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        chunk_id = chunk["chunk_id"]
        if chunk_id in seen_signatures:
            if seen_signatures[chunk_id] != signature:
                raise SystemExit(f"Conflicting duplicate chunk_id detected: {chunk_id}")
            continue

        seen_signatures[chunk_id] = signature
        expected_ids.append(chunk_id)
        deduped.append(chunk)

    return deduped, expected_ids


def batch_ids(ids: list[str], batch_size: int) -> list[list[str]]:
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def record_signature(chunk: dict, run_id: str) -> dict:
    return {
        "id": chunk["chunk_id"],
        "values": chunk["embedding"],
        "metadata": {
            "service": chunk["service"],
            "source_url": chunk["source_url"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "generation_id": chunk.get("generation_id", run_id),
        },
    }


def retry_upsert(index, vectors: list[dict], batch_num: int) -> tuple[bool, str | None]:
    """Upsert a batch with bounded exponential backoff."""
    for attempt in range(UPSERT_MAX_RETRIES):
        try:
            index.upsert(vectors=vectors)
            return True, None
        except Exception as e:
            if attempt < UPSERT_MAX_RETRIES - 1:
                wait = UPSERT_BACKOFF_SECONDS * (2 ** attempt)
                print(f"  [ERR] Batch {batch_num} attempt {attempt + 1} failed: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                return False, str(e)


def describe_total_vectors(stats) -> int:
    if isinstance(stats, dict):
        return int(stats.get("total_vector_count", 0))
    return int(getattr(stats, "total_vector_count", 0))


def fetch_vectors(index, ids: list[str]) -> dict:
    fetched = index.fetch(ids=ids)
    if isinstance(fetched, dict):
        return fetched.get("vectors", {}) or {}
    return getattr(fetched, "vectors", {}) or {}


def persist_upload_failure(run_id: str, failed_batches: list[dict], verification_errors: list[str]) -> None:
    os.makedirs("local-data", exist_ok=True)
    failed_path = "local-data/failed_batches.json"
    with open(failed_path, "w") as f:
        json.dump([item["batch_num"] for item in failed_batches], f, indent=2)
    print(f"  Failed batch numbers saved to {failed_path}")

    detail_path = "local-data/failed_batch_details.json"
    with open(detail_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "failed_batches": failed_batches,
                "verification_errors": verification_errors,
            },
            f,
            indent=2,
        )
    print(f"  Failed batch details saved to {detail_path}")

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=f"pinecone/failed_batches/{run_id}.json",
        Body=json.dumps(
            {
                "run_id": run_id,
                "failed_batches": failed_batches,
                "verification_errors": verification_errors,
            },
            indent=2,
        ),
        ContentType="application/json",
    )


def verify_index_state(index, expected_ids: list[str], expected_generation_id: str) -> list[str]:
    """Confirm the index matches the uploaded generation."""
    stats = index.describe_index_stats()
    total_vector_count = describe_total_vectors(stats)
    expected_total = len(expected_ids)

    stale_errors = []
    if total_vector_count > expected_total:
        stale_errors.append(
            f"index contains {total_vector_count} vectors but only {expected_total} were expected for this generation"
        )

    missing_ids = []
    stale_generation_ids = []
    for id_batch in batch_ids(expected_ids, BATCH_SIZE):
        vectors = fetch_vectors(index, id_batch)
        for chunk_id in id_batch:
            vector = vectors.get(chunk_id)
            if not vector:
                missing_ids.append(chunk_id)
                continue
            metadata = vector.get("metadata", {}) if isinstance(vector, dict) else getattr(vector, "metadata", {}) or {}
            if metadata.get("generation_id") != expected_generation_id:
                stale_generation_ids.append(chunk_id)

    if missing_ids:
        stale_errors.append(f"missing ids after upload: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")
    if stale_generation_ids:
        stale_errors.append(
            f"stale generation_id on ids: {stale_generation_ids[:10]}{'...' if len(stale_generation_ids) > 10 else ''}"
        )

    return stale_errors


def upload_to_pinecone(pc: Pinecone, chunks: list[dict], manifest: dict) -> None:
    """Upload embedded chunks to Pinecone in batches."""
    index = pc.Index(INDEX_NAME)
    deduped_chunks, expected_ids = validate_and_deduplicate_embeddings(chunks)
    generation_id = manifest.get("generation_id") or manifest.get("run_id")
    total = len(deduped_chunks)
    print(f"\nUploading {total} vectors to Pinecone index '{INDEX_NAME}'...")
    if total != len(chunks):
        print(f"  Deduplicated {len(chunks) - total} duplicate records before upload.")

    failed_batches = []
    for bn in range(0, total, BATCH_SIZE):
        batch = deduped_chunks[bn:bn + BATCH_SIZE]
        batch_num = bn // BATCH_SIZE
        vectors = [record_signature(chunk, generation_id) for chunk in batch]

        ok, error = retry_upsert(index, vectors, batch_num)
        if ok:
            print(f"  Uploaded batch {batch_num} ({len(batch)} vectors)")
        else:
            failed_batches.append({
                "batch_num": batch_num,
                "chunk_ids": [chunk["chunk_id"] for chunk in batch],
                "error": error,
                "attempts": UPSERT_MAX_RETRIES,
            })
            print(f"  [ERR] Batch {batch_num} failed after {UPSERT_MAX_RETRIES} attempts: {error}")

    if failed_batches:
        print(f"\n  WARNING: {len(failed_batches)} batches failed after retries.")

    time.sleep(3)  # give Pinecone a moment to index
    verification_errors = verify_index_state(index, expected_ids, generation_id)

    if failed_batches or verification_errors:
        if verification_errors:
            print("  [ERR] Post-upload verification failed:")
            for error in verification_errors:
                print(f"    - {error}")
        persist_upload_failure(generation_id, failed_batches, verification_errors)
        raise SystemExit(1)

    stats = index.describe_index_stats()
    total_vectors = describe_total_vectors(stats)
    print(f"\n  [OK] Pinecone index stats:")
    print(f"    Total vectors: {total_vectors}")
    print(f"    Dimension:     {get_index_field(stats, 'dimension')}")


def main():
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index
    create_index(pc)

    # Load embeddings from S3
    chunks, manifest = load_embeddings_from_s3()
    if not chunks:
        print("No embeddings found in the current embeddings run. Run 03_generate_embeddings.py first.")
        raise SystemExit(1)

    # Upload
    upload_to_pinecone(pc, chunks, manifest)
    print("\n  [OK] All vectors uploaded. Ready for queries!")


if __name__ == "__main__":
    main()
