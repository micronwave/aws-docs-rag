"""
01_ingest_docs.py
Downloads AWS documentation pages, cleans them, and uploads to S3.

Run: python scripts/01_ingest_docs.py
"""

import os
import re
import json
import time
import boto3
import requests
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin, urlparse

# ─── Configuration ────────────────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")

# Seed URLs — starting pages for each AWS service's user guide
SEED_URLS = {
    "s3": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
    "ec2": "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
    "lambda": "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
    "dynamodb": "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html",
    "vpc": "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
}

MAX_PAGES_PER_SERVICE = 30   # cap per service to keep dataset manageable
REQUEST_DELAY = 1            # seconds between HTTP requests

s3_client = boto3.client("s3", region_name=REGION)


def get_page(url: str, max_retries: int = 3) -> str | None:
    """Download a single page with retries. Returns HTML or None on failure."""
    headers = {"User-Agent": "AWS-RAG-Project-Educational/1.0"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s
                print(f"  ✗ Attempt {attempt + 1} failed: {url} — {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ✗ Failed after {max_retries} attempts: {url} — {e}")
                return None


def extract_links(html: str, base_url: str, service_path: str) -> list[str]:
    """Extract same-service documentation links from a page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    parsed_base = urlparse(base_url)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        parsed = urlparse(full)

        if (
            parsed.netloc == parsed_base.netloc
            and service_path in parsed.path
            and parsed.path.endswith(".html")
            and "#" not in href
        ):
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            links.append(clean)

    return list(set(links))


def clean_html(html: str) -> str:
    """Strip AWS page chrome, return plain text content."""
    soup = BeautifulSoup(html, "html.parser")

    main = (
        soup.find("div", {"id": "main-col-body"})
        or soup.find("main")
        or soup.find("div", {"class": "awsdocs-container"})
        or soup.find("body")
    )
    if not main:
        return ""

    for tag in main.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    for tag in main.find_all("div", {"class": re.compile(r"feedback|thumbs|awsdocs-page-header")}):
        tag.decompose()

    text = main.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def scrape_service(service_name: str, seed_url: str) -> list[dict]:
    """Crawl documentation for one AWS service starting from seed_url."""
    print(f"\n{'='*60}")
    print(f"  Scraping: {service_name}")
    print(f"  Seed:     {seed_url}")
    print(f"{'='*60}")

    parsed = urlparse(seed_url)
    service_path = parsed.path.rsplit("/", 1)[0] + "/"

    visited = set()
    queue = deque([seed_url])
    docs = []

    while queue and len(docs) < MAX_PAGES_PER_SERVICE:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        print(f"  [{len(docs)+1}/{MAX_PAGES_PER_SERVICE}] {url}")
        html = get_page(url)
        if not html:
            continue

        text = clean_html(html)
        if len(text) < 300:
            print(f"    ↳ skipped (only {len(text)} chars)")
            continue

        docs.append({
            "service": service_name,
            "url": url,
            "content": text,
            "char_count": len(text),
        })

        for link in extract_links(html, url, service_path):
            if link not in visited:
                queue.append(link)

        time.sleep(REQUEST_DELAY)

    print(f"  ✓ Collected {len(docs)} pages for {service_name}")
    return docs


def upload_to_s3(documents: list[dict]) -> None:
    """Upload cleaned documents to S3."""
    print(f"\nUploading {len(documents)} documents to s3://{S3_BUCKET}/raw-docs/ ...")

    for i, doc in enumerate(documents):
        key = f"raw-docs/{doc['service']}/{i:04d}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET, Key=key,
            Body=json.dumps(doc, indent=2),
            ContentType="application/json",
        )

    manifest = {
        "total_documents": len(documents),
        "services": list(set(d["service"] for d in documents)),
        "total_characters": sum(d["char_count"] for d in documents),
    }
    s3_client.put_object(
        Bucket=S3_BUCKET, Key="raw-docs/manifest.json",
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json",
    )
    print(f"  ✓ Upload complete.")
    print(f"  Manifest: {json.dumps(manifest, indent=2)}")


def main():
    all_docs = []
    for name, url in SEED_URLS.items():
        all_docs.extend(scrape_service(name, url))

    print(f"\n{'='*60}")
    print(f"  TOTAL: {len(all_docs)} documents collected")
    print(f"{'='*60}")

    if all_docs:
        upload_to_s3(all_docs)
        os.makedirs("local-data/raw-docs", exist_ok=True)
        with open("local-data/raw-docs/all_documents.json", "w") as f:
            json.dump(all_docs, f, indent=2)
        print("  Also saved locally → local-data/raw-docs/all_documents.json")
    else:
        print("  ✗ No documents collected. Check internet & URLs.")


if __name__ == "__main__":
    main()
