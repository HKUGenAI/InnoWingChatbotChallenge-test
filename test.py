import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import os
from typing import List, Dict, Set

# ====================== CONFIG ======================
SEED_URLS = [
    "https://innowings.engg.hku.hk/",
    "https://innoacademy.engg.hku.hk/",
]

ALLOWED_DOMAINS = {"innowings.engg.hku.hk", "innoacademy.engg.hku.hk"}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/133.0 Safari/537.36"
}

MAX_PAGES = 150          # Safety limit - increase if needed
DELAY = 1.0              # Seconds between requests (polite enough for this small site)

OUTPUT_FILE = "data.json"

# ====================== HELPER FUNCTIONS ======================
def is_internal_link(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in ALLOWED_DOMAINS or not parsed.netloc  # allow relative links

def simple_clean_text(soup: BeautifulSoup) -> str:
    # Very basic cleaning - remove script/style only
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Remove excessive blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def scrape_page(url: str) -> Dict[str, str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = simple_clean_text(soup)
        return {"url": url, "text": text}
    except Exception as e:
        print(f"❌ Failed {url}: {e}")
        return {"url": url, "text": f"[ERROR: {e}]"}

# ====================== SIMPLE CRAWLER ======================
def crawl_sites() -> List[Dict[str, str]]:
    visited: Set[str] = set()
    queue: List[str] = SEED_URLS[:]
    documents: List[Dict[str, str]] = []

    print("🚀 Starting simplest possible crawl...\n")

    while queue and len(documents) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue

        print(f"📄 [{len(documents)+1}/{MAX_PAGES}] Scraping → {url}")
        doc = scrape_page(url)
        if doc["text"].strip():
            documents.append(doc)
        visited.add(url)

        # Extract links (very basic)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                full_url = urljoin(url, a["href"])
                if is_internal_link(full_url) and full_url not in visited:
                    queue.append(full_url)
        except:
            pass  # ignore link extraction errors

        time.sleep(DELAY)

    print(f"\n✅ Crawl finished! Collected {len(documents)} pages.")

    # Save to data.json
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to {OUTPUT_FILE} — ready for ingest.py")
    return documents


if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        print(f"⚠️  {OUTPUT_FILE} already exists. Delete it if you want a fresh crawl.")
    else:
        crawl_sites()