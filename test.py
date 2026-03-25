from dotenv import load_dotenv
import os
import json
import hashlib
from openai import AzureOpenAI
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

load_dotenv()

# ====================== CONFIG ======================
DATASET = os.getenv("DATASET") or "data.json"  # Changed to data.json as requested
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
CHROMA_PATH = os.getenv("CHROMA_PATH") or "chroma_db/chroma_db"

API_Key = os.getenv("AZURE_OPENAI_API_KEY")
if not API_Key:
    raise RuntimeError("Missing Azure OpenAI credentials.")

client = AzureOpenAI(
    azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
    api_key=API_Key,
    api_version="2025-01-01-preview",
)

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding

# ====================== LOAD DOCUMENTS ======================
if os.path.exists(DATASET):
    print(f"📂 Loading documents from {DATASET}...")
    with open(DATASET, "r", encoding="utf-8") as f:
        documents: List[Dict] = json.load(f)
    print(f"   Loaded {len(documents)} raw pages.")
else:
    raise FileNotFoundError(
        f"❌ {DATASET} not found. Please run your scraper first to generate data.json (or set OUTPUT_FILE env var)."
    )

# ====================== CHROMA DB SETUP ======================
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

COLLECTION_NAME = "rag"

# Optional: uncomment for a completely fresh ingest (recommended first time)
# try:
#     chroma_client.delete_collection(name=COLLECTION_NAME)
#     print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}' for fresh ingest.")
# except:
#     pass

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ====================== CHUNKING (Industry Standard) ======================
print("\n🔪 Chunking documents using RecursiveCharacterTextSplitter (best-practice settings)...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # ~250-300 tokens for text-embedding-3-small (optimal balance)
    chunk_overlap=300,        # Critical for context continuity between chunks
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],  # Hierarchical splitting (paragraphs → sentences → words)
)

all_texts: List[str] = []
all_metadatas: List[Dict] = []
all_ids: List[str] = []

for doc in documents:
    url = doc["url"]
    text = doc.get("text", "").strip()
    if not text:
        continue

    chunks = text_splitter.split_text(text)

    for chunk_idx, chunk in enumerate(chunks):
        # Deterministic ID (hash of URL + chunk index) → safe re-ingest / upsert
        id_str = f"{url}:{chunk_idx}"
        chunk_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()

        all_texts.append(chunk)
        all_metadatas.append({
            "url": url,
            "source": "HKU InnoWings / InnoAcademy",
            "chunk_index": chunk_idx,
            "total_chunks_on_page": len(chunks),   # useful for debugging / future hierarchical RAG
        })
        all_ids.append(chunk_id)

print(f"   Created {len(all_texts)} chunks from {len(documents)} pages.")

# ====================== EMBEDDING + INGESTION ======================
print("🧬 Generating embeddings and storing in ChromaDB (this may take a while)...")

BATCH_SIZE = 50

for i in range(0, len(all_texts), BATCH_SIZE):
    batch_texts = all_texts[i : i + BATCH_SIZE]
    batch_metadatas = all_metadatas[i : i + BATCH_SIZE]
    batch_ids = all_ids[i : i + BATCH_SIZE]

    # Generate embeddings for the batch
    embeddings = [get_embedding(text) for text in batch_texts]

    # Use .upsert() instead of .add() → industry standard for idempotent ingestion
    collection.upsert(
        embeddings=embeddings,
        documents=batch_texts,
        metadatas=batch_metadatas,
        ids=batch_ids
    )

    print(f"  [{i + len(batch_texts)}/{len(all_texts)}] Added batch to ChromaDB")

print("\n🎉 Ingestion complete!")
print(f"   ChromaDB collection '{COLLECTION_NAME}' saved to: {CHROMA_PATH}")
print(f"   Total chunks stored: {collection.count()}")
print("   ✅ Ready for RAG! Use the same embedding model for queries.")