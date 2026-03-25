from dotenv import load_dotenv
import os
from typing import List
import chromadb
from openai import AzureOpenAI

# ====================== LOAD ENVIRONMENT ======================
load_dotenv()

API_Key = (
    os.getenv("AZURE_OPENAI_API_KEY")
    or os.getenv("AZURE_OPENAI_AD_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("API_Key")
)

if not API_Key:
    raise RuntimeError("Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment.")

client = AzureOpenAI(
    azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
    api_key=API_Key,
    api_version="2025-01-01-preview",
)

# ====================== CHROMA DB SETUP ======================
CHROMA_PATH = os.getenv("CHROMA_PATH") or "chroma_db/chroma_db"
COLLECTION_NAME = "rag"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# ====================== HELPER: GET EMBEDDING ======================
def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text.replace("\n", " "),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    return response.data[0].embedding

# ====================== RETRIEVAL FUNCTION ======================
def retrieve_documents(query: str, top_k: int = 5) -> List[dict]:
    """
    Retrieve top-k most relevant chunks from ChromaDB using cosine similarity.
    Returns list of dicts with 'text', 'url', and 'score'.
    """
    if not query.strip():
        # Return some default documents if query is empty
        results = collection.peek(limit=top_k)
        return [{"text": doc, "url": meta.get("url", ""), "score": 1.0}
                for doc, meta in zip(results["documents"], results["metadatas"])]

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        # Convert distance to similarity score (Chroma uses cosine distance by default)
        score = 1 - distance
        retrieved.append({
            "text": doc,
            "url": meta.get("url", ""),
            "source": meta.get("source", "HKU InnoWings / InnoAcademy"),
            "score": round(score, 4)
        })

    return retrieved


# ====================== CORE RAG FUNCTION ======================
def rag_answer(question: str) -> str:
    """
    Retrieve relevant chunks from ChromaDB and generate answer using Azure OpenAI.
    """
    context_docs = retrieve_documents(question, top_k=5)

    # Build clean context with sources
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(
            f"Document {i} (Source: {doc['url']}):\n{doc['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for HKU InnoWings and InnoAcademy. "
                "Answer the question using ONLY the provided context documents. "
                "If the answer cannot be found in the context, say: "
                "'I don't have enough information based on the provided documents.' "
                "Be concise, accurate, and professional. Always cite the source URL when possible."
            )
        },
        {
            "role": "user",
            "content": f"Context documents:\n{context}\n\nQuestion: {question}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
    )

    return response.choices[0].message.content


# ====================== PUBLIC API FUNCTION ======================
def generate_rag_answers(questions: List[str]) -> List[str]:
    """
    Main callable function for other scripts.
    
    Example usage:
        from rag import generate_rag_answers
        answers = generate_rag_answers([
            "What are the current SIGs in InnoWings?",
            "Tell me about recent Tech Talks in InnoAcademy."
        ])
        print(answers)
    """
    answers = []
    for question in questions:
        print(f"🤖 Answering: {question[:80]}{'...' if len(question) > 80 else ''}")
        answer = rag_answer(question)
        answers.append(answer)
    return answers


if __name__ == "__main__":
    # Simple test when running directly
    test_questions = [
        "What is InnoWings?",
        "Tell me about projects or SIGs."
    ]
    answers = generate_rag_answers(test_questions)
    for q, a in zip(test_questions, answers):
        print(f"\nQ: {q}\nA: {a}\n")