from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# load environment variables from .env (if present)
load_dotenv()

# Try common variable names and be robust to styling (same as your reference)
API_Key = (
    os.getenv("AZURE_OPENAI_API_KEY")
    or os.getenv("AZURE_OPENAI_AD_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("API_Key")
)

if not API_Key:
    raise RuntimeError(
        "Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment."
    )

client = AzureOpenAI(
    azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
    api_key=API_Key,
    api_version="2025-01-01-preview",
)

# ====================== BASIC RAG SETUP ======================
def retrieve_documents(query: str, top_k: int = 3) -> list[str]:
    """
    Very basic keyword-overlap retrieval (no extra dependencies, no embeddings needed).
    Students can later replace this with embeddings + FAISS or Azure AI Search.
    """
    if not query:
        return documents[:top_k]
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scores = []
    for doc in documents:
        doc_lower = doc.lower()
        doc_words = set(doc_lower.split())
        overlap = len(query_words.intersection(doc_words))
        score = overlap / len(query_words) if query_words else 0
        scores.append(score)
    
    # Get indices of top-k most relevant documents
    top_indices = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [documents[i] for i in top_indices]


def rag_answer(question: str) -> str:
    """
    Core RAG function: retrieve → augment prompt → generate with Azure OpenAI.
    """
    context_docs = retrieve_documents(question, top_k=3)
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)])
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use ONLY the provided context documents "
                "to answer the question. If the answer is not in the context, say "
                "'I don't have enough information based on the provided documents.' "
                "Be concise and accurate."
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
    )
    
    return response.choices[0].message.content


def generate_rag_answers(questions: list[str]) -> list[str]:
    """
    Main function for your other script to call.
    
    Example usage from another script:
        from rag_script import generate_rag_answers
        answers = generate_rag_answers(["Does Azure OpenAI support customer managed keys?", "Do other Azure AI services support this too?"])
        print(answers)
    """
    answers = []
    for question in questions:
        answer = rag_answer(question)
        answers.append(answer)
    return answers