import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://api-iw.azure-api.net/sig-embedding/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21"         # e.g. https://YOUR-RESOURCE.openai.azure.com/
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = "text-embedding-3-small"  # your deployment name for text-embedding-3-small
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
)

text = "This is a test"
resp = client.embeddings.create(
    model=deployment,
    input=[text],
)

vec = resp.data[0].embedding
print("embedding_length:", len(vec))
print("first_5_values:", vec[:5])
print("usage:", resp.usage)