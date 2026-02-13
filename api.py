from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# load environment variables from .env (if present)
load_dotenv()

# Try common variable names and be robust to styling
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

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
    {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
    {"role": "user", "content": "Do other Azure AI services support this too?"},
  ],
)

print(response.choices[0].message.content)
