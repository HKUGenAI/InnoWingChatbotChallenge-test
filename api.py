API Key: 951e29149d8c40549450940452bcd0c6

import os
from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
  api_key="<YOUR_API_KEY>",
  api_version="2025-01-01-preview"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "Do other Azure AI services support this too?"}
    ]
)

print(response.choices[0].message.content)
