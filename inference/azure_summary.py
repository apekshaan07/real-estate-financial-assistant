"""Azure OpenAI summarization integration for multi-cloud awareness."""

from __future__ import annotations

import os


def summarize_with_azure_or_local(text: str) -> str:
    use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

    if use_azure:
        try:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-15-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize business press releases in two concise sentences.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=180,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            return _local_summary(text, prefix=f"Azure fallback used: {exc}. ")

    return _local_summary(text)


def _local_summary(text: str, prefix: str = "") -> str:
    sentences = [part.strip() for part in text.split(".") if part.strip()]
    summary = ". ".join(sentences[:2])
    if summary and not summary.endswith("."):
        summary += "."
    return prefix + (summary or text)
