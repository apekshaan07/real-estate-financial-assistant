import json
import os


def summarize_with_bedrock_or_local(text: str) -> str:
    """Summarize text with AWS Bedrock when configured, otherwise use local fallback."""
    use_bedrock = os.getenv("USE_BEDROCK", "false").lower() == "true"

    if use_bedrock:
        try:
            import boto3

            bedrock_runtime = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
            )
            model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
            prompt = f"Summarize the following business update in 2 sentences:\n\n{text}"

            if model_id.startswith("amazon.nova"):
                body = json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": [{"text": prompt}]}
                        ],
                        "inferenceConfig": {
                            "maxTokens": 200,
                            "temperature": 0.3,
                        },
                    }
                )
                response = bedrock_runtime.invoke_model(modelId=model_id, body=body)
                payload = json.loads(response["body"].read())
                return payload["output"]["message"]["content"][0]["text"].strip()

            if model_id.startswith("amazon.titan"):
                body = json.dumps(
                    {
                        "inputText": prompt,
                        "textGenerationConfig": {
                            "maxTokenCount": 200,
                            "temperature": 0.3,
                        },
                    }
                )
                response = bedrock_runtime.invoke_model(modelId=model_id, body=body)
                payload = json.loads(response["body"].read())
                return payload["results"][0]["outputText"].strip()

            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
            response = bedrock_runtime.invoke_model(modelId=model_id, body=body)
            payload = json.loads(response["body"].read())
            return payload["content"][0]["text"].strip()

        except Exception as exc:
            return _local_summary(text, prefix=f"Bedrock fallback used: {exc}. ")

    return _local_summary(text)


def _local_summary(text: str, prefix: str = "") -> str:
    sentences = [part.strip() for part in text.split(".") if part.strip()]
    summary = ". ".join(sentences[:2])
    if summary and not summary.endswith("."):
        summary += "."
    return prefix + (summary or text)
