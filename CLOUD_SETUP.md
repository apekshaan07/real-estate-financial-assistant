# Cloud Setup Guide

Follow these steps to complete the partially done assignment items.

Run this checker anytime:

```bash
source venv/bin/activate
python scripts/verify_setup.py
```

The Streamlit sidebar **System Status** panel shows what is active vs fallback.

---

## 1. PostgreSQL (replace SQLite)

### Steps

```bash
chmod +x scripts/setup_postgres.sh
./scripts/setup_postgres.sh
```

This will:
- start Postgres with Docker
- write `DATABASE_URL` into `.env`
- load `sql/create_tables.sql` and `sql/seed_data.sql`

### Verify

```bash
python scripts/verify_setup.py
streamlit run app/streamlit_app.py
```

Dashboard should show: `Database backend: POSTGRES`

---

## 2. Vertex AI ADK chatbot (GCP)

### Prerequisites

- GCP project with billing enabled
- Vertex AI API enabled
- `gcloud` installed

### Steps

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Edit `.env`:

```bash
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true
USE_VERTEX_ADK=true
```

Restart Streamlit and open **Assistant**. Sidebar should show:

`Vertex Adk: Vertex AI ADK enabled`

Ask a question. Chat engine should show `vertex_ai_adk` in results.

### Record for assessment

Show `.env` values, `gcloud auth`, one chat question, and the ADK tool routing in your cloud setup video.

---

## 3. SageMaker hosted endpoints (AWS)

### Prerequisites

- AWS account
- SageMaker execution role
- AWS CLI configured: `aws configure`

### Steps

```bash
export AWS_REGION=us-east-1
export SAGEMAKER_ROLE=arn:aws:iam::ACCOUNT:role/YOUR_SAGEMAKER_ROLE

python scripts/package_sagemaker_models.py
python deployment/deploy_regression.py --endpoint-name housing-regression-endpoint
python deployment/deploy_classification.py --endpoint-name bank-classification-endpoint
```

Add to `.env`:

```bash
SAGEMAKER_REGRESSION_ENDPOINT=housing-regression-endpoint
SAGEMAKER_CLASSIFICATION_ENDPOINT=bank-classification-endpoint
```

Restart Streamlit, open **ML Predictions**, run both predictions.

You should see:

`Inference source: sagemaker:housing-regression-endpoint`

### Record for assessment

Show deploy commands, endpoint names in AWS console, and live predictions in Streamlit.

### Cost note

SageMaker endpoints cost money while running. Delete endpoints after demo if needed.

---

## 4. AWS Bedrock summarization

Edit `.env`:

```bash
USE_BEDROCK=true
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=amazon.titan-text-lite-v1
```

Ensure your IAM user/role has `bedrock:InvokeModel`.

Open **Press Releases** and check the Bedrock summary box.

---

## 5. Azure OpenAI summarization

Edit `.env`:

```bash
USE_AZURE_OPENAI=true
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

Open **Press Releases** and check the Azure summary box.

---

## 6. SEC EDGAR live extraction

Edit `.env`:

```bash
USE_SEC_EDGAR=true
SEC_COMPANY_CIK=0001045609
SEC_USER_AGENT=YourAppName/1.0 (your-real-email@domain.com)
```

SEC requires a descriptive User-Agent with contact email.

Open **SEC Financials**. Live mode tries to fetch real 10-K/10-Q filings and extract revenue, net income, and operating expenses. If extraction fails, mock data is used as fallback.

---

## 7. Final submission checklist

- [ ] Postgres running (not SQLite only)
- [ ] Vertex ADK chat demo recorded
- [ ] SageMaker endpoints deployed and used in ML Predictions
- [ ] Bedrock or Azure summarization shown
- [ ] SEC page explained (live vs fallback)
- [ ] Screen recording of full app
- [ ] Cloud setup recording
- [ ] Public deployment URL
- [ ] Google Drive upload (exclude `venv/`)

---

## Quick command summary

```bash
# Postgres
./scripts/setup_postgres.sh

# Check everything
python scripts/verify_setup.py

# SageMaker
export SAGEMAKER_ROLE=arn:aws:iam::ACCOUNT:role/ROLE
python deployment/deploy_regression.py
python deployment/deploy_classification.py

# Run app
streamlit run app/streamlit_app.py --server.fileWatcherType none
```
