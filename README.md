# Financial Assistant Web Application for Real Estate (Prologis)

End-to-end AI-driven Financial Assistant that combines structured property/financial data, SEC EDGAR insights, press release analysis, classic ML models deployed to SageMaker, and a Vertex AI ADK chatbot with multi-cloud summarization (AWS Bedrock + Azure OpenAI).

## Architecture

```text
User (Streamlit UI)
    |
    +--> Vertex AI ADK Agent (GCP)
    |       +--> property_financial_tool
    |       +--> sec_financial_tool
    |       +--> press_release_tool
    |       +--> financial_overview_tool
    |
    +--> Local Router Fallback
    |
    +--> Data Sources
    |       +--> Postgres / SQLite (Properties + Financials)
    |       +--> SEC EDGAR API + structured metrics
    |       +--> press_releases.json
    |
    +--> SageMaker Endpoints (AWS)
    |       +--> Random Forest Regression (California Housing)
    |       +--> Logistic Regression Classification (Bank Marketing)
    |
    +--> Multi-Cloud Summaries
            +--> AWS Bedrock
            +--> Azure OpenAI
```

## Project Structure

```text
real-estate-financial-assistant/
├── app/
│   ├── streamlit_app.py          # Main UI
│   ├── db.py                     # Postgres/SQLite data layer
│   ├── chat_router.py            # Local query routing
│   ├── sec_financials.py         # SEC EDGAR retrieval
│   └── press_release_loader.py
├── agent/
│   ├── financial_assistant_agent.py  # Vertex AI ADK agent
│   ├── tools.py                      # ADK tool functions
│   └── runner_service.py             # ADK runner + fallback
├── training/
│   ├── train_regression.py
│   └── train_classification.py
├── notebooks/
│   ├── train_regression.ipynb
│   └── train_classification.ipynb
├── deployment/
│   ├── deploy_regression.py
│   ├── deploy_classification.py
│   ├── regression/inference.py
│   └── classification/inference.py
├── inference/
│   ├── regression_inference.py
│   ├── classification_inference.py
│   ├── sagemaker_client.py
│   ├── bedrock_summary.py
│   └── azure_summary.py
├── sql/
│   ├── create_tables.sql
│   └── seed_data.sql
├── data/
│   └── press_releases.json
├── models/                       # Trained artifacts
├── docker-compose.yml            # Local Postgres
├── .env.example
└── requirements.txt
```

## Features

- **Streamlit UI** with conversational chat panel, dynamic results section, and dedicated ML predictions page
- **Postgres database** with 20 sample property records (SQLite fallback for local demo)
- **SEC EDGAR integration** with live filing metadata and structured 10-K/10-Q metrics (enable live extraction with `USE_SEC_EDGAR=true`)
- **Press release insights** stored in JSON with search and multi-cloud summarization
- **Random Forest regression** on California Housing dataset
- **Logistic Regression classification** on UCI Bank Marketing dataset
- **SageMaker deployment scripts** with local/cloud inference switching
- **Vertex AI ADK chatbot** with tool-based routing across data sources
- **AWS Bedrock** and **Azure OpenAI** summarization integrations

## Local Setup

### 1. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

For local demo, leave `DATABASE_URL` unset to use SQLite. Set GCP/AWS/Azure variables only when deploying cloud features.

### 3. Start Postgres (optional)

```bash
docker compose up -d
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/real_estate_finance
```

### 4. Train ML models

```bash
python training/train_regression.py
python training/train_classification.py
```

### 5. Run the application

```bash
streamlit run app/streamlit_app.py
```

## Chatbot Routing

The assistant routes questions using Vertex AI ADK tools when `USE_VERTEX_ADK=true` and `GOOGLE_CLOUD_PROJECT` are configured:

| Question Type | Tool / Source |
|---|---|
| Revenue, net income, expenses | `get_financial_overview` |
| Property filters (metro, type) | `query_property_financials` |
| SEC 10-K / 10-Q / last quarter | `query_sec_financials` |
| Acquisitions, expansions, announcements | `query_press_releases` |
| Ambiguous questions | `route_user_question` |

When Vertex AI is unavailable, the same routing logic runs locally via `app/chat_router.py`.

### Data flow (chat query)

```text
User question (Streamlit Assistant page)
    → agent/runner_service.py (ask_assistant)
    → Vertex AI ADK agent (Gemini + tools) OR app/chat_router.py fallback
    → Tools query Postgres/SQLite, SEC EDGAR, or press_releases.json
    → JSON + dataframe returned to Results panel
```

Example: *"Show industrial properties in Chicago"* → ADK calls `query_property_financials(metro_area="Chicago", property_type="Industrial")` → `app/db.py` filters 20 seeded properties → table shown in Results.

## SageMaker Deployment

### Prerequisites

- AWS credentials configured
- SageMaker execution role exported as `SAGEMAKER_ROLE`

### Deploy regression model

```bash
export SAGEMAKER_ROLE=arn:aws:iam::<account-id>:role/<sagemaker-role>
python deployment/deploy_regression.py --endpoint-name housing-regression-endpoint
```

### Deploy classification model

```bash
python deployment/deploy_classification.py --endpoint-name bank-classification-endpoint
```

### Connect Streamlit to endpoints

```bash
export SAGEMAKER_REGRESSION_ENDPOINT=housing-regression-endpoint
export SAGEMAKER_CLASSIFICATION_ENDPOINT=bank-classification-endpoint
```

## Vertex AI ADK Setup

1. Enable Vertex AI API in your GCP project
2. Authenticate: `gcloud auth application-default login`
3. Configure environment:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_GENAI_USE_VERTEXAI=true
export USE_VERTEX_ADK=true
export VERTEX_MODEL=gemini-2.5-flash
```

4. Run the app — the ADK agent in `agent/financial_assistant_agent.py` will handle chat queries

## Multi-Cloud Integrations

### AWS Bedrock

```bash
export USE_BEDROCK=true
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
```

### Azure OpenAI

```bash
export USE_AZURE_OPENAI=true
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

## Model Evaluation Results

### Regression (Random Forest, California Housing)

- RMSE: 0.5061
- MAE: 0.3278
- R²: 0.8045

### Classification (Logistic Regression, Bank Marketing)

- Accuracy: 0.8457
- Precision: 0.4182
- Recall: 0.8147
- F1 Score: 0.5527
- Confusion Matrix: `[[6786, 1199], [196, 862]]`

## Demo Script

1. Open **Assistant** and ask: "Did the company announce any acquisitions recently?"
2. Ask: "Show industrial properties in the Chicago region with revenue details"
3. Ask: "What was the net income reported last quarter?"
4. Open **Press Releases** to view Bedrock/Azure summaries
5. Open **SEC Financials** for 10-K/10-Q metrics
6. Open **ML Predictions** and run SageMaker/local inference for both models

## Public Deployment (Streamlit Community Cloud)

### 1. Push to GitHub

```bash
git add .
git commit -m "Financial Assistant submission"
git remote add origin https://github.com/YOUR_USERNAME/real-estate-financial-assistant.git
git branch -M main
git push -u origin main
```

Do **not** commit `.env` (already in `.gitignore`).

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. **Create app** → select your repo
3. **Main file path:** `app/streamlit_app.py`
4. **Python version:** 3.11 or 3.12
5. **Secrets:** paste values from `.streamlit/secrets.toml.example` (use your real AWS keys)
6. Deploy and copy the public URL (e.g. `https://your-app.streamlit.app`)

**Notes for cloud hosting:**

- Leave `DATABASE_URL` unset → app uses SQLite and auto-seeds from `sql/seed_data.sql`
- SageMaker + Bedrock work when AWS keys are in Secrets
- Vertex ADK on Streamlit Cloud may use local fallback unless you add a GCP service account to Secrets

### 3. Submission links

Add these to your course submission and Google Drive README:

```text
Public deployment URL: https://real-estate-financial-assistant-07.streamlit.app/
GitHub repository: https://github.com/apekshaan07/real-estate-financial-assistant
Google Drive folder: https://drive.google.com/drive/folders/1_yN5reWC-yJy1NPsiSHQGyp12uAyxeNH

```

## Google Drive Submission Package

Create a folder named `Real-Estate-Financial-Assistant` and upload:

| Item | Location in project |
|------|---------------------|
| Source code | Full project (or zip from `./scripts/package_submission.sh`) |
| Requirements | `requirements.txt` |
| SQL scripts | `sql/create_tables.sql`, `sql/seed_data.sql` |
| Training notebooks | `notebooks/train_regression.ipynb`, `notebooks/train_classification.ipynb` |
| Model artifacts | `models/*.joblib`, `models/*_metrics.json` |
| Inference code | `inference/`, `deployment/*/inference.py` |
| README | `README.md` |
| Demo videos | App demo + cloud setup recordings |
| Deployment URL | Text file or link in README |

```bash
chmod +x scripts/package_submission.sh
./scripts/package_submission.sh
```

Set Drive sharing to **Anyone with the link can view**.

## Deliverables Checklist

- [x] Source code
- [x] `requirements.txt`
- [x] SQL scripts (`sql/create_tables.sql`, `sql/seed_data.sql`)
- [x] Training notebooks
- [x] Model artifacts (`models/*.joblib`)
- [x] Inference code (`inference/`, `deployment/*/inference.py`)
- [x] README with setup, architecture, routing, and deployment steps
