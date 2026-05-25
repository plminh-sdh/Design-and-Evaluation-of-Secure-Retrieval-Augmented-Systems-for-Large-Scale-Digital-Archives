# Environment Setup Guide

Before running the notebook, complete the following steps to configure your environment.

---

## 1. Set Up a Free Qdrant Cluster

1. Create a free Qdrant Cloud cluster: https://qdrant.tech/documentation/cloud-quickstart/
2. After the cluster is successfully created, copy your Cluster URL and API Key from the Qdrant dashboard.
3. Paste them into the `.env` file as:
   - `QDRANT_CLUSTER_URL`
   - `QDRANT_CLUSTER_API_KEY`

---

## 2. Set Up HuggingFace Account & Dataset Access

The Viet Doc VQA dataset is gated and requires user authentication.

1. Visit the dataset page: https://huggingface.co/datasets/5CD-AI/Viet-Doc-VQA
2. Click **"Access repository"** or **"Agree to terms"** to gain access.
3. Go to your HuggingFace settings → **Access Tokens**.
4. Click **"New token"**, select **Read** permission.
5. Copy the generated token (format: `hf_xxxxxxxxxxxxxxx`).
6. Save it to the `.env` file as: `HUGGING_FACE_TOKEN`
7. When running the notebook, you may be asked to authenticate using the token once.

---

## 3. Install Tesseract OCR (for Image Processing / OCR)

This project requires Tesseract with Vietnamese language support. If you're running this notebook on Linux, skip this step, as the notebook automatically installs Tesseract to your machine.

- Windows installation guide: https://github.com/UB-Mannheim/tesseract/wiki

After installation, ensure that the Vietnamese language package is included.

Typical installation paths on Windows:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
C:\Program Files (x86)\Tesseract-OCR\tesseract.exe
```

Ensure this path is accessible by your environment or added to your system PATH.

---

## 4. Set Up OpenAI API Access (Required for Embeddings & LLM)

To generate embeddings and run the RAG pipeline, you must configure an OpenAI API key.
⚠️ Note: A ChatGPT Plus subscription does not automatically provide API credits — API usage is billed separately.

Follow the steps below to set up your API key:

### 4.1 Create an OpenAI API Key

- Go to the OpenAI API dashboard: https://platform.openai.com
- Log in using the same account you use for ChatGPT.
- In the left menu, click "API Keys" or go directly to: https://platform.openai.com/settings/organization/api-keys
- Click “Create new secret key”.
- Copy the generated key (it starts with `sk-...`).
- Save your secret key to the `.env` file as: `OPENAI_API_KEY`
  ⚠️ You will not be able to view this key again later, so store it safely.

### 4.2 Add Billing (Required to Avoid Quota Errors)

OpenAI requires a payment method to enable API usage.

- Go to the Billing page: https://platform.openai.com/settings/organization/billing/overview
- Add a payment method (credit card, debit card, etc.).
- Add prepaid credits (recommended: $5).

⚠️ **Important cost notice**:  
While embedding the Viet Doc VQA dataset is relatively inexpensive, the overall experimental pipeline—including document classification, graph construction, agentic query refinement, k-tuning, and evaluation—can incur **significantly higher API costs**, as these steps repeatedly invoke large language models and process substantial token volumes.

To avoid unexpected charges, we **strongly recommend** converting and experimenting with a **small subset of the dataset (e.g., ~200 records)** during development and evaluation. This subset is sufficient to reproduce the methodology and analyze relative retrieval behavior without incurring unnecessary computational expense.

If you skip this step, you will encounter:

```
You exceeded your current quota (insufficient_quota)
```

### 4.3 Verify the API Key (Optional)

Inside a Python cell, run:

```
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
resp = client.models.list()

print("OpenAI API connection successful!")
```

If this prints a model list, your API key is working correctly.

---

## 5. Set Up Neo4j Graph Database

Graph-based retrieval components require a running Neo4j database.

You can use either:

- Neo4j AuraDB (Cloud – Free Tier): https://neo4j.com/cloud/aura/
- Local Neo4j Desktop: https://neo4j.com/download/

After creating the database, note the following connection details:

- Bolt URI (e.g., `neo4j+s://xxxx.databases.neo4j.io`)
- Username (default is usually `neo4j`)
- Password (set during database creation)

Add the details to the `.env` file as:

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

---

## 6. Google Colab Users (Environment Variables via Secrets)

If you are running the notebook on **Google Colab**, the `.env` file is **not required**. Instead, you should store all sensitive credentials using **Colab Secrets**, which are securely managed and automatically exposed as environment variables.

1. Open your notebook in Google Colab.
2. In the left sidebar, click the **🔑 Secrets** icon.
3. Click **“Add a new secret”**.
4. Add the following key–value pairs (one secret per entry):
   | Secret Name | Description |
   | ------------------------ | ----------------------------------- |
   | `HUGGING_FACE_TOKEN` | HuggingFace access token (`hf_...`) |
   | `QDRANT_CLUSTER_URL` | Qdrant Cloud cluster URL |
   | `QDRANT_CLUSTER_API_KEY` | Qdrant Cloud API key |
   | `OPENAI_API_KEY` | Your OpenAI API key (`sk-...`) |
   | `NEO4J_URI` | Neo4j Bolt URI |
   | `NEO4J_USERNAME` | Neo4j username |
   | `NEO4J_PASSWORD` | Neo4j password |
5. Ensure **Notebook access** is enabled for each secret.
6. Restart the runtime if variables are not detected.
