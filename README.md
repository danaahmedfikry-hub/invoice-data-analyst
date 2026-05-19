# 🧾 Invoice Intelligence Platform

> AI-powered platform for extracting, analyzing, and querying Arabic & English invoices using Azure Document Intelligence, GPT-4.1, FAISS vector search, and a RAG chatbot — built with Streamlit.

---

## 📌 Overview

**Invoice Intelligence** is an end-to-end AI platform that automates the processing of invoices in both **Arabic and English**. It extracts structured data from uploaded invoice files, stores them in a local database, and enables natural language querying through a **RAG (Retrieval-Augmented Generation) chatbot** powered by GPT-4.1 and FAISS semantic search.

---

## 🚀 Features

- 📤 **Batch invoice upload** — PDF, JPG, PNG, TIFF
- 🤖 **AI validation** — automatically detects if a file is a valid invoice
- 🌐 **Bilingual support** — Arabic & English invoices fully supported
- 🔍 **Azure Document Intelligence** — extracts fields using the prebuilt-invoice model
- 🧠 **LLM schema generation** — GPT-4.1 translates Arabic field names to English and designs the database schema dynamically
- 🗄️ **SQLite storage** — normalized invoice data stored locally
- 📊 **Analytics dashboard** — AI-generated KPIs and charts
- 💬 **RAG chatbot** — ask questions about your invoices in English or Arabic
- 🔎 **FAISS semantic search** — header + per-line-item chunking strategy
- 📡 **LangSmith tracing** — full observability over every LLM call

---

## 🏗️ Architecture

```
Upload Invoice
      ↓
Azure Blob Storage
      ↓
Azure Document Intelligence (prebuilt-invoice, Arabic locale)
      ↓
GPT-4.1 — translate Arabic fields + generate SQLite schema
      ↓
SQLite Database
      ↓
Semantic Chunking (1 header chunk + N line item chunks)
      ↓
text-embedding-3-small → FAISS Index
      ↓
RAG Chatbot (FAISS retrieval + auto SQL + GPT-4.1 answer)
```

---

## 🗂️ Project Structure

```
invoice-intelligence/
├── app.py              # Streamlit UI
├── ai.py               # LLM calls, invoice processing, RAG chat
├── db.py               # SQLite, FAISS, embeddings, chunking
├── config.py           # Environment variables, clients, constants
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── .gitignore
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/invoice-intelligence.git
cd invoice-intelligence
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**

Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_MODEL` | Model name (e.g. `gpt-4.1`) |
| `AZURE_OPENAI_API_VERSION` | API version |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Embedding model (e.g. `text-embedding-3-small`) |
| `AZURE_DOC_INTELLIGENCE_KEY` | Azure Document Intelligence key |
| `AZURE_DOC_INTELLIGENCE_ENDPOINT` | Azure Document Intelligence endpoint |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string |
| `AZURE_CONTAINER_NAME` | Blob container name |
| `SQLITE_DB_PATH` | Local path for SQLite database |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing (`true`/`false`) |
| `LANGCHAIN_API_KEY` | LangSmith API key |
| `LANGCHAIN_PROJECT` | LangSmith project name |
| `LANGCHAIN_ENDPOINT` | LangSmith endpoint URL |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | GPT-4.1 (Azure OpenAI) |
| Embeddings | text-embedding-3-small |
| Document Extraction | Azure Document Intelligence |
| Vector Store | FAISS |
| Database | SQLite |
| Storage | Azure Blob Storage |
| Tracing | LangSmith |
| Language | Python |

---

## 📋 How to Use

1. **Process Invoices** — upload one or more invoice files and click Run Pipeline
2. **Analytics Dashboard** — view AI-generated KPIs and charts from your invoice data
3. **RAG Chat** — click "Index Invoices" then ask questions in English or Arabic
4. **Database Explorer** — browse raw invoice data and run custom SQL queries

---

## 🔒 Security Notes

- Never commit your `.env` file — it is excluded via `.gitignore`
- Use `.env.example` as a template for sharing configuration structure
- Rotate your Azure keys regularly

---

## 📄 License

This project is for educational and internal use.
