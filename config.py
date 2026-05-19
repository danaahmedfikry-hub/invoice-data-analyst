import os
import json
import re
import io
import base64
import time
import uuid
import hashlib
import unicodedata
import warnings
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import faiss
from openai import AzureOpenAI
from dotenv import load_dotenv
from langsmith import Client as LangSmithClient
from langsmith import traceable

warnings.filterwarnings("ignore")
load_dotenv()

AZURE_OPENAI_API_KEY            = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT           = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL              = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")
AZURE_OPENAI_API_VERSION        = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_EMBEDDING_MODEL    = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
AZURE_DOC_INTELLIGENCE_KEY      = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME            = os.getenv("AZURE_CONTAINER_NAME", "con")
SQLITE_DB_PATH                  = os.getenv("SQLITE_DB_PATH", "./data/Final_Project.db")
FAISS_DB_PATH                   = os.getenv("FAISS_DB_PATH", "./data/faiss_db")

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY    = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT    = os.getenv("LANGCHAIN_PROJECT", "final project")
LANGCHAIN_ENDPOINT   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

os.environ["LANGCHAIN_TRACING_V2"] = str(LANGCHAIN_TRACING_V2).lower()
os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_ENDPOINT"]   = LANGCHAIN_ENDPOINT

TOP_K     = 10
EMBED_DIM = 1536

Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(FAISS_DB_PATH).mkdir(parents=True, exist_ok=True)

FAISS_INDEX_FILE = str(Path(FAISS_DB_PATH) / "invoice_chunks.faiss")
FAISS_META_FILE  = str(Path(FAISS_DB_PATH) / "invoice_chunks_meta.json")

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

_ls_client = None
if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    try:
        _ls_client = LangSmithClient(api_url=LANGCHAIN_ENDPOINT, api_key=LANGCHAIN_API_KEY)
    except Exception:
        _ls_client = None

_AR_NUM_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def normalize_arabic_numerals(text: str) -> str:
    if not text:
        return text
    return text.translate(_AR_NUM_TABLE)

def is_arabic(text: str) -> bool:
    if not text:
        return False
    return any("\u0600" <= ch <= "\u06FF" for ch in text)

def safe_json_serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)

def _sanitize_col(col: str) -> str:
    return re.sub(r"[^\w]", "_", str(col)).strip("_")[:60]

def _strip_md_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()
