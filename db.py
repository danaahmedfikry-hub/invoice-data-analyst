import sqlite3
import json
import time
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from datetime import datetime
from azure.storage.blob import BlobServiceClient
import uuid

from config import (
    SQLITE_DB_PATH, FAISS_DB_PATH, FAISS_INDEX_FILE, FAISS_META_FILE,
    AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME,
    AZURE_OPENAI_EMBEDDING_MODEL, EMBED_DIM, TOP_K,
    openai_client, normalize_arabic_numerals, safe_json_serialize, _sanitize_col,
    LANGCHAIN_PROJECT
)
from langsmith import traceable


def init_db():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.text_factory = str
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS invoices (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            blob_name         TEXT,
            original_filename TEXT,
            ingested_at       TEXT
        );
        CREATE TABLE IF NOT EXISTS invoice_line_items (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id INTEGER,
            FOREIGN KEY (invoice_id) REFERENCES invoices(id)
        );
        CREATE TABLE IF NOT EXISTS raw_invoices (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            blob_name         TEXT,
            original_filename TEXT,
            raw_json          TEXT,
            ingested_at       TEXT
        );
    """)
    conn.commit()
    conn.close()


def upload_to_blob(file_bytes: bytes, filename: str) -> tuple[str, str]:
    blob_service     = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(AZURE_CONTAINER_NAME)
    try:
        container_client.create_container()
    except Exception:
        pass
    blob_name   = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{filename}"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(file_bytes, overwrite=True)
    return blob_client.url, blob_name


def _execute_ddl_safe(cursor: sqlite3.Cursor, ddl: str):
    ddl_safe = (
        ddl
        .replace("CREATE TABLE invoices",           "CREATE TABLE IF NOT EXISTS invoices")
        .replace("CREATE TABLE invoice_line_items", "CREATE TABLE IF NOT EXISTS invoice_line_items")
    )
    for stmt in [s.strip() for s in ddl_safe.split(";") if s.strip()]:
        try:
            cursor.execute(stmt)
        except sqlite3.OperationalError:
            pass


def _add_missing_columns(cursor: sqlite3.Cursor, table: str, needed_keys: list):
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    for col in needed_keys:
        safe_col = _sanitize_col(col)
        if safe_col not in existing:
            try:
                cursor.execute(f"ALTER TABLE [{table}] ADD COLUMN [{safe_col}] TEXT")
            except sqlite3.OperationalError:
                pass


def _flatten_values(d: dict) -> dict:
    flat = {}
    for k, v in d.items():
        safe_k = _sanitize_col(k)
        if isinstance(v, dict):
            for key in ("amount", "value", "content", "text"):
                if key in v:
                    flat[safe_k] = v[key]
                    break
            else:
                flat[safe_k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, list):
            flat[safe_k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[safe_k] = v
    return flat


def _coerce_value(v):
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    s = str(v).strip()
    try:
        if "." in s:
            return float(s)
        return int(s)
    except (ValueError, TypeError):
        return s


def store_in_sqlite(llm_result: dict, blob_name: str, original_filename: str) -> dict:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.text_factory = str
    cursor = conn.cursor()
    report = {}

    try:
        ddl = llm_result.get("ddl", "")
        if ddl:
            _execute_ddl_safe(cursor, ddl)
            conn.commit()
        report["tables_created"] = True

        normalized   = llm_result.get("normalized_data", {})
        invoice_data = _flatten_values(normalized.get("invoice", {}))
        invoice_data["blob_name"]         = blob_name
        invoice_data["original_filename"] = original_filename
        invoice_data["ingested_at"]       = datetime.now().isoformat()

        _add_missing_columns(cursor, "invoices", list(invoice_data.keys()))
        conn.commit()

        cursor.execute("PRAGMA table_info(invoices)")
        table_cols = {row[1] for row in cursor.fetchall()}
        filtered   = {k: v for k, v in invoice_data.items() if k in table_cols}
        if not filtered:
            filtered = {
                "blob_name":         blob_name,
                "original_filename": original_filename,
                "ingested_at":       datetime.now().isoformat(),
            }

        cols         = ", ".join(f"[{k}]" for k in filtered.keys())
        placeholders = ", ".join(["?"] * len(filtered))
        cursor.execute(
            f"INSERT INTO invoices ({cols}) VALUES ({placeholders})",
            [_coerce_value(v) for v in filtered.values()]
        )
        invoice_id           = cursor.lastrowid
        report["invoice_id"] = invoice_id

        line_items = normalized.get("line_items", [])
        li_count   = 0
        if line_items and invoice_id:
            all_li_keys = set()
            for item in line_items:
                all_li_keys.update(_flatten_values(item).keys())
            all_li_keys.add("invoice_id")
            _add_missing_columns(cursor, "invoice_line_items", list(all_li_keys))
            conn.commit()

            cursor.execute("PRAGMA table_info(invoice_line_items)")
            li_cols = {row[1] for row in cursor.fetchall()}

            for item in line_items:
                flat_item               = _flatten_values(item)
                flat_item["invoice_id"] = invoice_id
                filtered_item = {k: v for k, v in flat_item.items() if k in li_cols}
                if filtered_item:
                    cols         = ", ".join(f"[{k}]" for k in filtered_item.keys())
                    placeholders = ", ".join(["?"] * len(filtered_item))
                    cursor.execute(
                        f"INSERT INTO invoice_line_items ({cols}) VALUES ({placeholders})",
                        [_coerce_value(v) for v in filtered_item.values()]
                    )
                    li_count += 1

        conn.commit()
        report["line_items_inserted"] = li_count
        report["success"]             = True

    except Exception as e:
        conn.rollback()
        report["success"] = False
        report["error"]   = str(e)
        try:
            cursor.execute(
                "INSERT INTO raw_invoices (blob_name, original_filename, raw_json, ingested_at) VALUES (?,?,?,?)",
                (blob_name, original_filename,
                 json.dumps(llm_result, default=safe_json_serialize, ensure_ascii=False),
                 datetime.now().isoformat())
            )
            conn.commit()
            report["fallback"] = "Stored as raw JSON in raw_invoices table"
        except Exception as fb_err:
            report["fallback_error"] = str(fb_err)
    finally:
        conn.close()
    return report


def load_invoices_from_db() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.text_factory = str
    invoices_df   = pd.DataFrame()
    line_items_df = pd.DataFrame()
    try:
        invoices_df   = pd.read_sql("SELECT * FROM invoices", conn)
    except Exception:
        pass
    try:
        line_items_df = pd.read_sql("SELECT * FROM invoice_line_items", conn)
    except Exception:
        pass
    conn.close()
    return invoices_df, line_items_df


def build_sql_context() -> str:
    try:
        conn   = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        parts  = []
        for tbl in tables:
            cursor.execute(f"PRAGMA table_info({tbl})")
            cols     = cursor.fetchall()
            col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            parts.append(f"TABLE {tbl} ({col_defs})")
        for tbl in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
            cnt = cursor.fetchone()[0]
            parts.append(f"  -- {tbl} has {cnt} row(s)")
        conn.close()
        return "\n".join(parts)
    except Exception:
        return "Schema not available."


def execute_sql_safe(query: str) -> tuple:
    import re
    clean = query.strip()
    clean = re.sub(r"^[;\s]+", "", clean).strip()
    clean = clean.rstrip(";").strip()
    if not clean:
        return None, "Empty query."
    if not re.match(r"(?i)^\s*select\b", clean):
        return None, "Only SELECT statements are permitted."
    try:
        conn   = sqlite3.connect(SQLITE_DB_PATH)
        conn.text_factory = str
        result = pd.read_sql(clean, conn)
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)


def _build_header_chunk(row: pd.Series) -> str:
    HEADER_FIELDS = {
        "invoice_number", "invoice_date", "due_date", "vendor_name", "vendor_name_ar",
        "customer_name", "customer_name_ar", "total_amount", "tax_amount", "subtotal",
        "discount_amount", "currency_code", "payment_terms", "payment_method",
        "notes", "blob_name", "original_filename", "ingested_at", "id",
    }
    lines = [f"[INVOICE HEADER — ID: {row.get('id', '?')} | File: {row.get('original_filename', '?')}]"]
    for col, val in row.items():
        if pd.notna(val) and str(val).strip():
            clean_val = normalize_arabic_numerals(str(val))
            if col in HEADER_FIELDS or any(kw in col for kw in ("invoice", "vendor", "customer", "total", "tax", "date", "payment", "currency", "amount", "subtotal", "discount")):
                lines.append(f"  {col}: {clean_val}")
    return "\n".join(lines)


def _build_line_item_chunk(row: pd.Series, invoice_row: pd.Series, item_index: int) -> str:
    best_invoice_id   = invoice_row.get("invoice_id") or invoice_row.get("invoice_number") or invoice_row.get("id") or "?"
    original_filename = invoice_row.get("original_filename", "?")
    vendor            = invoice_row.get("vendor_name") or invoice_row.get("vendor_name_ar") or "?"
    invoice_num_val   = invoice_row.get("invoice_number") or invoice_row.get("invoice_id") or "?"
    invoice_date_val  = invoice_row.get("invoice_date") or "?"
    lines = [
        f"[LINE ITEM {item_index} — Invoice #{best_invoice_id} | File: {original_filename}]",
        f"  invoice_number: {invoice_num_val}",
        f"  invoice_date: {invoice_date_val}",
        f"  vendor_name: {vendor}",
    ]
    for col, val in row.items():
        if col in ("id", "invoice_id", "invoice_number", "invoice_date"):
            continue
        if pd.notna(val) and str(val).strip():
            clean_val = normalize_arabic_numerals(str(val))
            lines.append(f"  {col}: {clean_val}")
    return "\n".join(lines)


def build_semantic_chunks(invoices_df: pd.DataFrame, line_items_df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    all_texts = []
    all_metas = []
    has_li      = not line_items_df.empty and "invoice_id" in line_items_df.columns
    invoice_map = {str(r["id"]): r for _, r in invoices_df.iterrows()} if "id" in invoices_df.columns else {}

    for _, inv_row in invoices_df.iterrows():
        invoice_id = str(inv_row.get("id", ""))
        filename   = str(inv_row.get("original_filename", ""))
        header_text = _build_header_chunk(inv_row)
        all_texts.append(header_text)
        all_metas.append({
            "invoice_id":        invoice_id,
            "original_filename": filename,
            "chunk_type":        "header",
            "chunk_index":       0,
            "ingested_at":       datetime.now().isoformat(),
        })
        if has_li:
            try:
                li_subset = line_items_df[line_items_df["invoice_id"].astype(str) == invoice_id]
            except Exception:
                li_subset = pd.DataFrame()
            for li_idx, (_, li_row) in enumerate(li_subset.iterrows(), start=1):
                li_text = _build_line_item_chunk(li_row, inv_row, li_idx)
                all_texts.append(li_text)
                all_metas.append({
                    "invoice_id":        invoice_id,
                    "original_filename": filename,
                    "chunk_type":        "line_item",
                    "chunk_index":       li_idx,
                    "ingested_at":       datetime.now().isoformat(),
                })
    return all_texts, all_metas


@traceable(name="embed_texts", run_type="embedding", project_name=LANGCHAIN_PROJECT)
def _embed(texts: list[str]) -> list[list[float]]:
    all_emb    = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batch = [t if t.strip() else "empty invoice record" for t in batch]
        for attempt in range(4):
            try:
                response = openai_client.embeddings.create(
                    model=AZURE_OPENAI_EMBEDDING_MODEL,
                    input=batch,
                )
                all_emb.extend([item.embedding for item in response.data])
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    return all_emb


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    if Path(FAISS_INDEX_FILE).exists() and Path(FAISS_META_FILE).exists():
        index = faiss.read_index(FAISS_INDEX_FILE)
        if index.d != EMBED_DIM:
            return faiss.IndexFlatIP(EMBED_DIM), []
        with open(FAISS_META_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        index    = faiss.IndexFlatIP(EMBED_DIM)
        metadata = []
    return index, metadata


def _save_index(index: faiss.IndexFlatIP, metadata: list[dict]) -> None:
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


@traceable(name="ingest_into_vector_store", run_type="chain", project_name=LANGCHAIN_PROJECT)
def ingest_into_vector_store(invoices_df: pd.DataFrame, line_items_df: pd.DataFrame, progress_cb=None) -> dict:
    if invoices_df.empty:
        return {"chunks_upserted": 0, "invoices_processed": 0}
    if progress_cb:
        progress_cb(10, "Building semantic chunks…")
    all_docs, all_metas = build_semantic_chunks(invoices_df, line_items_df)
    if progress_cb:
        progress_cb(40, f"{len(all_docs)} chunks built — embedding…")
    raw_embeddings = _embed(all_docs)
    if progress_cb:
        progress_cb(80, "Normalising & building FAISS index…")
    vectors = np.array(raw_embeddings, dtype=np.float32)
    vectors = _l2_normalize(vectors)
    index   = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    for i, meta in enumerate(all_metas):
        meta["text"] = all_docs[i]
    if progress_cb:
        progress_cb(95, "Saving index to disk…")
    _save_index(index, all_metas)
    if progress_cb:
        progress_cb(100, f"✅ {len(all_docs)} chunks stored")
    return {"chunks_upserted": len(all_docs), "invoices_processed": len(invoices_df)}


def clear_vector_store() -> None:
    for p in [FAISS_INDEX_FILE, FAISS_META_FILE]:
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass


def vector_store_stats() -> dict:
    if Path(FAISS_INDEX_FILE).exists():
        index = faiss.read_index(FAISS_INDEX_FILE)
        return {"total_chunks": int(index.ntotal)}
    return {"total_chunks": 0}


@traceable(name="retrieve_chunks", run_type="retriever", project_name=LANGCHAIN_PROJECT)
def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    query           = normalize_arabic_numerals(query)
    index, metadata = _load_index()
    if index.ntotal == 0:
        return []
    q_emb           = np.array(_embed([query]), dtype=np.float32)
    q_emb           = _l2_normalize(q_emb)
    k               = min(top_k, index.ntotal)
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        results.append({
            "text":              meta.get("text", ""),
            "invoice_id":        meta.get("invoice_id"),
            "original_filename": meta.get("original_filename"),
            "chunk_type":        meta.get("chunk_type", "unknown"),
            "chunk_index":       meta.get("chunk_index"),
            "similarity":        round(float(score), 4),
        })
    return results
