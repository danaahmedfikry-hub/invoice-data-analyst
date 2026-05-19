import json
import re
import io
import base64
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from langsmith import traceable

from config import (
    openai_client, AZURE_OPENAI_MODEL, AZURE_DOC_INTELLIGENCE_KEY,
    AZURE_DOC_INTELLIGENCE_ENDPOINT, AZURE_STORAGE_CONNECTION_STRING,
    AZURE_CONTAINER_NAME, TOP_K, LANGCHAIN_PROJECT,
    normalize_arabic_numerals, safe_json_serialize, _strip_md_fences
)
from db import (
    upload_to_blob, store_in_sqlite, retrieve_chunks,
    build_sql_context, execute_sql_safe
)


@traceable(name="validate_invoice", run_type="llm", project_name=LANGCHAIN_PROJECT)
def is_valid_invoice(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    ext = Path(filename).suffix.lower()

    if ext in (".jpg", ".jpeg", ".png", ".tiff"):
        b64  = base64.b64encode(file_bytes).decode("utf-8")
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.strip('.')}"
        content = [
            {"type": "text",      "text": "Look at this document image."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    elif ext == ".pdf":
        try:
            import pypdf
            text = " ".join(
                page.extract_text() or ""
                for page in pypdf.PdfReader(io.BytesIO(file_bytes)).pages
            ).strip()
        except Exception:
            return False, f"**{filename}** could not be read — it may be corrupted or password-protected."
        if not text:
            return True, ""
        content = [{"type": "text", "text": f"Document text:\n{text[:3000]}"}]
    else:
        return False, f"**{filename}** has an unsupported file type: `{ext}`"

    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document classifier. "
                        "Your only job is to decide if a document is an invoice or not. "
                        "An invoice is a commercial document issued by a seller to a buyer, "
                        "listing goods or services, quantities, prices, and a total amount due. "
                        "Respond with ONLY a valid JSON object: "
                        '{"is_invoice": true/false, "reason": "one short sentence"}'
                    ),
                },
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=80,
        )
        raw    = _strip_md_fences(response.choices[0].message.content.strip())
        result = json.loads(raw)
        if result.get("is_invoice"):
            return True, ""
        return False, f"**{filename}** — {result.get('reason', 'not recognized as an invoice.')}"
    except Exception:
        return True, ""


@traceable(name="extract_invoice_data_doc_intelligence", run_type="tool", project_name=LANGCHAIN_PROJECT)
def extract_invoice_data(blob_url: str) -> dict:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential

    doc_client = DocumentIntelligenceClient(
        endpoint=AZURE_DOC_INTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOC_INTELLIGENCE_KEY),
    )
    poller = doc_client.begin_analyze_document(
        model_id="prebuilt-invoice",
        body={"urlSource": blob_url},
        locale="ar",
    )
    result    = poller.result()
    extracted = {}

    if result.documents:
        doc = result.documents[0]
        for field_name, field in doc.fields.items():
            if field is None:
                continue
            val = None
            for attr in ("value_string", "value_currency", "value_date",
                         "value_number", "value_integer", "value_phone_number",
                         "value_address", "value_selection_mark", "content"):
                candidate = getattr(field, attr, None)
                if candidate is not None:
                    if attr == "value_currency" and hasattr(candidate, "amount"):
                        val = str(candidate.amount)
                    elif isinstance(candidate, str) and candidate.strip():
                        val = candidate.strip()
                    elif not isinstance(candidate, str):
                        val = str(candidate)
                    if val:
                        break
            if val is None and hasattr(field, "value") and field.value is not None:
                val = str(field.value)
            if val:
                val = normalize_arabic_numerals(val)
            extracted[field_name] = {
                "value":      val,
                "confidence": round(field.confidence, 4) if field.confidence else None,
            }

        items_field = doc.fields.get("Items")
        if items_field and hasattr(items_field, "value_array") and items_field.value_array:
            line_items = []
            for item in items_field.value_array:
                if hasattr(item, "value_object") and item.value_object:
                    li = {}
                    for k, v in item.value_object.items():
                        raw = None
                        for attr in ("value_string", "content", "value_currency"):
                            cand = getattr(v, attr, None)
                            if attr == "value_currency" and cand and hasattr(cand, "amount"):
                                raw = str(cand.amount)
                                break
                            if cand and str(cand).strip():
                                raw = str(cand).strip()
                                break
                        if raw is None and hasattr(v, "value") and v.value is not None:
                            raw = str(v.value)
                        if raw:
                            raw = normalize_arabic_numerals(raw)
                        li[k] = raw
                    line_items.append(li)
            extracted["LineItems"] = {"value": line_items, "confidence": None}

    return extracted


@traceable(name="llm_generate_schema_and_sql", run_type="llm", project_name=LANGCHAIN_PROJECT)
def llm_generate_schema_and_sql(extracted_json: dict) -> dict:
    prompt = f"""You are a senior data engineer specializing in Arabic and bilingual (Arabic/English) invoice processing.

Analyze this invoice data extracted by Azure Document Intelligence. The data may contain Arabic field names, Arabic text values, and Eastern Arabic numerals.

Extracted invoice data:
{json.dumps(extracted_json, indent=2, default=safe_json_serialize, ensure_ascii=False)}

Your job:
1. Translate ALL Arabic field names to English snake_case. Handle dialectal variation, abbreviations, and spelling differences. Never use a static lookup — reason about the field semantics yourself.
   Examples: رقم الفاتورة→invoice_number, التاريخ→invoice_date, المورد/البائع/اسم_المورد→vendor_name,
   العميل/المشتري→customer_name, الإجمالي/المجموع/المبلغ_الإجمالي→total_amount,
   الضريبة/ضريبة_القيمة_المضافة→tax_amount, سعر_الوحدة/السعر/التكلفة→unit_price,
   الكمية/عدد→quantity, الوصف/البيان/الصنف→description, شروط_الدفع→payment_terms, العملة→currency
2. If a field value contains Arabic text that is a vendor name, customer name, item description, or similar, preserve the original Arabic in a separate column with _ar suffix (e.g., vendor_name_ar).
3. Convert ALL Eastern Arabic/Farsi numerals (٠١٢٣٤٥٦٧٨٩) to Western digits (0123456789) in all values.
4. Extract numeric amount only for currency REAL columns; put symbol in a separate currency_code TEXT column.
5. Design SQLite-compatible tables. Always include in invoices: id INTEGER PRIMARY KEY AUTOINCREMENT, blob_name TEXT, original_filename TEXT, ingested_at TEXT. Always include in invoice_line_items: id INTEGER PRIMARY KEY AUTOINCREMENT, invoice_id INTEGER.
6. DDL must be a single string with statements separated by semicolons.
7. normalized_data values must be Python-serialisable primitives (str, int, float, None).

Respond ONLY with valid JSON (no markdown fences, no explanation):
{{
  "entities": [
    {{"name": "...", "type": "string|number|date|currency|array", "description": "..."}}
  ],
  "schema": {{
    "invoices": {{
      "description": "Main invoice header table",
      "columns": {{
        "column_name": {{"type": "TEXT|REAL|INTEGER|DATE", "nullable": true|false, "description": "..."}}
      }}
    }},
    "invoice_line_items": {{
      "description": "Line items table",
      "columns": {{
        "column_name": {{"type": "TEXT|REAL|INTEGER|DATE", "nullable": true|false, "description": "..."}}
      }}
    }}
  }},
  "ddl": "CREATE TABLE IF NOT EXISTS invoices (...); CREATE TABLE IF NOT EXISTS invoice_line_items (...);",
  "normalized_data": {{
    "invoice": {{"field_name": "value"}},
    "line_items": [{{"field_name": "value"}}]
  }},
  "insights":        [{{"title": "...", "description": "..."}}],
  "anomalies":       [{{"title": "...", "description": "..."}}],
  "recommendations": [{{"title": "...", "description": "..."}}],
  "business_impact": "..."
}}
"""
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4000,
    )
    raw = _strip_md_fences(response.choices[0].message.content.strip())
    return json.loads(raw)


@traceable(name="process_single_invoice", run_type="chain", project_name=LANGCHAIN_PROJECT)
def process_single_invoice(file_bytes: bytes, filename: str) -> dict:
    result = {"filename": filename, "success": False}
    try:
        blob_url, blob_name  = upload_to_blob(file_bytes, filename)
        result["blob_url"]   = blob_url
        result["blob_name"]  = blob_name
        extracted            = extract_invoice_data(blob_url)
        result["extracted"]  = extracted
        llm_result           = llm_generate_schema_and_sql(extracted)
        result["llm_result"] = llm_result
        db_report            = store_in_sqlite(llm_result, blob_name, filename)
        result["db_report"]  = db_report
        if db_report.get("success"):
            result["success"] = True
        else:
            result["error"] = db_report.get("error", "DB storage failed silently")
    except Exception as e:
        result["error"] = str(e)
    return result


@traceable(name="ai_analyze_invoices", run_type="llm", project_name=LANGCHAIN_PROJECT)
def ai_analyze_invoices(invoices_df: pd.DataFrame, line_items_df: pd.DataFrame) -> dict:
    numeric_cols = invoices_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = invoices_df.select_dtypes(include=["object"]).columns.tolist()
    summary = {
        "invoices_count":    len(invoices_df),
        "columns":           invoices_df.columns.tolist(),
        "numeric_stats":     invoices_df[numeric_cols].describe().to_dict() if numeric_cols else {},
        "sample":            invoices_df.head(5).to_dict(orient="records"),
        "categorical_stats": {c: invoices_df[c].value_counts().head(10).to_dict() for c in cat_cols[:5]},
        "line_items_count":  len(line_items_df),
        "line_items_sample": line_items_df.head(5).to_dict(orient="records") if not line_items_df.empty else [],
    }
    prompt = f"""You are a senior financial analyst. Analyze this invoice database summary.

{json.dumps(summary, default=safe_json_serialize, indent=2, ensure_ascii=False)}

Respond ONLY with valid JSON:
{{
  "kpis":            [{{"name":"...","value":"...","description":"..."}}],
  "charts":          [{{"type":"bar|line|pie|scatter|histogram","title":"...","x_column":"...","y_column":"...","color_column":"...","description":"..."}}],
  "insights":        [{{"title":"...","description":"..."}}],
  "anomalies":       [{{"title":"...","description":"..."}}],
  "recommendations": [{{"title":"...","description":"..."}}],
  "business_impact": "..."
}}
Rules:
- kpis: 4-6 key financial metrics
- charts: 4-6 most insightful charts using ONLY real column names from the data
- insights/anomalies/recommendations: 3-4 each
- business_impact: short paragraph
"""
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2500,
    )
    raw = _strip_md_fences(response.choices[0].message.content.strip())
    return json.loads(raw)


def _df_to_natural_text(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "(no rows returned)"
    return df.head(max_rows).to_string(index=False)


@traceable(name="llm_rag_planner_call", run_type="llm", project_name=LANGCHAIN_PROJECT)
def _llm_plan_call(messages: list[dict]):
    return openai_client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=800,
    )


@traceable(name="llm_rag_answerer_call", run_type="llm", project_name=LANGCHAIN_PROJECT)
def _llm_answer_call(messages: list[dict]):
    return openai_client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )


@traceable(name="rag_chat", run_type="chain", project_name=LANGCHAIN_PROJECT)
def rag_chat(user_message: str, chat_history: list[dict], invoices_df: pd.DataFrame, line_items_df: pd.DataFrame) -> dict:
    retrieved = retrieve_chunks(user_message, top_k=TOP_K)

    if retrieved:
        context_block = "\n\n---\n\n".join(
            f"[Chunk {i+1} | Type: {c['chunk_type'].upper()} | Invoice #{c['invoice_id']} | "
            f"File: {c['original_filename']} | Relevance: {int(c['similarity']*100)}%]\n{c['text']}"
            for i, c in enumerate(retrieved)
        )
    else:
        context_block = "No relevant chunks found in the FAISS index. You MUST use SQL to answer."

    schema = build_sql_context()

    planner_system = f"""You are a financial data analyst assistant for an Invoice Intelligence platform.
You support Arabic and English invoices. All numeric values are in Western digits.

You have access to:
1. Retrieved invoice chunks below (semantically matched — each chunk is either a full invoice header OR a single line item)
2. A live SQLite database — you may write SELECT queries

=== DATABASE SCHEMA ===
{schema}

=== RETRIEVED CHUNKS ({len(retrieved)} chunks) ===
{context_block}

=== YOUR TASK ===
Decide how to answer the user's question.

OPTION A — If the chunks contain the exact answer already:
  Respond naturally in plain text. Be direct and concise. Do NOT write SQL.

OPTION B — If you need data not visible in the chunks (aggregations, searches, listings):
  Write ONLY the SQL query wrapped in <sql>...</sql>. Nothing else.
  The query will be executed and you will receive the results to answer naturally.

RULES:
- "total", "sum", "count", "average", "list all", "show me" → use SQL (OPTION B)
- Specific detail clearly visible in a chunk → answer directly (OPTION A)
- SQL must be valid SQLite SELECT only
- If user writes in Arabic → respond in Arabic
"""

    plan_messages = [{"role": "system", "content": planner_system}]
    for turn in chat_history[-10:]:
        plan_messages.append({"role": turn["role"], "content": turn["content"]})
    plan_messages.append({"role": "user", "content": user_message})

    plan_raw    = _llm_plan_call(plan_messages).choices[0].message.content.strip()
    sql_queries = []
    sql_results = []
    sql_errors  = []

    for m in re.finditer(r"<sql>(.*?)</sql>", plan_raw, re.DOTALL | re.IGNORECASE):
        q = m.group(1).strip()
        if q:
            df, err = execute_sql_safe(q)
            sql_queries.append(q)
            sql_errors.append(err)
            sql_results.append(df)

    if not sql_queries:
        display_answer = re.sub(r"<sql>.*?</sql>", "", plan_raw, flags=re.DOTALL | re.IGNORECASE).strip()
        seen_files     = list({c["original_filename"] for c in retrieved if c.get("original_filename")})
        sources        = seen_files or (["faiss index"] if retrieved else ["no data"])
        return {
            "answer": display_answer, "sources": sources,
            "retrieved_chunks": retrieved,
            "sql_queries": [], "sql_results": [], "sql_errors": [],
            "sql_query": None, "sql_result": None, "sql_error": None,
        }

    results_block_parts = []
    for i, (q, df, err) in enumerate(zip(sql_queries, sql_results, sql_errors)):
        if err:
            results_block_parts.append(f"Query {i+1} ERROR: {err}")
        elif df is not None and not df.empty:
            results_block_parts.append(f"Query {i+1} returned {len(df)} row(s):\n{_df_to_natural_text(df)}")
        else:
            results_block_parts.append(f"Query {i+1} returned no rows.")
    results_block = "\n\n".join(results_block_parts)

    answerer_system = f"""You are a financial data analyst assistant for an Invoice Intelligence platform.
You support Arabic and English invoices.

The user asked a question. SQL queries were run and the results are below.
Write a direct, natural, conversational answer based ONLY on these results.

=== SQL RESULTS ===
{results_block}

=== RULES ===
- Answer in plain, natural language — like a human analyst would
- State the exact value(s) from the results
- Do NOT show raw tables, code blocks, or markdown unless it genuinely helps readability
- Do NOT say "Running query..." or "the query will return..." — you already have the results
- Do NOT reproduce the SQL
- If multiple rows: summarise clearly with a short list
- Cite invoice IDs or filenames where relevant
- Format currencies clearly (e.g. 2.00 SAR or $10.00)
- If user wrote in Arabic → respond in Arabic
- Keep it concise — 1-4 sentences for simple lookups, a short list for multi-row results
"""

    display_answer = _llm_answer_call([
        {"role": "system", "content": answerer_system},
        {"role": "user",   "content": user_message},
    ]).choices[0].message.content.strip()

    seen_files = list({c["original_filename"] for c in retrieved if c.get("original_filename")})
    sources    = seen_files or (["faiss index"] if retrieved else ["no data"])
    sources.append("live SQL")

    return {
        "answer": display_answer, "sources": sources,
        "retrieved_chunks": retrieved,
        "sql_queries": sql_queries, "sql_results": sql_results, "sql_errors": sql_errors,
        "sql_query":  sql_queries[0] if sql_queries else None,
        "sql_result": sql_results[0] if sql_results else None,
        "sql_error":  sql_errors[0]  if sql_errors  else None,
    }
