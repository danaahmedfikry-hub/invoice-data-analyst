import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import (
    AZURE_OPENAI_MODEL, AZURE_OPENAI_EMBEDDING_MODEL,
    TOP_K, LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, LANGCHAIN_ENDPOINT,
    is_arabic
)
from db import (
    init_db, load_invoices_from_db, ingest_into_vector_store,
    clear_vector_store, vector_store_stats, execute_sql_safe
)
from ai import (
    is_valid_invoice, process_single_invoice, ai_analyze_invoices, rag_chat
)

st.set_page_config(
    page_title="Invoice Intelligence",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .stApp { background: #080c10; color: #cdd6f4; }
  section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2a3a; }
  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
  .arabic-text { direction: rtl; text-align: right; font-family: 'Segoe UI', 'Arial', sans-serif; unicode-bidi: embed; }
  .pipeline-step { background: linear-gradient(135deg, #0d1b2a 0%, #112233 100%); border: 1px solid #1e3a5f; border-left: 3px solid #00b4d8; border-radius: 8px; padding: 16px 20px; margin: 8px 0; transition: border-color .2s, transform .1s; }
  .pipeline-step:hover { border-left-color: #48cae4; transform: translateX(2px); }
  .pipeline-step.active { border-left-color: #f0c040; background: linear-gradient(135deg, #1a1500 0%, #120f00 100%); border-color: #f0c04066; }
  .pipeline-step.done   { border-left-color: #2ea043; background: linear-gradient(135deg, #0d2b1a 0%, #071d11 100%); border-color: #2ea04366; }
  .step-num   { font-family: 'IBM Plex Mono', monospace; color: #00b4d8; font-size:.75rem; letter-spacing:.1em; }
  .step-title { font-weight: 600; color: #cdd6f4; font-size:1rem; }
  .step-desc  { color: #6e8090; font-size:.8rem; margin-top:4px; }
  .metric-card { background: linear-gradient(135deg, #0d1b2a 0%, #112233 100%); border: 1px solid #1e3a5f; border-radius: 10px; padding: 18px; text-align: center; transition: border-color .2s; }
  .metric-card:hover { border-color: #00b4d8; }
  .metric-value { font-size: 1.8rem; font-weight: 700; color: #00b4d8; font-family: 'IBM Plex Mono', monospace; }
  .metric-label { font-size: .8rem; color: #6e8090; margin-top: 4px; letter-spacing:.05em; text-transform:uppercase; }
  .info-box { background: linear-gradient(135deg, #0a1929 0%, #0d1e30 100%); border: 1px solid #1f6feb; border-radius: 10px; padding: 18px; margin: 8px 0; }
  .info-title { color: #58a6ff; font-weight: 600; font-size: 1rem; margin-bottom: 6px; }
  .info-text  { color: #8b9eb5; font-size: .88rem; line-height: 1.6; }
  .warn-box { background: linear-gradient(135deg, #1a0f00 0%, #140c00 100%); border: 1px solid #d29922; border-radius: 10px; padding: 18px; margin: 8px 0; }
  .warn-title { color: #f0c040; font-weight: 600; font-size: 1rem; margin-bottom: 6px; }
  .success-box { background: linear-gradient(135deg, #071d11 0%, #051508 100%); border: 1px solid #2ea043; border-radius: 10px; padding: 18px; margin: 8px 0; }
  .success-title { color: #3fb950; font-weight: 600; font-size: 1rem; margin-bottom: 6px; }
  .error-box { background: linear-gradient(135deg, #1a0707 0%, #140505 100%); border: 1px solid #da3633; border-radius: 10px; padding: 18px; margin: 8px 0; }
  .error-title { color: #f85149; font-weight: 600; font-size: 1rem; margin-bottom: 6px; }
  .section-header { font-family: 'IBM Plex Mono', monospace; font-size: .9rem; color: #4a6a8a; letter-spacing: .15em; text-transform: uppercase; border-bottom: 1px solid #1e2a3a; padding-bottom: 8px; margin: 28px 0 16px; }
  .file-badge { display: inline-flex; align-items: center; gap: 6px; background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 6px; padding: 6px 12px; margin: 4px; font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: #cdd6f4; }
  .file-badge.done  { border-color: #2ea043; color: #3fb950; }
  .file-badge.error { border-color: #da3633; color: #f85149; }
  .batch-progress { background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 10px; padding: 20px; margin: 12px 0; }
  .batch-title { font-family: 'IBM Plex Mono', monospace; font-size: .85rem; color: #00b4d8; margin-bottom: 12px; letter-spacing: .08em; }
  .stButton > button { background: linear-gradient(135deg, #023e8a, #0077b6); color: white; border: none; border-radius: 8px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; letter-spacing:.05em; padding: 10px 24px; width: 100%; transition: opacity .2s, transform .1s; }
  .stButton > button:hover { opacity: .85; transform: translateY(-1px); }
  .summary-stat { background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 8px; padding: 14px; text-align: center; }
  .summary-stat-val { font-size: 1.6rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
  .summary-stat-lbl { font-size: .75rem; color: #6e8090; text-transform: uppercase; letter-spacing: .07em; margin-top: 4px; }
  .blob-badge { display:inline-block; background:#023e8a22; border:1px solid #0077b655; color:#48cae4; border-radius:4px; padding:2px 8px; font-family:'IBM Plex Mono',monospace; font-size:.75rem; margin:2px; }
  .chat-container { background: #0a0f15; border: 1px solid #1e2a3a; border-radius: 14px; padding: 0; overflow: hidden; margin-bottom: 16px; }
  .chat-header { background: linear-gradient(90deg, #023e8a 0%, #0077b6 100%); padding: 14px 20px; display: flex; align-items: center; gap: 10px; }
  .chat-header-title { font-family: 'IBM Plex Mono', monospace; font-size: .9rem; color: white; font-weight: 600; letter-spacing: .08em; }
  .chat-header-sub { font-size: .72rem; color: rgba(255,255,255,0.6); font-family: 'IBM Plex Mono', monospace; }
  .chat-bubble { max-width: 82%; padding: 12px 16px; border-radius: 12px; font-size: .88rem; line-height: 1.6; word-wrap: break-word; margin: 6px 0; }
  .chat-bubble.user { background: linear-gradient(135deg, #023e8a, #0077b6); color: #e0f4ff; margin-left: auto; border-bottom-right-radius: 3px; }
  .chat-bubble.assistant { background: linear-gradient(135deg, #0d1b2a, #112233); color: #cdd6f4; border: 1px solid #1e3a5f; border-bottom-left-radius: 3px; }
  .bubble-label { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .08em; margin-bottom: 6px; }
  .bubble-sources { margin-top: 10px; padding-top: 8px; border-top: 1px solid #1e3a5f; font-size: .72rem; color: #4a6a8a; font-family: 'IBM Plex Mono', monospace; }
  .source-chip { display: inline-block; background: #023e8a22; border: 1px solid #0077b655; color: #48cae4; border-radius: 4px; padding: 1px 7px; font-size: .68rem; margin: 2px; font-family: 'IBM Plex Mono', monospace; }
  .rag-status { background: #0d1117; border: 1px solid #1e3a5f; border-radius: 8px; padding: 10px 16px; font-family: 'IBM Plex Mono', monospace; font-size: .75rem; color: #4a6a8a; display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 12px; }
  .rag-status.ready   { border-color: #2ea04366; color: #3fb950; }
  .rag-status.no-data { border-color: #d2992266; color: #f0c040; }
  .chunk-card { background: #060c12; border: 1px solid #1e3a5f; border-radius: 6px; padding: 10px 14px; margin: 6px 0; }
  .chunk-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .chunk-meta { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; color: #00b4d8; }
  .chunk-text { font-family: 'IBM Plex Mono', monospace; font-size: .76rem; color: #6e8090; white-space: pre-wrap; max-height: 120px; overflow-y: auto; }
  [data-testid="stDataFrame"]   { border-radius: 8px; overflow: hidden; }
  [data-testid="stTabs"] button { font-family: 'IBM Plex Mono', monospace; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cdd6f4", family="IBM Plex Sans"),
)
COLORS = ["#00b4d8", "#48cae4", "#0096c7", "#90e0ef", "#caf0f8", "#ade8f4"]


def render_chart(df: pd.DataFrame, spec: dict):
    ctype = spec.get("type", "bar").lower()
    title = spec.get("title", "Chart")
    x_col = spec.get("x_column")
    y_col = spec.get("y_column")
    c_col = spec.get("color_column")
    cols  = df.columns.tolist()
    x_col = x_col if x_col in cols else None
    y_col = y_col if y_col in cols else None
    c_col = c_col if c_col in cols else None

    try:
        if ctype == "bar" and x_col and y_col:
            fig = px.bar(df, x=x_col, y=y_col, color=c_col, title=title, color_discrete_sequence=COLORS)
        elif ctype == "line" and x_col and y_col:
            fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, color=c_col, title=title, color_discrete_sequence=COLORS)
        elif ctype == "pie" and x_col:
            if y_col:
                fig = px.pie(df, names=x_col, values=y_col, title=title, color_discrete_sequence=COLORS)
            else:
                cnt = df[x_col].value_counts().reset_index()
                cnt.columns = [x_col, "count"]
                fig = px.pie(cnt, names=x_col, values="count", title=title, color_discrete_sequence=COLORS)
        elif ctype == "scatter" and x_col and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=c_col, title=title, color_discrete_sequence=COLORS)
        elif ctype == "histogram" and x_col:
            fig = px.histogram(df, x=x_col, title=title, nbins=20, color_discrete_sequence=["#00b4d8"])
        else:
            if x_col:
                cnt = df[x_col].value_counts().head(15).reset_index()
                cnt.columns = [x_col, "count"]
                fig = px.bar(cnt, x=x_col, y="count", title=title, color_discrete_sequence=["#00b4d8"])
            else:
                return None
        fig.update_layout(**PLOTLY_THEME, title_font_size=13, margin=dict(t=40, b=20, l=20, r=20))
        return fig
    except Exception as e:
        st.warning(f"Could not render '{title}': {e}")
        return None


def render_pipeline_sidebar(current_step: int):
    steps = [
        ("01", "Upload Invoices",        "PDF/JPG/PNG → Azure Blob Storage"),
        ("02", "Document Intelligence",  "Azure prebuilt-invoice + Arabic locale"),
        ("03", "Entity & Schema",        "LLM translates Arabic fields (no static map)"),
        ("04", "Store to SQLite",        "Normalised English columns in DB"),
        ("05", "KPI Dashboard",          "AI-generated charts & insights"),
        ("06", "Semantic Chunk → Embed", "Header chunk + 1 chunk per line item"),
        ("07", "RAG Chatbot",            "FAISS retrieval + auto SQL + grounded answers"),
    ]
    with st.sidebar:
        st.markdown("## 🧾 Invoice Intelligence\n---")
        for i, (num, title, desc) in enumerate(steps):
            css  = "done" if i < current_step else ("active" if i == current_step else "")
            icon = "✅" if i < current_step else ("⚡" if i == current_step else "○")
            st.markdown(f"""
            <div class="pipeline-step {css}">
              <div class="step-num">STEP {num} {icon}</div>
              <div class="step-title">{title}</div>
              <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"**Model:** `{AZURE_OPENAI_MODEL}`")
        st.markdown(f"**Embeddings:** `text-embedding-3-small`")
        st.markdown("**Vector DB:** FAISS · **DB:** SQLite")
        st.markdown(f"**TOP_K:** `{TOP_K}` chunks")
        st.markdown("**Chunking:** Semantic (header + per-line-item)")

        st.markdown("---")
        if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
            st.markdown(f"🟢 **LangSmith** tracing active  \n`{LANGCHAIN_PROJECT}`", help=f"Endpoint: {LANGCHAIN_ENDPOINT}")
        else:
            st.markdown("⚪ **LangSmith** tracing disabled")

        inv_df, li_df = load_invoices_from_db()
        vs_stats      = vector_store_stats()
        st.markdown("---")
        st.markdown("**📦 Database Stats**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Invoices",   len(inv_df))
        c2.metric("Line Items", len(li_df))
        c3.metric("Chunks",     vs_stats["total_chunks"])

        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.markdown("---")
            st.markdown("**💬 Chat**")
            st.caption(f"{len(st.session_state.chat_history)} messages")
            if st.button("🗑️ Clear Chat", key="clear_chat_sidebar"):
                st.session_state.chat_history = []
                st.rerun()


def main():
    init_db()

    if "pipeline_step"  not in st.session_state:
        st.session_state.pipeline_step  = 0
    if "chat_history"   not in st.session_state:
        st.session_state.chat_history   = []
    if "chat_input_key" not in st.session_state:
        st.session_state.chat_input_key = 0

    render_pipeline_sidebar(st.session_state.pipeline_step)

    st.markdown("# 🧾 Invoice Intelligence Platform")
    st.markdown("End-to-end: **Blob → Doc AI (Arabic) → LLM Schema → SQLite → Semantic Chunk → Embed → FAISS → RAG Chat**")
    st.markdown("---")

    tab_process, tab_dashboard, tab_chat, tab_db = st.tabs([
        "⚙️ Process Invoices",
        "📊 Analytics Dashboard",
        "💬 RAG Chat",
        "🗄️ Database Explorer",
    ])

    with tab_process:
        st.markdown('<div class="section-header">⬆️ Upload Invoices (Batch)</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload one or more invoice files (Arabic & English supported)",
            type=["pdf", "jpg", "jpeg", "png", "tiff"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.markdown("""
            <div style="border:2px dashed #1e3a5f;border-radius:12px;padding:50px 40px;text-align:center;
                        background:linear-gradient(135deg,#0a1520 0%,#0d1b2a 100%);">
              <h3 style="font-family:'IBM Plex Mono',monospace;color:#00b4d8">⬆️ DROP INVOICES HERE</h3>
              <p style="color:#4a6a8a">PDF · JPG · PNG · TIFF &nbsp;|&nbsp; <strong>Arabic &amp; English</strong></p>
            </div>""", unsafe_allow_html=True)
            st.session_state.pipeline_step = 0
            return

        total_size = sum(f.size for f in uploaded_files)
        st.markdown(f"""
        <div class="batch-progress">
          <div class="batch-title">📁 BATCH QUEUE — {len(uploaded_files)} file(s) · {total_size/1024:.1f} KB total</div>
          <div>{"".join(f'<span class="file-badge">📄 {f.name}</span>' for f in uploaded_files)}</div>
        </div>""", unsafe_allow_html=True)

        image_files = [f for f in uploaded_files if f.type in ["image/jpeg", "image/png", "image/jpg"]]
        if image_files:
            with st.expander(f"👁️ Preview images ({len(image_files)})"):
                cols = st.columns(min(len(image_files), 3))
                for idx, img_file in enumerate(image_files):
                    with cols[idx % 3]:
                        st.image(img_file.getvalue(), caption=img_file.name, use_container_width=True)

        st.markdown("---")

        if st.button(f"🚀 Run Pipeline on All {len(uploaded_files)} Invoice(s)", type="primary"):
            invalid_files = []
            with st.spinner("🔍 Validating files with AI…"):
                for uploaded in uploaded_files:
                    valid, reason = is_valid_invoice(uploaded.getvalue(), uploaded.name)
                    if not valid:
                        invalid_files.append(reason)

            if invalid_files:
                for reason in invalid_files:
                    st.error(f"🚫 {reason}")
                st.warning("Please remove the invalid files and try again. No files were processed.")
                st.stop()

            st.session_state.pipeline_step = 1
            progress_bar = st.progress(0, text="Starting batch…")
            all_results  = []
            succeeded    = 0

            for idx, uploaded in enumerate(uploaded_files):
                file_bytes = uploaded.getvalue()
                progress_bar.progress(int(idx / len(uploaded_files) * 100),
                                      text=f"Processing {idx+1}/{len(uploaded_files)}: {uploaded.name}")
                result = process_single_invoice(file_bytes, uploaded.name)
                all_results.append(result)
                if result["success"]:
                    succeeded += 1

            progress_bar.progress(100, text="Batch complete!")

            if "dashboard_analysis" in st.session_state:
                del st.session_state["dashboard_analysis"]

            st.session_state.pipeline_step       = 5
            st.session_state["last_batch_results"] = all_results

            col_s, col_f, col_t = st.columns(3)
            failed = len(all_results) - succeeded
            for col, val, label, color in [
                (col_s, succeeded,        "✅ Succeeded", "#3fb950"),
                (col_f, failed,           "❌ Failed",    "#f85149"),
                (col_t, len(all_results), "📄 Total",     "#00b4d8"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="summary-stat">
                      <div class="summary-stat-val" style="color:{color}">{val}</div>
                      <div class="summary-stat-lbl">{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">📋 Per-File Results</div>', unsafe_allow_html=True)
            for res in all_results:
                fname = res["filename"]
                if res["success"]:
                    llm = res.get("llm_result", {})
                    db  = res.get("db_report", {})
                    with st.expander(f"✅ {fname} — Invoice ID: {db.get('invoice_id','?')} | {db.get('line_items_inserted',0)} line items"):
                        col1, col2 = st.columns(2)
                        with col1:
                            for item in llm.get("insights", []):
                                st.markdown(f'<div class="info-box"><div class="info-title">{item["title"]}</div><div class="info-text">{item["description"]}</div></div>', unsafe_allow_html=True)
                        with col2:
                            for item in llm.get("anomalies", []):
                                st.markdown(f'<div class="warn-box"><div class="warn-title">{item["title"]}</div><div class="info-text">{item["description"]}</div></div>', unsafe_allow_html=True)
                        if res.get("extracted"):
                            with st.expander("🔍 Extracted Fields"):
                                for k, v in res["extracted"].items():
                                    if isinstance(v, dict):
                                        disp_val = v.get("value", "")
                                        conf     = v.get("confidence")
                                        conf_str = f" (conf: {conf:.0%})" if conf else ""
                                        st.markdown(f"**{k}**: {disp_val}{conf_str}")
                else:
                    with st.expander(f"❌ {fname} — FAILED", expanded=True):
                        st.markdown(f'<div class="error-box"><div class="error-title">❌ Error</div><div class="info-text">{res.get("error","Unknown error")}</div></div>', unsafe_allow_html=True)

            if succeeded > 0:
                st.info("💡 Next: go to **RAG Chat** tab and click **Index Invoices** to build the FAISS index.")

    with tab_dashboard:
        st.markdown('<div class="section-header">📊 Invoice Analytics Dashboard</div>', unsafe_allow_html=True)
        invoices_df, line_items_df = load_invoices_from_db()

        if invoices_df.empty:
            st.markdown('<div class="info-box"><div class="info-title">No invoice data yet</div><div class="info-text">Process at least one invoice first.</div></div>', unsafe_allow_html=True)
            return

        if st.button("🔄 Regenerate AI Analysis"):
            if "dashboard_analysis" in st.session_state:
                del st.session_state["dashboard_analysis"]

        if "dashboard_analysis" not in st.session_state:
            with st.spinner("🤖 AI analyzing invoice database…"):
                try:
                    st.session_state["dashboard_analysis"] = ai_analyze_invoices(invoices_df, line_items_df)
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
                    return

        analysis = st.session_state["dashboard_analysis"]

        kpis = analysis.get("kpis", [])
        if kpis:
            st.markdown('<div class="section-header">🎯 KPIs</div>', unsafe_allow_html=True)
            cols = st.columns(min(len(kpis), 4))
            for i, kpi in enumerate(kpis[:8]):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-value">{kpi.get('value','—')}</div>
                      <div class="metric-label">{kpi.get('name','')}</div>
                    </div>""", unsafe_allow_html=True)

        charts = analysis.get("charts", [])
        if charts:
            st.markdown('<div class="section-header">📈 Charts</div>', unsafe_allow_html=True)
            figs = [(s, render_chart(invoices_df, s)) for s in charts]
            figs = [(s, f) for s, f in figs if f is not None]
            for i in range(0, len(figs), 2):
                cols = st.columns(2)
                for j, (spec, fig) in enumerate(figs[i:i+2]):
                    with cols[j]:
                        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">📥 Export</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("⬇️ Invoices CSV", invoices_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "invoices.csv", "text/csv", use_container_width=True)
        with c2:
            if not line_items_df.empty:
                st.download_button("⬇️ Line Items CSV", line_items_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "line_items.csv", "text/csv", use_container_width=True)
        with c3:
            st.download_button("⬇️ AI Analysis JSON", json.dumps(analysis, indent=2, ensure_ascii=False).encode("utf-8"), "analysis.json", "application/json", use_container_width=True)

    with tab_chat:
        invoices_df, line_items_df = load_invoices_from_db()
        vs_stats     = vector_store_stats()
        chunk_count  = vs_stats["total_chunks"]
        has_invoices = not invoices_df.empty
        has_vectors  = chunk_count > 0

        st.markdown('<div class="section-header">🗂️ FAISS Index Control</div>', unsafe_allow_html=True)
        vc1, vc2, vc3 = st.columns([3, 1, 1])
        with vc1:
            if has_vectors:
                st.markdown(f'<div class="rag-status ready"><span>✅ FAISS INDEX READY</span><span>{chunk_count} semantic chunks · IndexFlatIP cosine · top-{TOP_K} · Arabic ✓</span></div>', unsafe_allow_html=True)
            elif has_invoices:
                st.markdown('<div class="rag-status no-data">⚠️ Invoices in SQLite but not yet indexed — click "Index Invoices" →</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="rag-status no-data">⚠️ No invoices processed yet. Go to ⚙️ Process Invoices first.</div>', unsafe_allow_html=True)
        with vc2:
            if has_invoices and st.button("🗂️ Index Invoices", use_container_width=True, disabled=not has_invoices):
                prog   = st.progress(0, text="Starting…")
                report = ingest_into_vector_store(
                    invoices_df, line_items_df,
                    progress_cb=lambda pct, msg: prog.progress(pct, text=msg),
                )
                prog.empty()
                st.success(f"✅ {report['chunks_upserted']} semantic chunks indexed from {report['invoices_processed']} invoices")
                st.rerun()
        with vc3:
            if has_vectors and st.button("🗑️ Clear Index", use_container_width=True):
                clear_vector_store()
                st.rerun()

        with st.expander("ℹ️ How the RAG pipeline works", expanded=False):
            st.markdown(f"""
            <div class="info-box">
              <div class="info-title">Semantic RAG Architecture — FAISS + SQL Fallback (Arabic-aware)</div>
              <div class="info-text">
                <strong>Chunking:</strong> 1 HEADER chunk + N LINE ITEM chunks per invoice<br>
                <strong>Embedding:</strong> text-embedding-3-small ({AZURE_OPENAI_EMBEDDING_MODEL}) · L2-normalised → cosine similarity<br>
                <strong>Retrieval:</strong> Top-{TOP_K} chunks · LLM decides: answer from chunks OR write SQL<br>
                <strong>SQL:</strong> auto-executed → results fed to second LLM pass for natural-language answer
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">💬 Chat</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="chat-container">
          <div class="chat-header">
            <span style="font-size:1.4rem">🤖</span>
            <div>
              <div class="chat-header-title">INVOICE RAG ASSISTANT · Arabic &amp; English</div>
              <div class="chat-header-sub">Semantic chunks · FAISS TOP_K={TOP_K} · Auto-SQL · Grounded answers</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:30px 20px;color:#2a4a6a;font-family:'IBM Plex Mono',monospace;font-size:.82rem;">
              <div style="font-size:2rem;margin-bottom:10px">💬</div>
              Ask in English or Arabic — answers grounded on semantically retrieved chunks + live SQL
            </div>""", unsafe_allow_html=True)
            suggestions = [
                "What is the total invoice value?",
                "ما هو إجمالي قيمة الفواتير؟",
                "Which vendor has the highest spend?",
                "Show me all invoices with their amounts",
                "What are the top 5 line items by amount?",
                "Are there any anomalies in the data?",
            ]
            cols = st.columns(3)
            for i, s in enumerate(suggestions):
                with cols[i % 3]:
                    if st.button(s, key=f"sug_{i}", use_container_width=True):
                        st.session_state.chat_history.append({"role": "user", "content": s})
                        st.rerun()
        else:
            for msg in st.session_state.chat_history:
                role    = msg["role"]
                content = msg["content"]
                if role == "user":
                    direction = "rtl" if is_arabic(content) else "ltr"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:flex-end;margin:6px 0">
                      <div class="chat-bubble user" style="direction:{direction}">
                        <div class="bubble-label" style="color:rgba(200,240,255,0.6);text-align:right">YOU</div>
                        {content}
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    sources_html = ""
                    if msg.get("sources"):
                        chips = "".join(f'<span class="source-chip">{s}</span>' for s in msg["sources"])
                        sources_html = f'<div class="bubble-sources">📎 Sources: {chips}</div>'
                    direction = "rtl" if is_arabic(content) else "ltr"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:flex-start;margin:6px 0">
                      <div class="chat-bubble assistant" style="direction:{direction}">
                        <div class="bubble-label" style="color:#00b4d8">⚡ INVOICE RAG AI</div>
                        {content}
                        {sources_html}
                      </div>
                    </div>""", unsafe_allow_html=True)

                    chunks = msg.get("retrieved_chunks", [])
                    if chunks:
                        with st.expander(f"🔍 Retrieved {len(chunks)} semantic chunks (TOP_K={TOP_K})", expanded=False):
                            for i, chunk in enumerate(chunks):
                                sim_pct    = int(chunk["similarity"] * 100)
                                color      = "#2ea043" if sim_pct >= 80 else "#f0c040" if sim_pct >= 60 else "#da3633"
                                chunk_type = chunk.get("chunk_type", "unknown").upper()
                                st.markdown(f"""
                                <div class="chunk-card">
                                  <div class="chunk-header">
                                    <span class="chunk-meta">CHUNK {i+1} · [{chunk_type}] · Invoice #{chunk['invoice_id']} · {chunk['original_filename']}</span>
                                    <span style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:{color};background:{color}22;padding:2px 8px;border-radius:4px;border:1px solid {color}55">{sim_pct}% match</span>
                                  </div>
                                  <div class="chunk-text">{chunk['text']}</div>
                                </div>""", unsafe_allow_html=True)

                    sql_queries = msg.get("sql_queries") or ([msg["sql_query"]] if msg.get("sql_query") else [])
                    sql_results = msg.get("sql_results") or ([msg.get("sql_result")] if msg.get("sql_result") is not None else [])
                    sql_errors  = msg.get("sql_errors")  or ([msg.get("sql_error")]  if msg.get("sql_error")  is not None else [])
                    for qi, q in enumerate(sql_queries):
                        with st.expander(f"🔍 SQL Query {qi+1}", expanded=False):
                            st.code(q, language="sql")
                        df  = sql_results[qi] if qi < len(sql_results) else None
                        err = sql_errors[qi]  if qi < len(sql_errors)  else None
                        if df is not None and not df.empty:
                            st.dataframe(df, use_container_width=True, height=min(200, 50 + len(df)*35))
                        elif err:
                            st.markdown(f'<div class="error-box"><div class="error-title">SQL Error</div><div class="info-text">{err}</div></div>', unsafe_allow_html=True)
                        elif df is not None and df.empty:
                            st.caption("*(Query returned no rows)*")

        pending = None
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            pending = st.session_state.chat_history[-1]["content"]

        if pending and has_vectors:
            with st.spinner("🔍 Searching FAISS… 🤖 Generating…"):
                try:
                    rag_result = rag_chat(
                        user_message=pending,
                        chat_history=st.session_state.chat_history[:-1],
                        invoices_df=invoices_df,
                        line_items_df=line_items_df,
                    )
                    st.session_state.chat_history.append({
                        "role":             "assistant",
                        "content":          rag_result["answer"],
                        "sources":          rag_result["sources"],
                        "retrieved_chunks": rag_result.get("retrieved_chunks", []),
                        "sql_queries":      rag_result.get("sql_queries", []),
                        "sql_results":      rag_result.get("sql_results", []),
                        "sql_errors":       rag_result.get("sql_errors", []),
                        "sql_query":        rag_result.get("sql_query"),
                        "sql_result":       rag_result.get("sql_result"),
                        "sql_error":        rag_result.get("sql_error"),
                    })
                    st.rerun()
                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Error: {e}", "sources": []})
                    st.rerun()
        elif pending and not has_vectors:
            st.warning("⚠️ FAISS index is empty. Click **Index Invoices** above first.")

        st.markdown("---")
        col_input, col_clear = st.columns([5, 1])
        with col_input:
            user_input = st.chat_input(
                "Ask about your invoices (English or Arabic)…",
                key=f"chat_input_{st.session_state.chat_input_key}",
                disabled=not has_vectors,
            )
        with col_clear:
            if st.button("🗑️ Clear", key="clear_chat_main", use_container_width=True):
                st.session_state.chat_history   = []
                st.session_state.chat_input_key += 1
                st.rerun()
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.rerun()

    with tab_db:
        st.markdown('<div class="section-header">🗄️ Database Explorer</div>', unsafe_allow_html=True)
        invoices_df, line_items_df = load_invoices_from_db()
        sub1, sub2 = st.tabs(["📋 Invoices", "📦 Line Items"])
        with sub1:
            if invoices_df.empty:
                st.info("No invoices stored yet.")
            else:
                search  = st.text_input("🔎 Filter", key="inv_search")
                show_df = invoices_df
                if search:
                    mask    = show_df.astype(str).apply(lambda c: c.str.contains(search, case=False, na=False)).any(axis=1)
                    show_df = show_df[mask]
                st.dataframe(show_df, use_container_width=True, height=350)
                st.caption(f"{len(show_df)} row(s) shown")
        with sub2:
            if line_items_df.empty:
                st.info("No line items stored yet.")
            else:
                st.dataframe(line_items_df, use_container_width=True, height=350)
                st.caption(f"{len(line_items_df)} row(s)")

        st.markdown('<div class="section-header">💻 SQL Query</div>', unsafe_allow_html=True)
        sql_query = st.text_area("Run a custom SQL query (SELECT only)", value="SELECT * FROM invoices LIMIT 10", height=80)
        if st.button("▶️ Execute"):
            df, err = execute_sql_safe(sql_query)
            if err:
                st.error(f"SQL Error: {err}")
            else:
                st.dataframe(df, use_container_width=True)
                st.caption(f"{len(df)} rows returned")

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;color:#2a4a6a;font-size:.75rem;padding:8px;font-family:'IBM Plex Mono',monospace">
      🧾 INVOICE INTELLIGENCE · SEMANTIC CHUNKS · text-embedding-3-small · FAISS · TOP_K={TOP_K} · LLM Arabic Translation
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
