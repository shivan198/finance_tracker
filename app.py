"""
Finance Tracker — Flask backend.

Single responsibility: accept a bank statement PDF, parse it with parse.py,
compute insights/chart data with analyze_statement.py, and return everything
as JSON for the frontend to render.

Routes:
    GET  /              -> the single-page app shell
    POST /api/analyze   -> multipart file upload, returns JSON results
    POST /api/chat      -> ask a question about the most recently analyzed statement

Chat architecture: this app tries a RAG-based chat first (rag_chat.py —
embeddings + Chroma retrieval + a LangChain LCEL chain with structured
output). If that stack isn't installed or Ollama's embedding model isn't
pulled, it falls back to the simpler prompt-based chat in
analyze_statement.py. Either way the frontend sees the same response shape.

Note on state: this app assumes a single local user (you), so the last
analyzed statement (and its RAG chain) is kept in a module-level variable
rather than a database or per-session store. That's a deliberate
simplification for a local personal tool — if this were ever deployed for
multiple concurrent users, that state would need to move to a session or
database instead.
"""

import os
import tempfile
import traceback

import pandas as pd
from flask import Flask, jsonify, render_template, request

from parse import parse_bank_statement
from analyze_statement import (
    answer_statement_question,
    build_category_chart_data,
    build_chart_data,
    build_summary_for_llm,
    generate_llm_insights,
)

app = Flask(__name__)

MAX_UPLOAD_MB = 20
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Which local model to use. Override with:
#   OLLAMA_MODEL=qwen2.5:14b python app.py
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Set to "0" to skip the RAG stack entirely and always use the simpler
# prompt-based chat — useful if you haven't set up langchain/chromadb yet.
USE_RAG = os.environ.get("USE_RAG", "1") != "0"

# Holds the most recently analyzed statement (and its RAG chain, if built)
# so /api/chat can answer follow-up questions. See module docstring.
_last_analysis = {"df": None, "currency": None, "rag": None}


def build_dataframe(pdf_path: str) -> dict:
    """
    Runs the full parse -> typed DataFrame -> validation pipeline for one PDF.
    Returns a dict with the dataframe and metadata, or raises ValueError if
    nothing could be parsed (caller turns that into a clean error response).
    """
    result = parse_bank_statement(pdf_path)
    df = pd.DataFrame(result["transactions"])

    if df.empty:
        raise ValueError(
            "No transactions could be extracted from this PDF. "
            "It may be a scanned/image-based statement, or an unsupported layout."
        )

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for col in ["withdrawal", "deposit", "balance"]:
        df[col] = pd.to_numeric(df[col].replace("", None), errors="coerce")

    df["validation_flag"] = False
    if result["flagged_indices"]:
        df.loc[result["flagged_indices"], "validation_flag"] = True

    return {
        "df": df,
        "used_dynamic_thresholds": result["used_dynamic_thresholds"],
        "flagged_indices": result["flagged_indices"],
        "currency": result["currency"],
    }


def dataframe_to_transactions(df: pd.DataFrame) -> list:
    """Row-wise JSON-safe records for the frontend transaction table."""
    records = []
    for _, row in df.iterrows():
        records.append({
            "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
            "description": "" if pd.isna(row["description"]) else str(row["description"]),
            "deposit": None if pd.isna(row["deposit"]) else round(float(row["deposit"]), 2),
            "withdrawal": None if pd.isna(row["withdrawal"]) else round(float(row["withdrawal"]), 2),
            "balance": None if pd.isna(row["balance"]) else round(float(row["balance"]), 2),
            "flagged": bool(row["validation_flag"]),
            "category": row["category"] if "category" in df.columns and pd.notna(row.get("category")) else None,
        })
    return records


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    uploaded = request.files.get("file")
    if not uploaded or uploaded.filename == "":
        return jsonify({"ok": False, "error": "No file was uploaded."}), 400

    if not uploaded.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "Please upload a PDF bank statement."}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            uploaded.save(tmp.name)
            tmp_path = tmp.name

        parsed = build_dataframe(tmp_path)
        df = parsed["df"]
        currency = parsed["currency"]

        summary = build_summary_for_llm(df)  # already JSON-safe (rounded floats, str dates)
        chart_data = build_chart_data(df)
        transactions = dataframe_to_transactions(df)

        insights = None
        insights_error = None
        try:
            insights = generate_llm_insights(df, model=OLLAMA_MODEL, currency=currency)
        except Exception as exc:
            insights_error = (
                f"Local LLM insights unavailable ({exc}). "
                f"Make sure Ollama is running and '{OLLAMA_MODEL}' is pulled."
            )

        # LLM-based categorization (optional — chart card hides itself if this fails)
        category_error = None
        try:
            from rag_chat import categorize_transactions
            df["category"] = categorize_transactions(df, model=OLLAMA_MODEL)
            chart_data["by_category"] = build_category_chart_data(df)
            transactions = dataframe_to_transactions(df)  # rebuild now that 'category' exists
        except Exception as exc:
            category_error = str(exc)
            chart_data["by_category"] = []

        # Build the RAG chain once per statement (reused across chat turns).
        # Falls back silently to the simpler chat path if unavailable.
        rag_chain = None
        rag_error = None
        if USE_RAG:
            try:
                from rag_chat import build_rag_chain
                rag_chain = build_rag_chain(df, currency, model=OLLAMA_MODEL)
            except Exception as exc:
                rag_error = str(exc)

        _last_analysis["df"] = df
        _last_analysis["currency"] = currency
        _last_analysis["rag"] = rag_chain

        return jsonify({
            "ok": True,
            "filename": uploaded.filename,
            "summary": summary,
            "chart_data": chart_data,
            "transactions": transactions,
            "flagged_count": len(parsed["flagged_indices"]),
            "used_dynamic_thresholds": parsed["used_dynamic_thresholds"],
            "currency": currency,
            "insights": insights,
            "insights_error": insights_error,
            "category_error": category_error,
            "rag_enabled": rag_chain is not None,
            "rag_error": rag_error if not rag_chain else None,
        })

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 422
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Unexpected error while processing the file: {exc}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()

    df = _last_analysis.get("df")
    if df is None:
        return jsonify({"ok": False, "error": "Upload and analyze a statement first."}), 400

    if not question:
        return jsonify({"ok": False, "error": "Ask something first."}), 400

    rag_chain = _last_analysis.get("rag")
    if rag_chain is not None:
        try:
            answer, sources = rag_chain["ask"](question)
            return jsonify({"ok": True, "answer": answer, "sources": sources, "mode": "rag"})
        except Exception as exc:
            traceback.print_exc()
            # Fall through to the simpler path rather than failing outright.

    try:
        answer = answer_statement_question(
            df, question,
            currency=_last_analysis.get("currency"),
            model=OLLAMA_MODEL,
        )
        return jsonify({"ok": True, "answer": answer, "sources": [], "mode": "simple"})
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": f"Local LLM unavailable ({exc}). Make sure Ollama is running and '{OLLAMA_MODEL}' is pulled.",
        }), 503


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
