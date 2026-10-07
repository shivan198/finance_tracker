"""
rag_chat.py — RAG (retrieval-augmented generation) + LangChain layer.

Why this exists (vs. the earlier plain-prompt approach in analyze_statement.py):

  The earlier `answer_statement_question()` sent the first N transactions,
  chronologically, to the LLM on every question. That works for small
  statements but doesn't scale, and it isn't really "RAG" — it's just
  truncation. This module instead:

    1. Embeds every transaction into a local vector store (Chroma).
    2. For each question, retrieves only the semantically relevant
       transactions (e.g. "food" retrieves grocery/restaurant transactions
       even if the word "food" never appears in any narration).
    3. Composes retriever -> prompt -> model -> structured output as an
       LCEL (LangChain Expression Language) chain, the standard modern
       LangChain pattern, rather than hand-rolling string concatenation.
    4. Uses Pydantic-validated structured output instead of trusting free
       text, and — importantly — does NOT ask the model to self-report
       which transactions it used. We already know, because we retrieved
       them; asking the model to enumerate sources risks hallucinated
       citations. The "based on" transaction list returned to the caller
       comes directly from the retriever's metadata.

Everything here is optional infrastructure: if langchain/chromadb aren't
installed, or Ollama's embedding model isn't pulled, `build_rag_chain()`
raises, and the caller (app.py) falls back to the simpler
`analyze_statement.answer_statement_question()` path. See README.md.
"""

import json
import uuid
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from analyze_statement import build_summary_for_llm, REFUSAL_TEXT

# Local embedding model pulled via Ollama: `ollama pull nomic-embed-text`.
# Keeps the whole stack on one local model server rather than adding a
# second dependency (e.g. sentence-transformers) just for embeddings.
EMBEDDING_MODEL = "nomic-embed-text"

CATEGORY_LIST = [
    "Income / Salary",
    "Transfers (UPI/NEFT/IMPS)",
    "Food & Dining",
    "Shopping",
    "Bills & Utilities",
    "Rent & Housing",
    "Entertainment",
    "Transport",
    "Fees & Charges",
    "Investments / Savings",
    "Other",
]


class ChatAnswer(BaseModel):
    """Structured output for the chat endpoint — validated, not free text."""
    answer: str = Field(description="A concise, direct answer to the question, using only the given context.")
    is_related_to_statement: bool = Field(
        description="True only if the question is genuinely answerable from this statement's data."
    )


class CategorizedItem(BaseModel):
    index: int = Field(description="The transaction's index number, exactly as given in the prompt.")
    category: str = Field(description=f"One of: {', '.join(CATEGORY_LIST)}")


class CategorizationBatch(BaseModel):
    items: list[CategorizedItem]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def _transaction_to_text(row: pd.Series) -> str:
    direction = "Withdrawal" if pd.notna(row.get("withdrawal")) and row.get("withdrawal") else "Deposit"
    amount = row.get("withdrawal") if direction == "Withdrawal" else row.get("deposit")
    date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row.get("date")) else "unknown date"
    return f"{date_str} | {direction} of {amount} | {row.get('description', '')}"


def build_vectorstore(df: pd.DataFrame):
    """
    Embeds each transaction and returns an in-memory Chroma vector store.
    A fresh, uniquely-named ephemeral collection is created per call — this
    app assumes one statement at a time (see app.py's single-user note), so
    there's no need to persist or manage collections across sessions.
    """
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma

    documents = []
    for idx, row in df.iterrows():
        documents.append(Document(
            page_content=_transaction_to_text(row),
            metadata={
                "row_index": int(idx),
                "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row.get("date")) else None,
                "description": str(row.get("description", "")),
            },
        ))

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    collection_name = f"statement_{uuid.uuid4().hex[:8]}"
    return Chroma.from_documents(documents, embedding=embeddings, collection_name=collection_name)


# ---------------------------------------------------------------------------
# RAG chat chain
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """You are a financial assistant restricted to ONE task: answering
questions about a specific bank statement using only the context given below.
This data belongs to the person asking.

Rules (follow strictly):
- Only use the retrieved transactions and summary below. Never invent transactions, dates, or amounts.
- All amounts are in {currency_code} ({currency_symbol}). Always use "{currency_symbol}", never "$" unless the code is USD.
- Set is_related_to_statement to false if the question is general knowledge, unrelated to
  personal finance, or not answerable from the data below — in that case, leave "answer" empty.
- Otherwise set is_related_to_statement to true and answer concisely using only the data given.

Statement summary:
{summary}

Retrieved transactions (semantically relevant to the question):
{context}
"""


def _format_docs(docs) -> str:
    if not docs:
        return "(no matching transactions retrieved)"
    return "\n".join(f"- {d.page_content}" for d in docs)


def build_rag_chain(df: pd.DataFrame, currency: dict, model: str, k: int = 12):
    """
    Builds and returns a dict with the retriever and a callable `ask(question)`
    that runs the full LCEL chain and returns (answer_text, source_dates).

    Raises ImportError / RuntimeError if langchain/chromadb/the embedding
    model aren't available — callers should catch this and fall back.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_ollama import ChatOllama

    vectorstore = build_vectorstore(df)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    summary_json = json.dumps(build_summary_for_llm(df))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ]).partial(
        summary=summary_json,
        currency_code=currency.get("code", "USD"),
        currency_symbol=currency.get("symbol", "$"),
    )

    llm = ChatOllama(model=model, temperature=0)
    structured_llm = llm.with_structured_output(ChatAnswer)

    # LCEL composition: retriever feeds "context", the raw question passes
    # through unchanged, then both flow into the prompt -> model -> parser.
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    def ask(question: str):
        retrieved_docs = retriever.invoke(question)
        source_dates = sorted({d.metadata.get("date") for d in retrieved_docs if d.metadata.get("date")})

        result: ChatAnswer = chain.invoke(question)

        if not result.is_related_to_statement:
            return REFUSAL_TEXT, []

        return result.answer, source_dates

    return {"retriever": retriever, "ask": ask}


# ---------------------------------------------------------------------------
# Batch categorization (structured output, chunked for larger statements)
# ---------------------------------------------------------------------------

def categorize_transactions(df: pd.DataFrame, model: str, batch_size: int = 40) -> pd.Series:
    """
    Returns a Series aligned with df's index, one category label per row,
    assigned via LangChain structured output. Processes in chunks so large
    statements don't blow past a reasonable prompt size in one call.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=model, temperature=0)
    structured_llm = llm.with_structured_output(CategorizationBatch)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify each transaction below into exactly one category from this list:\n"
         f"{', '.join(CATEGORY_LIST)}\n\n"
         "Return one item per transaction index given. Use the description and amount "
         "direction (withdrawal vs deposit) to judge the category — e.g. UPI/NEFT/IMPS "
         "transfers without a clear merchant should usually be 'Transfers (UPI/NEFT/IMPS)' "
         "unless the description clearly indicates rent, salary, or another specific category."),
        ("human", "{transactions_block}"),
    ])
    chain = prompt | structured_llm

    categories = pd.Series(index=df.index, dtype=object)
    categories[:] = "Other"  # safe default if a batch fails

    indices = list(df.index)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        lines = []
        for idx in batch_idx:
            row = df.loc[idx]
            lines.append(f"{idx}: {_transaction_to_text(row)}")
        transactions_block = "\n".join(lines)

        try:
            result: CategorizationBatch = chain.invoke({"transactions_block": transactions_block})
            for item in result.items:
                if item.index in categories.index:
                    label = item.category if item.category in CATEGORY_LIST else "Other"
                    categories.loc[item.index] = label
        except Exception:
            # Leave this batch's rows as the "Other" default rather than
            # failing the whole request over one bad batch.
            continue

    return categories
