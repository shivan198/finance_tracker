"""
eval_rag.py — a small evaluation harness for the RAG retriever.

Most beginner LLM projects skip evaluation entirely. This script measures
something concrete: for a labeled set of (question, expected_row_indices)
pairs, does the retriever actually surface the transactions a human would
consider relevant?

Metrics computed, per question and averaged:
  - Recall@k:    fraction of expected relevant rows that appear in the top-k
                 retrieved results (did we find what we needed?)
  - Precision@k: fraction of the top-k retrieved rows that were actually
                 expected/relevant (how much noise came along with it?)

How to build your own labeled set:
  1. Parse a statement you know well (yours, or one of the synthetic samples).
  2. Pick 5-10 realistic questions someone might ask about it.
  3. For each question, manually note which row indices (from the DataFrame)
     are genuinely relevant to answering it.
  4. Fill in LABELED_EXAMPLES below with your own statement + questions.

This part of the evaluation logic (scoring) is pure Python and doesn't need
Ollama/Chroma to run — see test_scoring() at the bottom, which is exercised
without any LLM calls at all. Running eval_rag() end-to-end does need the
RAG stack (rag_chat.py) working, since it actually calls the retriever.
"""

import sys

import pandas as pd

from parse import parse_bank_statement


def recall_at_k(retrieved_indices: list, expected_indices: set) -> float:
    if not expected_indices:
        return 1.0  # nothing was expected to be found
    found = set(retrieved_indices) & expected_indices
    return len(found) / len(expected_indices)


def precision_at_k(retrieved_indices: list, expected_indices: set) -> float:
    if not retrieved_indices:
        return 0.0
    found = set(retrieved_indices) & expected_indices
    return len(found) / len(retrieved_indices)


def evaluate_retriever(retriever, labeled_examples: list) -> None:
    """
    labeled_examples: list of {"question": str, "expected_row_indices": set[int]}
    Prints per-question and averaged recall/precision.
    """
    recalls, precisions = [], []

    print(f"{'Question':<45} {'Recall@k':>10} {'Precision@k':>13}")
    print("-" * 70)

    for example in labeled_examples:
        docs = retriever.invoke(example["question"])
        retrieved_indices = [d.metadata["row_index"] for d in docs]
        expected = set(example["expected_row_indices"])

        r = recall_at_k(retrieved_indices, expected)
        p = precision_at_k(retrieved_indices, expected)
        recalls.append(r)
        precisions.append(p)

        print(f"{example['question'][:44]:<45} {r:>10.2f} {p:>13.2f}")

    print("-" * 70)
    print(f"{'AVERAGE':<45} {sum(recalls)/len(recalls):>10.2f} {sum(precisions)/len(precisions):>13.2f}")


def test_scoring():
    """Pure-Python sanity check of the scoring math — no LLM/retriever needed."""
    retrieved = [1, 2, 3, 4, 5]
    expected = {2, 3, 9}

    r = recall_at_k(retrieved, expected)
    p = precision_at_k(retrieved, expected)

    # 2 of 3 expected rows were retrieved -> recall = 2/3
    # 2 of 5 retrieved rows were relevant -> precision = 2/5
    assert abs(r - (2 / 3)) < 1e-9, f"recall mismatch: {r}"
    assert abs(p - (2 / 5)) < 1e-9, f"precision mismatch: {p}"

    # Edge cases
    assert recall_at_k([], {1, 2}) == 0.0
    assert recall_at_k([1, 2], set()) == 1.0
    assert precision_at_k([], {1, 2}) == 0.0

    print("test_scoring: all assertions passed.")


if __name__ == "__main__":
    if "--test-scoring" in sys.argv:
        test_scoring()
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python eval_rag.py statement.pdf")
        print("       python eval_rag.py --test-scoring   (runs the scoring self-test only)")
        sys.exit(1)

    pdf_path = sys.argv[1]

    result = parse_bank_statement(pdf_path)
    df = pd.DataFrame(result["transactions"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for col in ["withdrawal", "deposit", "balance"]:
        df[col] = pd.to_numeric(df[col].replace("", None), errors="coerce")

    print(f"Loaded {len(df)} transactions. Row indices for reference:\n")
    print(df[["description"]].to_string())
    print()
    print("Fill in LABELED_EXAMPLES with real questions + expected row indices")
    print("for this statement, then edit this script to call evaluate_retriever().")

    # Example shape — replace with real questions/indices for your statement:
    LABELED_EXAMPLES = [
        # {"question": "how much did I spend on rent?", "expected_row_indices": {2}},
        # {"question": "what deposits came in this month?", "expected_row_indices": {0, 6}},
    ]

    if LABELED_EXAMPLES:
        from rag_chat import build_vectorstore
        vectorstore = build_vectorstore(df)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        evaluate_retriever(retriever, LABELED_EXAMPLES)
    else:
        print("\n(No LABELED_EXAMPLES defined yet — nothing to evaluate.)")
