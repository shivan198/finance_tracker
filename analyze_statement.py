"""
Basic analytics for a parsed bank statement DataFrame.

Expects a DataFrame with these columns (matches parse_bank_statement() output):
    date        - datetime64
    description - str
    deposit     - float (NaN if none)
    withdrawal  - float (NaN if none)
    balance     - float
    validation_flag - bool (optional, used to warn about unreliable rows)

Usage:
    from bank_statement_extract_pdfplumber import parse_bank_statement
    import pandas as pd

    result = parse_bank_statement("statement.pdf")
    df = pd.DataFrame(result['transactions'])
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    for col in ['withdrawal', 'deposit', 'balance']:
        df[col] = pd.to_numeric(df[col].replace('', None), errors='coerce')

    from analyze_statement import print_insights, generate_charts
    print_insights(df)
    generate_charts(df, output_dir="charts")
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe default for headless/script use; remove if running in Jupyter and want inline plots
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Text insights
# ---------------------------------------------------------------------------

def print_insights(df: pd.DataFrame):
    """Print a plain-text summary of key figures from the statement."""
    df = df.copy()
    df['deposit'] = df['deposit'].fillna(0)
    df['withdrawal'] = df['withdrawal'].fillna(0)

    total_income = df['deposit'].sum()
    total_expense = df['withdrawal'].sum()
    net_change = total_income - total_expense
    num_transactions = len(df)
    num_income_tx = (df['deposit'] > 0).sum()
    num_expense_tx = (df['withdrawal'] > 0).sum()

    avg_expense = df.loc[df['withdrawal'] > 0, 'withdrawal'].mean()
    avg_income = df.loc[df['deposit'] > 0, 'deposit'].mean()

    largest_expense_idx = df['withdrawal'].idxmax() if num_expense_tx > 0 else None
    largest_income_idx = df['deposit'].idxmax() if num_income_tx > 0 else None

    date_min = df['date'].min()
    date_max = df['date'].max()

    print("=" * 60)
    print("STATEMENT SUMMARY")
    print("=" * 60)
    if pd.notna(date_min) and pd.notna(date_max):
        print(f"Period: {date_min.date()} to {date_max.date()}")
    print(f"Total transactions: {num_transactions}")
    print()
    print(f"Total income (deposits):   {total_income:>14,.2f}  ({num_income_tx} transactions)")
    print(f"Total expenses (withdrawals): {total_expense:>11,.2f}  ({num_expense_tx} transactions)")
    print(f"Net change:                 {net_change:>13,.2f}")
    print()
    if num_expense_tx > 0:
        print(f"Average expense: {avg_expense:,.2f}")
    if num_income_tx > 0:
        print(f"Average income:  {avg_income:,.2f}")
    print()

    if largest_expense_idx is not None:
        row = df.loc[largest_expense_idx]
        print(f"Largest expense: {row['withdrawal']:,.2f} on {row['date'].date() if pd.notna(row['date']) else '?'}"
              f" — {str(row['description'])[:60]}")
    if largest_income_idx is not None:
        row = df.loc[largest_income_idx]
        print(f"Largest income:  {row['deposit']:,.2f} on {row['date'].date() if pd.notna(row['date']) else '?'}"
              f" — {str(row['description'])[:60]}")

    if 'validation_flag' in df.columns and df['validation_flag'].any():
        n_flagged = df['validation_flag'].sum()
        print()
        print(f"⚠ {n_flagged} transaction(s) failed balance validation — treat those figures with caution.")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def generate_charts(df: pd.DataFrame, output_dir: str = "charts"):
    """
    Generates and saves 4 basic charts as PNG files:
      1. Monthly income vs expense (bar)
      2. Balance over time (line)
      3. Top 10 largest expenses (horizontal bar)
      4. Daily net cash flow (bar, colored by +/-)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = df.copy()
    df['deposit'] = df['deposit'].fillna(0)
    df['withdrawal'] = df['withdrawal'].fillna(0)
    df = df.dropna(subset=['date']).sort_values('date')

    # --- Chart 1: Monthly income vs expense ---
    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly = df.groupby('month')[['deposit', 'withdrawal']].sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    monthly.plot(kind='bar', ax=ax, color=['#2ca02c', '#d62728'])
    ax.set_title('Monthly Income vs Expenses')
    ax.set_xlabel('Month')
    ax.set_ylabel('Amount')
    ax.legend(['Income', 'Expenses'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '1_monthly_income_vs_expense.png'), dpi=120)
    plt.close(fig)

    # --- Chart 2: Balance over time ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['date'], df['balance'], marker='o', markersize=3, color='#1f77b4')
    ax.set_title('Account Balance Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Balance')
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '2_balance_over_time.png'), dpi=120)
    plt.close(fig)

    # --- Chart 3: Top 10 largest expenses ---
    top_expenses = df[df['withdrawal'] > 0].nlargest(10, 'withdrawal')[['date', 'description', 'withdrawal']]
    labels = [f"{str(d)[:20]}" for d in top_expenses['description']]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels[::-1], top_expenses['withdrawal'][::-1], color='#d62728')
    ax.set_title('Top 10 Largest Expenses')
    ax.set_xlabel('Amount')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '3_top_expenses.png'), dpi=120)
    plt.close(fig)

    # --- Chart 4: Daily net cash flow ---
    df['net'] = df['deposit'] - df['withdrawal']
    daily_net = df.groupby(df['date'].dt.date)['net'].sum()
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in daily_net]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(daily_net.index.astype(str), daily_net.values, color=colors)
    ax.set_title('Daily Net Cash Flow')
    ax.set_xlabel('Date')
    ax.set_ylabel('Net Amount (Deposits - Withdrawals)')
    ax.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '4_daily_net_cashflow.png'), dpi=120)
    plt.close(fig)

    print(f"4 charts saved to '{output_dir}/' folder.")


# ---------------------------------------------------------------------------
# LLM-generated insights
# ---------------------------------------------------------------------------

def build_summary_for_llm(df: pd.DataFrame) -> dict:
    """
    Computes all numbers in Python (never trust an LLM to do arithmetic on
    financial data) and packages them into a compact dict. This is what gets
    sent to the LLM — never the raw transaction list — so the prompt stays
    small, fast, and the model's job is purely to write natural-language
    commentary about numbers we've already verified, not calculate them.
    """
    df = df.copy()
    df['deposit'] = df['deposit'].fillna(0)
    df['withdrawal'] = df['withdrawal'].fillna(0)
    df = df.dropna(subset=['date']).sort_values('date')

    total_income = round(df['deposit'].sum(), 2)
    total_expense = round(df['withdrawal'].sum(), 2)
    net_change = round(total_income - total_expense, 2)

    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly = df.groupby('month')[['deposit', 'withdrawal']].sum().round(2)
    monthly_summary = [
        {"month": m, "income": row['deposit'], "expense": row['withdrawal']}
        for m, row in monthly.iterrows()
    ]

    top_expenses = df[df['withdrawal'] > 0].nlargest(5, 'withdrawal')
    top_expenses_summary = [
        {"date": str(r['date'].date()), "amount": round(r['withdrawal'], 2), "description": str(r['description'])[:60]}
        for _, r in top_expenses.iterrows()
    ]

    top_income = df[df['deposit'] > 0].nlargest(5, 'deposit')
    top_income_summary = [
        {"date": str(r['date'].date()), "amount": round(r['deposit'], 2), "description": str(r['description'])[:60]}
        for _, r in top_income.iterrows()
    ]

    return {
        "period_start": str(df['date'].min().date()) if not df.empty else None,
        "period_end": str(df['date'].max().date()) if not df.empty else None,
        "total_transactions": len(df),
        "total_income": total_income,
        "total_expense": total_expense,
        "net_change": net_change,
        "monthly_breakdown": monthly_summary,
        "top_5_expenses": top_expenses_summary,
        "top_5_income": top_income_summary,
    }


def generate_llm_insights(df: pd.DataFrame, model: str = "qwen2.5:7b", currency: dict = None) -> str:
    """
    Sends the pre-computed summary (not raw transactions) to a local Ollama
    model and returns natural-language commentary: spending patterns,
    notable observations, and simple suggestions.

    `currency` should be the dict returned by parse.detect_currency(), e.g.
    {'code': 'INR', 'symbol': '₹', 'confidence': 'high'}. Without it, models
    tend to default to '$' regardless of what currency the statement is
    actually in, which is misleading for non-USD statements.

    Requires: pip install ollama, and Ollama running locally with the model pulled.
    """
    import ollama

    summary = build_summary_for_llm(df)
    currency = currency or {'code': 'USD', 'symbol': '$', 'confidence': 'low'}

    prompt = f"""You are a financial assistant. Based on the following pre-computed
bank statement summary (all numbers are already verified — do not recalculate
or second-guess them), write a short analysis for the account holder.

IMPORTANT: All amounts in this statement are in {currency['code']} (symbol: {currency['symbol']}).
Use "{currency['symbol']}" for every amount you mention — do not use "$" unless
the currency code above is actually USD.

Cover:
1. Overall cash flow (income vs expenses, net change)
2. Any notable spending pattern across months
3. The largest expense and what it might be (based on description)
4. One or two practical observations or suggestions

Keep it concise (150-200 words), in plain conversational language, no headers or bullet spam.

Summary data:
{summary}
"""

    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response['message']['content']


# ---------------------------------------------------------------------------
# Statement-scoped chat
# ---------------------------------------------------------------------------

# Words that signal a question is plausibly about personal finance / this
# statement. Not exhaustive — this is a cheap first-pass filter, not the only
# line of defense (the LLM prompt below is the second, stricter layer).
_FINANCE_KEYWORDS = [
    'spend', 'spent', 'spending', 'expense', 'expenses', 'income', 'earn',
    'earned', 'balance', 'transaction', 'transactions', 'deposit', 'deposits',
    'withdraw', 'withdrawal', 'withdrawals', 'budget', 'save', 'saving',
    'savings', 'month', 'date', 'merchant', 'amount', 'statement', 'pay',
    'paid', 'payment', 'bill', 'bills', 'subscription', 'subscriptions',
    'largest', 'biggest', 'smallest', 'total', 'category', 'categories',
    'credit', 'debit', 'upi', 'neft', 'atm', 'salary', 'rent', 'refund',
    'charge', 'charges', 'fee', 'fees', 'net', 'cash flow', 'flow',
]


def _looks_statement_related(question: str, df: pd.DataFrame) -> bool:
    """
    Cheap pre-filter: does the question contain finance-ish vocabulary, OR
    overlap with words actually appearing in this statement's narrations
    (e.g. a merchant name)? If neither, it's almost certainly off-topic
    (e.g. "tell me a joke", "what's the capital of France") and we can
    reject it without spending an LLM call at all.
    """
    q_lower = question.lower()

    if any(kw in q_lower for kw in _FINANCE_KEYWORDS):
        return True

    # Check for overlap with words that actually appear in this statement's
    # descriptions (catches things like "how much did I pay Freshmart?")
    q_words = set(w.strip('.,?!') for w in q_lower.split() if len(w) > 3)
    if q_words and 'description' in df.columns:
        desc_text = ' '.join(df['description'].dropna().astype(str)).lower()
        if any(w in desc_text for w in q_words):
            return True

    return False


# Shared across both the legacy prompt-based chat below and the RAG-based
# chat in rag_chat.py, so the frontend can reliably detect a refusal by
# exact string match regardless of which path answered the question.
REFUSAL_TEXT = "I can only answer questions about this uploaded statement."


def answer_statement_question(df: pd.DataFrame, question: str,
                                currency: dict = None,
                                model: str = "qwen2.5:7b",
                                max_transactions: int = 200) -> str:
    """
    Answers a question using ONLY this statement's data. Two layers of
    guarding against off-topic questions:
      1. A cheap keyword/overlap pre-filter (no LLM call if it obviously
         doesn't relate to personal finance or this statement's content).
      2. A strict system-style instruction telling the model to refuse
         anything not answerable from the given data, in case something
         passes layer 1 but is still off-topic (e.g. "what's your favorite
         bank?" contains 'bank'-adjacent vocabulary but isn't a real question
         about this statement).

    NOTE: this is the simpler, non-RAG fallback path (always sends the first
    `max_transactions` rows rather than retrieving relevant ones). See
    rag_chat.py for the retrieval-augmented version app.py prefers when
    available.

    Requires: pip install ollama, and Ollama running with the model pulled.
    """
    REFUSAL = REFUSAL_TEXT

    if not question or not question.strip():
        return "Ask me something about this statement — spending, balances, specific transactions, and so on."

    if not _looks_statement_related(question, df):
        return REFUSAL

    import ollama

    currency = currency or {'code': 'USD', 'symbol': '$', 'confidence': 'low'}
    summary = build_summary_for_llm(df)

    tx_records = dataframe_records_for_chat(df, max_rows=max_transactions)

    prompt = f"""You are a financial assistant restricted to ONE task: answering
questions about the specific bank statement data given below. This data
belongs to the person asking — it is their own statement.

Rules (follow strictly):
- Only use the data given below. Never invent transactions, dates, or amounts.
- All amounts are in {currency['code']} ({currency['symbol']}). Always use "{currency['symbol']}", never "$" unless the code above is USD.
- If the question is NOT answerable from this data — general knowledge, unrelated topics,
  requests to do something else, or anything not about this statement — respond with
  EXACTLY this sentence and nothing else: "{REFUSAL}"
- Otherwise, answer concisely and directly using only the data given.

Statement summary:
{summary}

Transactions (date, description, deposit, withdrawal, balance):
{tx_records}

Question: {question}
"""

    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response['message']['content'].strip()


def dataframe_records_for_chat(df: pd.DataFrame, max_rows: int = 200) -> list:
    """Compact transaction records for the chat prompt (capped to keep prompts fast)."""
    work = df.head(max_rows).copy()
    records = []
    for _, row in work.iterrows():
        records.append({
            "date": row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else None,
            "description": str(row['description'])[:80] if pd.notna(row['description']) else '',
            "deposit": round(float(row['deposit']), 2) if pd.notna(row['deposit']) else None,
            "withdrawal": round(float(row['withdrawal']), 2) if pd.notna(row['withdrawal']) else None,
            "balance": round(float(row['balance']), 2) if pd.notna(row['balance']) else None,
        })
    return records


def build_category_chart_data(df: pd.DataFrame, category_col: str = "category") -> list:
    """
    Groups withdrawal totals by category (assigned by rag_chat.categorize_transactions)
    into [{category, total}] for a pie/bar chart. Returns [] if the category
    column isn't present, so the frontend can simply hide that chart card.
    """
    if category_col not in df.columns:
        return []

    work = df.copy()
    work["withdrawal"] = work["withdrawal"].fillna(0)
    grouped = work.groupby(category_col)["withdrawal"].sum().round(2)
    grouped = grouped[grouped > 0].sort_values(ascending=False)

    return [{"category": cat, "total": total} for cat, total in grouped.items()]


# ---------------------------------------------------------------------------
# Chart data for a web frontend (JSON-shaped, not matplotlib)
# ---------------------------------------------------------------------------

def build_chart_data(df: pd.DataFrame) -> dict:
    """
    Shapes the DataFrame into plain JSON-serializable structures for a
    frontend charting library (e.g. Chart.js). Keeps the web app free of
    server-side image rendering/file I/O, which matters for hosting on
    ephemeral/free-tier environments.
    """
    if df is None or df.empty:
        return {"monthly": [], "balance": [], "top_expenses": [], "daily_net": []}

    work = df.copy()
    work['deposit'] = work['deposit'].fillna(0)
    work['withdrawal'] = work['withdrawal'].fillna(0)
    work = work.dropna(subset=['date']).sort_values('date')

    work['month'] = work['date'].dt.to_period('M').astype(str)
    monthly_grouped = work.groupby('month')[['deposit', 'withdrawal']].sum().round(2)
    monthly = [
        {"month": m, "income": row['deposit'], "expense": row['withdrawal']}
        for m, row in monthly_grouped.iterrows()
    ]

    balance = [
        {"date": row['date'].strftime('%Y-%m-%d'), "balance": round(row['balance'], 2)}
        for _, row in work.dropna(subset=['balance']).iterrows()
    ]

    top_expenses_df = work[work['withdrawal'] > 0].nlargest(10, 'withdrawal')
    top_expenses = [
        {
            "date": row['date'].strftime('%Y-%m-%d'),
            "description": str(row['description'])[:40],
            "amount": round(row['withdrawal'], 2),
        }
        for _, row in top_expenses_df.iterrows()
    ]

    work['net'] = work['deposit'] - work['withdrawal']
    daily_grouped = work.groupby(work['date'].dt.strftime('%Y-%m-%d'))['net'].sum().round(2)
    daily_net = [{"date": d, "net": v} for d, v in daily_grouped.items()]

    return {
        "monthly": monthly,
        "balance": balance,
        "top_expenses": top_expenses,
        "daily_net": daily_net,
    }


# ---------------------------------------------------------------------------
# Run standalone (for quick testing against a CSV export)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_statement.py transactions.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    for col in ['withdrawal', 'deposit', 'balance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print_insights(df)
    generate_charts(df)

    print("\nGenerating LLM insights (requires Ollama running locally)...")
    try:
        insights = generate_llm_insights(df)
        print("\n" + "=" * 60)
        print("LLM INSIGHTS")
        print("=" * 60)
        print(insights)
    except Exception as e:
        print(f"Could not generate LLM insights: {e}")
        print("Make sure Ollama is running and the model is pulled (ollama pull qwen2.5:7b)")
