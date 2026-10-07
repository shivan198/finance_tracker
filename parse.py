"""
Bank Statement PDF Parser (pdfplumber version)

Same column-based transaction parsing approach as the original qpdf-stream
version, but text/position extraction now uses pdfplumber instead of manual
PDF content-stream parsing. This removes the qpdf system dependency and all
the hex/CMap/ToUnicode decoding logic, since pdfplumber handles font decoding
internally.

Usage:
    python bank_statement_extract_pdfplumber.py statement.pdf
    python bank_statement_extract_pdfplumber.py statement.pdf --debug
    python bank_statement_extract_pdfplumber.py statement.pdf --verbose
    python bank_statement_extract_pdfplumber.py statement.pdf --csv transactions.csv
    python bank_statement_extract_pdfplumber.py statement.pdf --amount-columns 420,560,650
"""

import re
import sys
import csv
from collections import defaultdict

import pdfplumber


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------

def is_numeric(text: str) -> bool:
    t = text.strip()
    return bool(re.match(r'^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$', t))


def parse_amount_text(text: str):
    """
    Parse amount-like strings and return normalized numeric text, or None.
    Supports common statement variants like commas, CR/DR suffixes, and trailing '-'.
    """
    t = text.strip()
    if not t:
        return None

    t = t.replace('₹', '').replace('INR', '').replace('Rs.', '').replace('Rs', '').strip()

    # Strip stray single letters glued between digits/commas — a PDF rendering
    # artifact seen in some statements (e.g. "164N,748.80" -> "164,748.80"),
    # likely from an overlapping footnote marker or font-kerning quirk.
    # This runs before CR/DR suffix handling below, so it won't interfere
    # with legitimate 2-letter credit/debit markers at the start or end.
    t = re.sub(r'(?<=\d)[A-Za-z](?=[\d,])', '', t)

    sign = 1

    if t.startswith('(') and t.endswith(')'):
        sign = -1
        t = t[1:-1].strip()

    upper = t.upper()
    if upper.endswith('CR') or upper.endswith('DR'):
        t = t[:-2].strip()
        upper = t.upper()
    if upper.startswith('CR') or upper.startswith('DR'):
        t = t[2:].strip()

    if t.endswith('-'):
        sign = -1
        t = t[:-1].strip()

    t = t.replace(',', '').replace(' ', '')
    if not re.match(r'^-?\d+(?:\.\d+)?$', t):
        return None

    if sign < 0 and not t.startswith('-'):
        t = '-' + t
    return t


def snap_y(y: float, tolerance: float = 1.5) -> float:
    return round(y / tolerance) * tolerance


def parse_float_arg(flag: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{flag} expects a numeric value, got: {value}") from exc


def parse_amount_columns_arg(value: str):
    parts = [part.strip() for part in value.split(',')]
    if len(parts) != 3:
        raise ValueError("--amount-columns expects 3 comma-separated values: deposit_min,withdraw_min,balance_min")

    try:
        deposit_min, withdraw_min, balance_min = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--amount-columns values must be numeric") from exc

    if not (deposit_min < withdraw_min < balance_min):
        raise ValueError("--amount-columns must satisfy deposit_min < withdraw_min < balance_min")

    return deposit_min, withdraw_min, balance_min


# ---------------------------------------------------------------------------
# Stage 1 – extract text blocks with position, using pdfplumber
# ---------------------------------------------------------------------------

def extract_all_blocks(file_path: str) -> list:
    """
    Replacement for the qpdf-based stream parser.

    Produces the same block shape the original code relied on:
        {'x', 'y', 'raw_y', 'text', 'page', 'stream_index'}
    so parse_transactions() below works completely unchanged.

    'stream_index' here represents one text line per page, in top-to-bottom
    reading order — this mirrors the role the original BT/ET stream order
    played (grouping + sequencing text fragments for the lookahead logic).
    """
    all_blocks = []
    stream_index = 0

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)

            if not words:
                continue

            # Group words into lines by y-position (top), tolerant of small offsets
            lines = defaultdict(list)
            for w in words:
                y_key = snap_y(w['top'])
                lines[y_key].append(w)

            # Walk top-to-bottom so stream_index approximates natural reading order
            for y_key in sorted(lines.keys()):
                line_words = sorted(lines[y_key], key=lambda w: w['x0'])
                for w in line_words:
                    all_blocks.append({
                        'x': round(w['x0'], 2),
                        'y': y_key,
                        'raw_y': w['top'],
                        'text': w['text'],
                        'page': page_num,
                        'stream_index': stream_index,
                    })
                stream_index += 1

    return all_blocks


# ---------------------------------------------------------------------------
# Stage 1b – dynamic column detection from header row text
# ---------------------------------------------------------------------------

HEADER_KEYWORDS = {
    'withdrawal': ['withdrawal', 'debit', 'dramount', 'withdrawalamt', 'dr'],
    'deposit': ['deposit', 'credit', 'cramount', 'depositamt', 'cr'],
    'balance': ['balance', 'closingbalance', 'closing', 'availablebalance'],
}


def detect_column_boundaries(blocks: list, verbose: bool = False) -> dict:
    """
    Scan for the actual header row (not the whole page) and return the
    x-positions of Withdrawal/Deposit/Balance-type columns.

    IMPORTANT: we only trust a line where MULTIPLE column keywords appear
    together at the same y-position. Bank narration text frequently contains
    words like 'CREDIT', 'DR', 'CR' (e.g. 'NEFT CREDIT', 'UPI/DR/...'), and a
    naive whole-page scan would mistake those for header columns, dragging
    the detected boundary far to the left into the Narration column. Requiring
    several matches on the same line filters that noise out, since a
    transaction narration line will essentially never contain multiple
    distinct column keywords together the way a real header row does.
    """
    lines = defaultdict(list)
    for b in blocks:
        lines[b['y']].append(b)

    best_matches = {}
    best_score = 0

    for y, line_blocks in lines.items():
        line_matches = defaultdict(list)
        for b in line_blocks:
            text_clean = re.sub(r'[^a-z]', '', b['text'].lower())
            if not text_clean:
                continue
            for col_type, keywords in HEADER_KEYWORDS.items():
                for kw in keywords:
                    kw_clean = re.sub(r'[^a-z]', '', kw)
                    if kw_clean and kw_clean in text_clean:
                        line_matches[col_type].append(b['x'])

        score = len(line_matches)  # how many distinct column types matched on this one line
        if score >= 2 and score > best_score:
            best_score = score
            best_matches = line_matches
            if verbose:
                print(f"[header] candidate header row at y={y:.1f}, matched columns={list(line_matches.keys())}")

    if best_score < 2:
        if verbose:
            print("[header] no line had 2+ column keywords together; header detection failed")
        return {}

    boundaries = {k: min(v) for k, v in best_matches.items()}
    if verbose:
        print(f"[header] using header row boundaries: {boundaries}")
    return boundaries


def build_dynamic_thresholds(boundaries: dict, verbose: bool = False):
    """
    Convert detected header x-positions into deposit/withdraw/balance ranges.
    Returns None if fewer than 2 of the 3 amount columns were detected,
    since a single point isn't enough to infer sensible ranges.
    """
    if len(boundaries) < 2:
        return None

    # Order columns left-to-right by their detected x-position (order varies by bank)
    ordered = sorted(boundaries.items(), key=lambda item: item[1])
    edges = [x for _, x in ordered] + [ordered[-1][1] + 500]  # generous right-side catch-all

    thresholds = {}
    for i, (col_name, _) in enumerate(ordered):
        left = edges[i] - 8   # small buffer so header text itself doesn't fall outside its own range
        right = edges[i + 1] - 8
        thresholds[col_name] = (left, right)

    if verbose:
        print(f"[thresholds] dynamically detected: {thresholds}")

    return thresholds


# ---------------------------------------------------------------------------
# Currency detection — infer currency from the statement's own text rather
# than assuming USD. Checks literal symbols first (most direct signal), then
# falls back to bank-network keywords that are strong regional indicators.
# ---------------------------------------------------------------------------

CURRENCY_SYMBOLS = {
    '₹': {'code': 'INR', 'symbol': '₹'},
    '£': {'code': 'GBP', 'symbol': '£'},
    '€': {'code': 'EUR', 'symbol': '€'},
    '¥': {'code': 'JPY', 'symbol': '¥'},
}

CURRENCY_KEYWORDS = {
    'INR': {'symbol': '₹', 'keywords': [
        'upi', 'neft', 'imps', 'rtgs', 'ifsc', '@ybl', '@okhdfc', '@okicici',
        '@oksbi', '@okaxis', 'paytm', 'gpay', 'phonepe', 'inr',
    ]},
    'USD': {'symbol': '$', 'keywords': [
        'ach', 'routing number', 'swift', 'wire transfer', 'usd',
    ]},
    'GBP': {'symbol': '£', 'keywords': [
        'sort code', 'bacs', 'chaps', 'faster payment', 'gbp',
    ]},
    'EUR': {'symbol': '€', 'keywords': ['iban', 'sepa', 'eur']},
}


def detect_currency(blocks: list, verbose: bool = False) -> dict:
    """
    Returns {'code': 'INR', 'symbol': '₹', 'confidence': 'high'|'low'}.
    Falls back to USD with low confidence if nothing matches, rather than
    silently guessing — callers can surface that uncertainty if useful.
    """
    all_text = ' '.join(b['text'] for b in blocks)

    for symbol, info in CURRENCY_SYMBOLS.items():
        if symbol in all_text:
            if verbose:
                print(f"[currency] literal symbol '{symbol}' found -> {info['code']}")
            return {'code': info['code'], 'symbol': info['symbol'], 'confidence': 'high'}

    text_lower = all_text.lower()
    scores = {}
    for code, info in CURRENCY_KEYWORDS.items():
        score = sum(text_lower.count(kw) for kw in info['keywords'])
        if score > 0:
            scores[code] = score

    if scores:
        best_code = max(scores, key=scores.get)
        if verbose:
            print(f"[currency] keyword scores={scores} -> {best_code}")
        return {'code': best_code, 'symbol': CURRENCY_KEYWORDS[best_code]['symbol'], 'confidence': 'high'}

    if verbose:
        print("[currency] no signal found, defaulting to USD (low confidence)")
    return {'code': 'USD', 'symbol': '$', 'confidence': 'low'}


# ---------------------------------------------------------------------------
# Stage 2 – transaction parsing based on x/y position (unchanged logic)
# ---------------------------------------------------------------------------

DATE_RE = re.compile(
    r'\b\d{2}-(?:\d{2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}(?:\d{2})?\b'
    r'|\b\d{2}/\d{2}/\d{2,4}\b',
    re.IGNORECASE
)


def parse_transactions(blocks: list,
                        date_x_max: float = 100.0,
                        desc_x_min: float = 50.0,
                        desc_x_max: float = 350.0,
                        deposit_x_min: float = 350.0,
                        deposit_x_max: float = 420.0,
                        withdraw_x_min: float = 420.0,
                        withdraw_x_max: float = 500.0,
                        balance_x_min: float = 500.0,
                        amount_y_tolerance: float = 5.0,
                        desc_lookahead: int = 3,
                        verbose: bool = False) -> list:
    """
    For each DATE block found:
      - Amounts: same Y as date, X in amount columns
      - Description: the next stream index(es) after date that contain
        text in the description X range and no date / no amount-column numbers.
    """
    blocks_stream = sorted(blocks, key=lambda b: b['stream_index'])

    by_si = defaultdict(list)
    for b in blocks_stream:
        by_si[b['stream_index']].append(b)

    si_list = sorted(by_si.keys())
    transactions = []
    consumed_si = set()

    def vprint(msg: str):
        if verbose:
            print(msg)

    def classify_amount_by_x(x: float):
        if x >= balance_x_min:
            return 'balance'
        if deposit_x_min <= x < deposit_x_max:
            return 'deposit'
        if withdraw_x_min <= x < withdraw_x_max:
            return 'withdrawal'
        return None

    for si_pos, si in enumerate(si_list):
        if si in consumed_si:
            continue

        si_blocks = by_si[si]
        page = si_blocks[0]['page']

        date_block = None
        for b in si_blocks:
            if DATE_RE.search(b['text']) and b['x'] < date_x_max:
                date_block = b
                break
        if date_block is None:
            continue

        found_date = DATE_RE.search(date_block['text']).group(0)
        date_y = date_block['y']
        consumed_si.add(si)
        vprint(
            f"[detect] date={found_date} page={page} si={si} y={date_y:.1f} "
            f"x={date_block['x']:.2f} text='{date_block['text'][:80]}'"
        )

        found_amounts = {'deposit': '', 'withdrawal': '', 'balance': ''}
        desc_lines = []

        for b in si_blocks:
            if b is date_block:
                continue
            parsed_amount = parse_amount_text(b['text'])
            same_row = abs(b['y'] - date_y) < amount_y_tolerance
            bucket = classify_amount_by_x(b['x']) if same_row and parsed_amount is not None else None

            if bucket:
                found_amounts[bucket] = parsed_amount
                vprint(f"[amount] si={si} y={b['y']:.1f} x={b['x']:.2f} -> {bucket}={parsed_amount} (same-si)")
                continue

            # Not a date, not a recognized amount -> treat as narration on the
            # date's own line. This was previously dropped entirely, which is
            # why descriptions could be missing their first few words/lines.
            # Skip anything that still LOOKS like a stray/garbled number
            # (is_numeric) so we don't glue rendering artifacts into narration.
            text_val = b['text'].strip()
            if text_val and b['x'] >= desc_x_min and not DATE_RE.search(text_val) and not is_numeric(text_val):
                desc_lines.append(text_val)
                vprint(f"[desc] si={si} y={b['y']:.1f} x={b['x']:.2f} '{text_val[:100]}' (same-si)")

        for fwd_pos in range(si_pos + 1, min(si_pos + 1 + desc_lookahead + 10, len(si_list))):
            fsi = si_list[fwd_pos]
            if fsi in consumed_si:
                continue

            fblocks = by_si[fsi]
            fpage = fblocks[0]['page']
            if fpage != page:
                break

            amount_blocks = [
                b for b in fblocks
                if abs(b['y'] - date_y) < amount_y_tolerance
                and parse_amount_text(b['text']) is not None
                and b['x'] >= deposit_x_min
            ]

            # NOTE: no upper x-bound here anymore. A hardcoded desc_x_max
            # created a "dead zone" for statements whose narration column is
            # wider than expected -- text landing between desc_x_max and the
            # real amount column start was silently dropped. Instead: if it's
            # not a date and doesn't classify into a known amount column,
            # treat it as narration.
            desc_blocks = [
                b for b in fblocks
                if b['x'] >= desc_x_min
                and not DATE_RE.search(b['text'])
                and classify_amount_by_x(b['x']) is None
                and not is_numeric(b['text'].strip())
            ]

            has_new_date = any(
                DATE_RE.search(b['text']) and b['x'] < date_x_max
                for b in fblocks
            )
            if has_new_date:
                vprint(f"[stop] encountered next date candidate at si={fsi}; ending lookahead for si={si}")
                break

            if amount_blocks:
                consumed_si.add(fsi)
                for b in amount_blocks:
                    bucket = classify_amount_by_x(b['x'])
                    if not bucket:
                        continue
                    parsed_amount = parse_amount_text(b['text'])
                    if parsed_amount is None:
                        continue
                    found_amounts[bucket] = parsed_amount
                    vprint(f"[amount] si={fsi} y={b['y']:.1f} x={b['x']:.2f} -> {bucket}={parsed_amount}")

            if desc_blocks:
                consumed_si.add(fsi)
                for b in sorted(desc_blocks, key=lambda b: -b['y']):
                    line = b['text'].strip()
                    if line:
                        desc_lines.append(line)
                        vprint(f"[desc] si={fsi} y={b['y']:.1f} x={b['x']:.2f} '{line[:100]}'")

        deposit = found_amounts['deposit']
        withdrawal = found_amounts['withdrawal']
        balance = found_amounts['balance']

        if not balance:
            for b in si_blocks:
                parsed_amount = parse_amount_text(b['text'])
                if parsed_amount is not None and b['x'] >= balance_x_min:
                    balance = parsed_amount
                    vprint(f"[fallback] same-si balance from x={b['x']:.2f} y={b['y']:.1f} -> {balance}")

        description = ' '.join(desc_lines).strip()

        if found_date and (balance or deposit or withdrawal):
            vprint(
                f"[tx] date={found_date} withdrawal={withdrawal or '-'} "
                f"deposit={deposit or '-'} balance={balance or '-'} "
                f"desc_lines={len(desc_lines)}"
            )
            transactions.append({
                'date': found_date,
                'description': description,
                'deposit': deposit,
                'withdrawal': withdrawal,
                'balance': balance,
            })
        else:
            vprint(f"[skip] date={found_date} had no usable amounts")

    return transactions


# ---------------------------------------------------------------------------
# Stage 3 – validation safety net (catches what detection/parsing misses)
# ---------------------------------------------------------------------------

def validate_transactions(transactions: list) -> list:
    """
    Golden-rule check: previous balance - withdrawal + deposit == current balance.
    Returns a list of indices into `transactions` whose numbers don't add up —
    these are exactly the rows most likely to have a missed/misclassified amount,
    regardless of which bank or layout produced them.
    """
    flagged = []

    def to_float(val):
        try:
            return float(val) if val else 0.0
        except ValueError:
            return None

    for i in range(1, len(transactions)):
        prev_bal = to_float(transactions[i - 1]['balance'])
        curr_bal = to_float(transactions[i]['balance'])
        withdrawal = to_float(transactions[i]['withdrawal'])
        deposit = to_float(transactions[i]['deposit'])

        if None in (prev_bal, curr_bal, withdrawal, deposit):
            flagged.append(i)
            continue

        expected = prev_bal - withdrawal + deposit
        if abs(expected - curr_bal) > 0.01:
            flagged.append(i)

    return flagged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_bank_statement(file_path: str,
                          date_x_max: float = 100.0,
                          desc_x_min: float = 50.0,
                          desc_x_max: float = 350.0,
                          deposit_x_min: float = 350.0,
                          deposit_x_max: float = 420.0,
                          withdraw_x_min: float = 420.0,
                          withdraw_x_max: float = 500.0,
                          balance_x_min: float = 500.0,
                          amount_y_tolerance: float = 5.0,
                          auto_detect_columns: bool = True,
                          verbose: bool = False) -> dict:
    """
    Returns a dict:
        {
            'transactions': [...],
            'flagged_indices': [...],   # rows that failed the balance-continuity check
            'used_dynamic_thresholds': bool,
        }
    """
    blocks = extract_all_blocks(file_path)
    if verbose:
        print(f"[path] pdfplumber blocks count={len(blocks)}")

    used_dynamic = False

    if auto_detect_columns:
        boundaries = detect_column_boundaries(blocks, verbose=verbose)
        dynamic_thresholds = build_dynamic_thresholds(boundaries, verbose=verbose)

        if dynamic_thresholds and all(k in dynamic_thresholds for k in ('withdrawal', 'deposit', 'balance')):
            deposit_x_min, deposit_x_max = dynamic_thresholds['deposit']
            withdraw_x_min, withdraw_x_max = dynamic_thresholds['withdrawal']
            balance_x_min, _ = dynamic_thresholds['balance']
            used_dynamic = True
            if verbose:
                print("[path] using dynamically detected column boundaries")
        else:
            if verbose:
                print("[path] header detection incomplete, falling back to provided/default thresholds")

    rows = parse_transactions(
        blocks,
        date_x_max=date_x_max,
        desc_x_min=desc_x_min,
        desc_x_max=desc_x_max,
        deposit_x_min=deposit_x_min,
        deposit_x_max=deposit_x_max,
        withdraw_x_min=withdraw_x_min,
        withdraw_x_max=withdraw_x_max,
        balance_x_min=balance_x_min,
        amount_y_tolerance=amount_y_tolerance,
        verbose=verbose,
    )

    flagged = validate_transactions(rows)
    currency = detect_currency(blocks, verbose=verbose)

    return {
        'transactions': rows,
        'flagged_indices': flagged,
        'used_dynamic_thresholds': used_dynamic,
        'currency': currency,
    }


# ---------------------------------------------------------------------------
# Debug – stream order view
# ---------------------------------------------------------------------------

def debug_blocks(file_path: str):
    blocks = extract_all_blocks(file_path)
    print(f"{'SI':>5}  {'PG':>3}  {'Y':>7}  {'X':>7}  TEXT")
    print('-' * 72)
    for b in sorted(blocks, key=lambda b: (b['stream_index'], -b['y'])):
        marker = ' <DATE>' if DATE_RE.search(b['text']) else \
                 ' <AMT>' if is_numeric(b['text']) else ''
        print(f"{b['stream_index']:>5}  {b['page']:>3}  {b['y']:>7.1f}  "
              f"{b['x']:>7.2f}  {b['text'][:55]}{marker}")


def export_transactions_csv(rows: list, output_path: str):
    fieldnames = ['date', 'description', 'withdrawal', 'deposit', 'balance']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = sys.argv[1:]
    pdf_path = 'statement.pdf'
    verbose = False
    debug = False
    csv_path = None
    auto_detect_columns = True

    date_x_max = 100.0
    desc_x_min = 50.0
    desc_x_max = 350.0
    deposit_x_min = 350.0
    deposit_x_max = 420.0
    withdraw_x_min = 420.0
    withdraw_x_max = 500.0
    balance_x_min = 500.0
    amount_y_tolerance = 5.0

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--verbose':
            verbose = True
        elif arg == '--debug':
            debug = True
        elif arg == '--no-auto-detect':
            auto_detect_columns = False
        elif arg == '--amount-columns':
            if i + 1 >= len(args):
                print("Error: --amount-columns requires a value like 420,560,650", file=sys.stderr)
                sys.exit(2)
            try:
                deposit_x_min, withdraw_x_min, balance_x_min = parse_amount_columns_arg(args[i + 1])
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(2)
            deposit_x_max = withdraw_x_min
            withdraw_x_max = balance_x_min
            i += 1
        elif arg == '--csv':
            if i + 1 >= len(args):
                print("Error: --csv requires an output file path.", file=sys.stderr)
                sys.exit(2)
            csv_path = args[i + 1]
            i += 1
        elif arg in {
            '--date-x-max',
            '--desc-x-min',
            '--desc-x-max',
            '--deposit-x-min',
            '--deposit-x-max',
            '--withdraw-x-min',
            '--withdraw-x-max',
            '--balance-x-min',
            '--amount-y-tolerance',
        }:
            if i + 1 >= len(args):
                print(f"Error: {arg} requires a numeric value.", file=sys.stderr)
                sys.exit(2)
            try:
                value = parse_float_arg(arg, args[i + 1])
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(2)

            if arg == '--date-x-max':
                date_x_max = value
            elif arg == '--desc-x-min':
                desc_x_min = value
            elif arg == '--desc-x-max':
                desc_x_max = value
            elif arg == '--deposit-x-min':
                deposit_x_min = value
            elif arg == '--deposit-x-max':
                deposit_x_max = value
            elif arg == '--withdraw-x-min':
                withdraw_x_min = value
            elif arg == '--withdraw-x-max':
                withdraw_x_max = value
            elif arg == '--balance-x-min':
                balance_x_min = value
            elif arg == '--amount-y-tolerance':
                amount_y_tolerance = value
            i += 1
        elif arg.startswith('--'):
            print(f"Error: unknown option {arg}", file=sys.stderr)
            sys.exit(2)
        else:
            pdf_path = arg
        i += 1

    if debug:
        debug_blocks(pdf_path)
        sys.exit(0)

    result = parse_bank_statement(
        pdf_path,
        date_x_max=date_x_max,
        desc_x_min=desc_x_min,
        desc_x_max=desc_x_max,
        deposit_x_min=deposit_x_min,
        deposit_x_max=deposit_x_max,
        withdraw_x_min=withdraw_x_min,
        withdraw_x_max=withdraw_x_max,
        balance_x_min=balance_x_min,
        amount_y_tolerance=amount_y_tolerance,
        auto_detect_columns=auto_detect_columns,
        verbose=verbose,
    )

    data = result['transactions']
    flagged = result['flagged_indices']

    mode = "auto-detected from headers" if result['used_dynamic_thresholds'] else "manual/default thresholds"
    print(f"[columns] {mode}")

    if csv_path:
        export_transactions_csv(data, csv_path)
        print(f"CSV exported: {csv_path} ({len(data)} rows)")

    if not data:
        print("No transactions found.")
        print("Tip: run with --debug and adjust X thresholds if needed, "
              "or pass --amount-columns manually.")
    else:
        hdr = (f"{'DATE':<12}  {'DESCRIPTION':<40}  "
               f"{'WITHDRAWAL':>12}  {'DEPOSIT':>12}  {'BALANCE':>14}  FLAG")
        print(hdr)
        print('-' * len(hdr))
        for idx, row in enumerate(data):
            flag_marker = ' <-- CHECK' if idx in flagged else ''
            print(
                f"{row['date']:<12}  "
                f"{row['description'][:40]:<40}  "
                f"{row['withdrawal']:>12}  "
                f"{row['deposit']:>12}  "
                f"{row['balance']:>14}"
                f"{flag_marker}"
            )
        print(f"\nTotal transactions: {len(data)}")

        if flagged:
            print(f"\n⚠ {len(flagged)} row(s) failed balance validation "
                  f"(prev_balance - withdrawal + deposit != current_balance):")
            print(f"  Row indices: {flagged}")
            print("  These likely have a missed or misclassified withdrawal/deposit amount.")
            print("  Consider re-running with --debug to inspect, or adjusting --amount-columns.")
        else:
            print("\n✓ All rows passed balance validation.")