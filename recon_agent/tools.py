"""Tools the agent can call, registered for the agent.

Read tools live here; the two write tools (propose_match, flag_for_review) live
in guardrails.py.
"""
from __future__ import annotations

import difflib
import itertools
import re

from .guardrails import (BANK_CHARGE_TOLERANCE, MIN_AUTO_CONFIDENCE,  # noqa: F401  (re-exported)
                         flag_for_review, propose_match)
from .ledger import Ledger, ToolError  # noqa: F401  (re-exported)
from .tool_specs import TOOL_SPECS  # noqa: F401  (re-exported)


# --------------------------------------------------------------------------- tools

def get_bank_line(ledger: Ledger, line_id: str) -> dict:
    if line_id not in ledger.bank_lines:
        raise ToolError(f"unknown bank line {line_id}")
    return ledger.bank_lines[line_id]


def find_customer(ledger: Ledger, text: str, limit: int = 3) -> list[dict]:
    """Fuzzy-match free text from a bank description to known customers."""
    text_u = text.upper()
    scored = []
    for c in ledger.customers.values():
        best = 0.0
        for cand in [c["name"].upper(), *[a.upper() for a in c["aliases"]]]:
            if cand in text_u:
                best = max(best, 1.0)
                continue
            # compare against the leading words of the description
            head = " ".join(text_u.split()[: len(cand.split()) + 1])
            best = max(best, difflib.SequenceMatcher(None, cand, head).ratio())
        scored.append({"customer_id": c["customer_id"], "name": c["name"], "score": round(best, 2)})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:limit]


REF_RE = re.compile(r"\b(?:SI-?|INV\s?)?(\d{4})\b", re.IGNORECASE)


def extract_invoice_refs(text: str) -> list[str]:
    return [f"SI-{m}" for m in REF_RE.findall(text)]


def search_open_invoices(ledger: Ledger, customer_id: str | None = None,
                         invoice_refs: list[str] | None = None,
                         amount: float | None = None) -> list[dict]:
    rows = ledger.open_invoices()
    if customer_id:
        rows = [r for r in rows if r["customer_id"] == customer_id]
    if invoice_refs:
        refs = {r.upper() for r in invoice_refs}
        rows = [r for r in rows if r["invoice_no"].upper() in refs]
    if amount is not None:
        rows = [r for r in rows if abs(r["amount"] - amount) < 0.005]
    return rows


def find_invoice_combinations(ledger: Ledger, customer_id: str, amount: float,
                              max_size: int = 3) -> list[list[str]]:
    """Sets of a customer's open invoices that add up to the payment."""
    rows = search_open_invoices(ledger, customer_id=customer_id)
    hits = []
    for k in range(2, max_size + 1):
        for combo in itertools.combinations(rows, k):
            if abs(sum(r["amount"] for r in combo) - amount) < 0.005:
                hits.append([r["invoice_no"] for r in combo])
    return hits


TOOLS = {
    "get_bank_line": get_bank_line,
    "find_customer": find_customer,
    "search_open_invoices": search_open_invoices,
    "find_invoice_combinations": find_invoice_combinations,
    "propose_match": propose_match,
    "flag_for_review": flag_for_review,
}
TERMINAL_TOOLS = {"propose_match", "flag_for_review"}
