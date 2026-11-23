"""Guardrails: the only code paths that write a decision.

The agent can only *propose* a match. propose_match checks it against hard
business rules and either accepts it or rejects it with a reason the agent can
react to.
"""
from __future__ import annotations

from .ledger import Ledger, ToolError

# Payments may be short by a small bank charge and still settle an invoice.
BANK_CHARGE_TOLERANCE = 100.00
# Below this confidence a proposal is routed to a human instead of auto-posted.
MIN_AUTO_CONFIDENCE = 0.80


def _line(ledger: Ledger, line_id: str) -> dict:
    if line_id not in ledger.bank_lines:
        raise ToolError(f"unknown bank line {line_id}")
    return ledger.bank_lines[line_id]


def propose_match(ledger: Ledger, line_id: str, invoice_nos: list[str],
                  confidence: float, reasoning: str) -> dict:
    """Guardrailed write. Returns {'accepted': bool, ...}."""
    line = _line(ledger, line_id)
    if line_id in ledger.decisions:
        return {"accepted": False, "reason": "bank line already decided"}
    if not invoice_nos:
        return {"accepted": False, "reason": "no invoices given"}
    invs = []
    for n in invoice_nos:
        if n not in ledger.invoices:
            return {"accepted": False, "reason": f"invoice {n} does not exist"}
        if n in ledger.allocated:
            return {"accepted": False, "reason": f"invoice {n} already allocated in this run"}
        invs.append(ledger.invoices[n])
    if len({i["customer_id"] for i in invs}) > 1:
        return {"accepted": False, "reason": "invoices belong to different customers"}
    if line["amount"] <= 0:
        return {"accepted": False, "reason": "bank line is not a receipt"}

    total = sum(i["amount"] for i in invs)
    diff = round(line["amount"] - total, 2)
    if diff > BANK_CHARGE_TOLERANCE:
        return {"accepted": False, "reason": f"overpayment of {diff:.2f} - needs a human to decide (refund/credit)"}
    if diff < -BANK_CHARGE_TOLERANCE:
        if len(invs) != 1:
            return {"accepted": False, "reason": f"short by {-diff:.2f} across several invoices - ambiguous allocation"}
        kind = "partial"
    elif diff != 0:
        kind = "settled_with_bank_charge"
    else:
        kind = "exact"
    if confidence < MIN_AUTO_CONFIDENCE:
        return {"accepted": False, "reason": f"confidence {confidence:.2f} below {MIN_AUTO_CONFIDENCE} - flag for review"}

    for n in invoice_nos:
        if kind != "partial":
            ledger.allocated.add(n)
    ledger.decisions[line_id] = {"action": "match", "invoices": invoice_nos, "kind": kind,
                                 "difference": diff, "confidence": confidence, "reasoning": reasoning}
    return {"accepted": True, "kind": kind, "difference": diff}


def flag_for_review(ledger: Ledger, line_id: str, reason: str) -> dict:
    _line(ledger, line_id)
    if line_id in ledger.decisions:
        return {"ok": False, "reason": "bank line already decided"}
    ledger.decisions[line_id] = {"action": "review", "invoices": [], "reasoning": reason}
    return {"ok": True}
