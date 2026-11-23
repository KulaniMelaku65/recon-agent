"""Policies decide the agent's next tool call.

Two interchangeable policies share the same tools, loop, guardrails and traces:

* HeuristicPolicy - deterministic, offline baseline. Runs anywhere, no API key.
* AnthropicPolicy - an LLM (Claude) choosing tools via the Messages API tool-use
  interface. Uses only the standard library (urllib), no SDK needed.

Because both run through the same loop, the eval harness can compare them on
the same labelled data.
"""
from __future__ import annotations

from .base import PolicyStep, ToolCall
from .llm_policy import SYSTEM_PROMPT, AnthropicPolicy  # noqa: F401  (re-exported)
from .tools import extract_invoice_refs


class HeuristicPolicy:
    """Rule-based planner. Reads the episode history and emits the next call."""

    name = "heuristic"

    def start(self, line_id: str) -> None:
        self.line_id = line_id
        self.s: dict = {"stage": "read"}

    def next(self, history: list[tuple[ToolCall, object]]) -> PolicyStep:
        s, lid = self.s, self.line_id
        last = history[-1][1] if history else None

        def call(name, **args):
            return PolicyStep([ToolCall(name, args)])

        def flag(reason):
            s["stage"] = "done"
            return call("flag_for_review", line_id=lid, reason=reason)

        def propose(invs, conf, why):
            s["stage"] = "proposed"
            return call("propose_match", line_id=lid, invoice_nos=invs, confidence=conf, reasoning=why)

        if s["stage"] == "read":
            s["stage"] = "customer"
            return call("get_bank_line", line_id=lid)

        if s["stage"] == "customer":
            s["line"] = last
            if last["amount"] <= 0:
                return flag("non-positive amount: bank fee/charge, not a customer receipt")
            s["stage"] = "refs"
            return call("find_customer", text=last["description"])

        if s["stage"] == "refs":
            best = last[0] if last else None
            if not best or best["score"] < 0.6:
                return flag("could not identify the payer from the description")
            s["customer"] = best["customer_id"]
            s["refs"] = extract_invoice_refs(s["line"]["description"])
            if s["refs"]:
                s["stage"] = "by_ref"
                return call("search_open_invoices", customer_id=s["customer"], invoice_refs=s["refs"])
            s["stage"] = "by_amount"
            return call("search_open_invoices", customer_id=s["customer"], amount=s["line"]["amount"])

        if s["stage"] == "by_ref":
            if last:
                return propose([r["invoice_no"] for r in last], 0.95, f"invoice refs {s['refs']} in description")
            s["stage"] = "by_amount"
            return call("search_open_invoices", customer_id=s["customer"], amount=s["line"]["amount"])

        if s["stage"] == "by_amount":
            if len(last) == 1:
                return propose([last[0]["invoice_no"]], 0.9, "single open invoice with the exact amount")
            if len(last) > 1:
                return flag(f"{len(last)} open invoices with the same amount - ambiguous")
            s["stage"] = "combos"
            return call("find_invoice_combinations", customer_id=s["customer"], amount=s["line"]["amount"])

        if s["stage"] == "combos":
            if len(last) == 1:
                return propose(last[0], 0.85, "unique combination of open invoices sums to the payment")
            if len(last) > 1:
                return flag("several invoice combinations fit - ambiguous")
            s["stage"] = "near"
            return call("search_open_invoices", customer_id=s["customer"])

        if s["stage"] == "near":
            amt = s["line"]["amount"]
            near = [r for r in last if 0 <= r["amount"] - amt <= 100]
            if len(near) == 1:
                return propose([near[0]["invoice_no"]], 0.85, "amount within bank-charge tolerance")
            return flag("no open invoice explains this amount")

        if s["stage"] == "proposed":
            if isinstance(last, dict) and not last.get("accepted"):
                return flag(f"guardrail rejected proposal: {last.get('reason')}")
            s["stage"] = "done"
            return PolicyStep([], text="done")

        return PolicyStep([], text="done")


def make_policy(name: str):
    if name == "heuristic":
        return HeuristicPolicy()
    if name == "anthropic":
        return AnthropicPolicy()
    raise ValueError(f"unknown policy {name}")
