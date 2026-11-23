"""AnthropicPolicy: Claude chooses the tools via the Messages API tool-use
interface. Standard library only (urllib), no SDK."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import PolicyStep, ToolCall
from .tool_specs import TOOL_SPECS

SYSTEM_PROMPT = """You are a reconciliation agent for an ERP accounts-receivable team.
For ONE bank statement line, decide whether it pays open customer invoices.

Procedure:
1. Read the line. Non-positive amounts are fees/charges, not receipts -> flag_for_review.
2. Identify the customer with find_customer. If no candidate scores >= 0.6, flag it.
3. If the description contains invoice references, search those first.
4. Otherwise look for an open invoice with the exact amount, then for combinations.
5. If several invoices fit equally well, do NOT guess - flag_for_review.
6. Call propose_match with an honest confidence. Guardrails may reject it; read
   the reason and either fix the proposal or flag_for_review.
Finish by calling exactly one of propose_match (accepted) or flag_for_review."""


class AnthropicPolicy:
    """Claude choosing tools. Needs ANTHROPIC_API_KEY."""

    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("set ANTHROPIC_API_KEY to use the anthropic policy")
        self.model = model or os.environ.get("RECON_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def start(self, line_id: str) -> None:
        self.messages = [{"role": "user", "content": f"Reconcile bank line {line_id}."}]

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST", headers={
            "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 529, 500) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise

    def next(self, history: list[tuple[ToolCall, object]]) -> PolicyStep:
        # feed back results of the previous step's calls
        if history and getattr(self, "_pending", None):
            results = []
            for call, result in history[-len(self._pending):]:
                results.append({"type": "tool_result", "tool_use_id": call.id,
                                 "content": json.dumps(result, default=str),
                                 "is_error": isinstance(result, dict) and "error" in result})
            self.messages.append({"role": "user", "content": results})
        resp = self._post({"model": self.model, "max_tokens": self.max_tokens, "system": SYSTEM_PROMPT,
                           "tools": TOOL_SPECS, "messages": self.messages})
        self.messages.append({"role": "assistant", "content": resp["content"]})
        calls = [ToolCall(b["name"], b["input"], b["id"]) for b in resp["content"] if b["type"] == "tool_use"]
        text = " ".join(b["text"] for b in resp["content"] if b["type"] == "text")
        self._pending = calls
        return PolicyStep(calls, text=text, usage=resp.get("usage", {}))
