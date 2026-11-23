import json
import tempfile
import unittest
from pathlib import Path

from recon_agent.agent import Tracer, run, run_episode
from recon_agent.evals import load_truth, score
from recon_agent.policies import AnthropicPolicy, HeuristicPolicy, PolicyStep, ToolCall
from recon_agent.tools import Ledger, extract_invoice_refs, propose_match

DATA = Path(__file__).resolve().parent.parent / "data"


def tracer():
    return Tracer(Path(tempfile.mkdtemp()) / "t.jsonl", "test")


class GuardrailTests(unittest.TestCase):
    def setUp(self):
        self.l = Ledger.load(DATA)

    def test_overpayment_rejected(self):
        r = propose_match(self.l, "B10", ["SI-1011"], 0.99, "x")
        self.assertFalse(r["accepted"])
        self.assertIn("overpayment", r["reason"])

    def test_low_confidence_rejected(self):
        self.assertFalse(propose_match(self.l, "B01", ["SI-1001"], 0.5, "x")["accepted"])

    def test_cross_customer_rejected(self):
        self.assertFalse(propose_match(self.l, "B05", ["SI-1007", "SI-1003"], 0.9, "x")["accepted"])

    def test_no_double_allocation(self):
        self.assertTrue(propose_match(self.l, "B01", ["SI-1001"], 0.9, "x")["accepted"])
        self.assertFalse(propose_match(self.l, "B09", ["SI-1001"], 0.9, "x")["accepted"])

    def test_bank_charge_tolerance(self):
        r = propose_match(self.l, "B04", ["SI-1006"], 0.9, "x")
        self.assertEqual(r["kind"], "settled_with_bank_charge")

    def test_fee_line_not_a_receipt(self):
        self.assertFalse(propose_match(self.l, "B12", ["SI-1001"], 0.9, "x")["accepted"])

    def test_refs(self):
        self.assertEqual(extract_invoice_refs("TANA LOG INV SI-1007 SI-1008"), ["SI-1007", "SI-1008"])
        self.assertEqual(extract_invoice_refs("PAYMT INV 1003"), ["SI-1003"])


class LoopTests(unittest.TestCase):
    def test_runaway_policy_is_forced_to_review(self):
        class Looper:
            name = "looper"
            def start(self, line_id): self.lid = line_id
            def next(self, history): return PolicyStep([ToolCall("get_bank_line", {"line_id": self.lid})])
        r = run_episode(Ledger.load(DATA), Looper(), "B01", tracer())
        self.assertEqual(r["action"], "review")

    def test_bad_tool_call_becomes_error_result(self):
        class Bad:
            name = "bad"
            def start(self, line_id): self.n = 0
            def next(self, history):
                self.n += 1
                if self.n == 1:
                    return PolicyStep([ToolCall("delete_ledger", {})])
                self.seen = history[-1][1]
                return PolicyStep([ToolCall("flag_for_review", {"line_id": "B01", "reason": "r"})])
        p = Bad()
        run_episode(Ledger.load(DATA), p, "B01", tracer())
        self.assertIn("error", p.seen)

    def test_heuristic_baseline_is_safe(self):
        results, _ = run(DATA, HeuristicPolicy(), tempfile.mkdtemp())
        rep = score(results, load_truth(DATA / "ground_truth.csv"))
        self.assertEqual(rep["wrong_auto_posts"], 0)


class AnthropicPlumbingTests(unittest.TestCase):
    """Checks the tool-use message protocol without calling the API."""

    def test_tool_results_are_fed_back(self):
        import os
        os.environ.setdefault("ANTHROPIC_API_KEY", "test")
        p = AnthropicPolicy()
        replies = iter([
            {"content": [{"type": "text", "text": "Reading line."},
                         {"type": "tool_use", "id": "t1", "name": "get_bank_line", "input": {"line_id": "B01"}}],
             "usage": {"input_tokens": 10, "output_tokens": 5}},
            {"content": [{"type": "tool_use", "id": "t2", "name": "propose_match",
                          "input": {"line_id": "B01", "invoice_nos": ["SI-1001"], "confidence": 0.95,
                                    "reasoning": "ref and amount match"}}],
             "usage": {"input_tokens": 20, "output_tokens": 8}},
        ])
        sent = []
        p._post = lambda body: (sent.append(json.loads(json.dumps(body))), next(replies))[1]
        r = run_episode(Ledger.load(DATA), p, "B01", tracer())
        self.assertEqual(r["action"], "match")
        self.assertEqual(r["input_tokens"], 30)
        tool_result = sent[1]["messages"][-1]["content"][0]
        self.assertEqual(tool_result["tool_use_id"], "t1")


if __name__ == "__main__":
    unittest.main()
