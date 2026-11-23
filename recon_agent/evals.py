"""Score agent decisions against a labelled ground truth.

The metric that matters most in finance is `wrong_auto_posts`: a wrong match
that would have been posted without a human looking at it. A line sent to
review costs a few minutes of an accountant's time; a wrong auto-post costs a
misstated ledger. So the agent is judged on safety first, automation second.
"""
from __future__ import annotations

import csv
from pathlib import Path


def load_truth(path: str | Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["line_id"]: {"action": r["expected_action"],
                               "invoices": sorted(filter(None, r["expected_invoices"].split(";"))),
                               "note": r["note"]} for r in csv.DictReader(f)}


def score(results: list[dict], truth: dict[str, dict]) -> dict:
    rows, correct, auto, auto_ok, wrong_auto, missed = [], 0, 0, 0, 0, 0
    matchable = sum(1 for t in truth.values() if t["action"] == "match")
    for r in results:
        t = truth[r["line_id"]]
        got = sorted(r.get("invoices", []))
        ok = r["action"] == t["action"] and (t["action"] == "review" or got == t["invoices"])
        correct += ok
        if r["action"] == "match":
            auto += 1
            auto_ok += ok
            wrong_auto += not ok
        elif t["action"] == "match":
            missed += 1
        rows.append({"line_id": r["line_id"], "expected": f"{t['action']} {';'.join(t['invoices'])}".strip(),
                     "got": f"{r['action']} {';'.join(got)}".strip(), "ok": ok,
                     "tool_calls": r["tool_calls"], "note": t["note"]})
    n = len(results)
    return {
        "rows": rows,
        "accuracy": correct / n,
        "auto_match_precision": auto_ok / auto if auto else 1.0,
        "auto_match_recall": auto_ok / matchable if matchable else 1.0,
        "review_rate": (n - auto) / n,
        "wrong_auto_posts": wrong_auto,
        "missed_automations": missed,
        "avg_tool_calls": sum(r["tool_calls"] for r in results) / n,
        "tokens": sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in results),
    }


def render(report: dict) -> str:
    out = [f"{'line':<5} {'expected':<22} {'got':<22} {'ok':<3} {'calls':<5} note"]
    for r in report["rows"]:
        out.append(f"{r['line_id']:<5} {r['expected']:<22} {r['got']:<22} {'Y' if r['ok'] else 'N':<3} "
                   f"{r['tool_calls']:<5} {r['note']}")
    out.append("")
    out.append(f"accuracy              {report['accuracy']:.0%}")
    out.append(f"auto-match precision  {report['auto_match_precision']:.0%}")
    out.append(f"auto-match recall     {report['auto_match_recall']:.0%}")
    out.append(f"review rate           {report['review_rate']:.0%}")
    out.append(f"WRONG AUTO-POSTS      {report['wrong_auto_posts']}")
    out.append(f"avg tool calls/line   {report['avg_tool_calls']:.1f}")
    if report["tokens"]:
        out.append(f"tokens                {report['tokens']}")
    return "\n".join(out)
