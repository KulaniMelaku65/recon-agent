"""CLI.

    python -m recon_agent run    --policy heuristic|anthropic
    python -m recon_agent eval   --policy heuristic|anthropic
    python -m recon_agent trace  runs/<file>.jsonl [--line B10]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import run
from .evals import load_truth, render, score
from .policies import make_policy


def cmd_run(a):
    results, trace = run(a.data, make_policy(a.policy), a.runs)
    for r in results:
        inv = ";".join(r["invoices"]) or "-"
        print(f"{r['line_id']}  {r['action']:<6} {inv:<18} {r['reasoning']}")
    print(f"\ntrace: {trace}")
    return results, trace


def cmd_eval(a):
    results, trace = run(a.data, make_policy(a.policy), a.runs)
    report = score(results, load_truth(Path(a.data) / "ground_truth.csv"))
    print(render(report))
    print(f"\ntrace: {trace}")
    return 1 if report["wrong_auto_posts"] else 0


def cmd_trace(a):
    for line in open(a.file, encoding="utf-8"):
        e = json.loads(line)
        if a.line and e.get("line_id") != a.line:
            continue
        if e["kind"] == "tool":
            res = json.dumps(e["result"], default=str)
            print(f"[{e['line_id']} step {e['step']}] {e['tool']}({json.dumps(e['args'])})\n    -> {res[:160]}")
        elif e["kind"] == "thought":
            print(f"[{e['line_id']} step {e['step']}] thought: {e['text'][:200]}")
        elif e["kind"] == "decision":
            print(f"[{e['line_id']}] DECISION {e['decision']['action']} {e['decision'].get('invoices')} "
                  f"- {e['decision']['reasoning']}\n")


def main():
    p = argparse.ArgumentParser(prog="recon_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("eval", cmd_eval)):
        s = sub.add_parser(name)
        s.add_argument("--policy", default="heuristic", choices=["heuristic", "anthropic"])
        s.add_argument("--data", default="data")
        s.add_argument("--runs", default="runs")
        s.set_defaults(fn=fn)
    t = sub.add_parser("trace")
    t.add_argument("file")
    t.add_argument("--line")
    t.set_defaults(fn=cmd_trace)
    a = p.parse_args()
    rc = a.fn(a)
    raise SystemExit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
