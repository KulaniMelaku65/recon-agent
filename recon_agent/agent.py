"""The agent loop: one short episode per bank line.

Each episode is: policy picks tool(s) -> we execute -> result goes back to the
policy -> repeat, until a terminal tool succeeds or the step budget runs out.
Every step is written to a JSONL trace so a run can be replayed and audited.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .tools import TERMINAL_TOOLS, TOOLS, Ledger, ToolError, flag_for_review

MAX_STEPS = 8


@dataclass
class Tracer:
    path: Path
    run_id: str

    def log(self, **event) -> None:
        event = {"run_id": self.run_id, "ts": round(time.time(), 3), **event}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")


def run_episode(ledger: Ledger, policy, line_id: str, tracer: Tracer) -> dict:
    policy.start(line_id)
    history: list = []
    tokens = {"input_tokens": 0, "output_tokens": 0}
    for step in range(1, MAX_STEPS + 1):
        t0 = time.perf_counter()
        decision = policy.next(history)
        think_ms = round((time.perf_counter() - t0) * 1000, 1)
        for k in tokens:
            tokens[k] += decision.usage.get(k, 0)
        if decision.text:
            tracer.log(line_id=line_id, step=step, kind="thought", text=decision.text, latency_ms=think_ms)
        if not decision.calls:
            break
        for call in decision.calls:
            t1 = time.perf_counter()
            try:
                if call.name not in TOOLS:
                    raise ToolError(f"unknown tool {call.name}")
                result = TOOLS[call.name](ledger, **call.args)
            except (ToolError, TypeError) as e:
                result = {"error": str(e)}
            tracer.log(line_id=line_id, step=step, kind="tool", tool=call.name, args=call.args,
                       result=result, latency_ms=round((time.perf_counter() - t1) * 1000, 2),
                       policy_latency_ms=think_ms)
            history.append((call, result))
        if line_id in ledger.decisions:
            # terminal tool succeeded; give the policy no more turns
            if any(c.name in TERMINAL_TOOLS for c in decision.calls):
                break
    else:
        tracer.log(line_id=line_id, kind="budget_exceeded", max_steps=MAX_STEPS)

    if line_id not in ledger.decisions:  # safety net: never leave a line silently undecided
        flag_for_review(ledger, line_id, "agent did not reach a decision within the step budget")
        tracer.log(line_id=line_id, kind="forced_review")

    decision = ledger.decisions[line_id]
    tool_calls = len(history)
    tracer.log(line_id=line_id, kind="decision", decision=decision, tool_calls=tool_calls, **tokens)
    return {"line_id": line_id, **decision, "tool_calls": tool_calls, **tokens}


def run(data_dir: str | Path, policy, runs_dir: str | Path = "runs") -> tuple[list[dict], Path]:
    ledger = Ledger.load(data_dir)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    tracer = Tracer(Path(runs_dir) / f"{run_id}-{policy.name}.jsonl", run_id)
    tracer.log(kind="run_start", policy=policy.name, lines=len(ledger.bank_lines))
    results = [run_episode(ledger, policy, lid, tracer) for lid in ledger.bank_lines]
    tracer.log(kind="run_end")
    return results, tracer.path
