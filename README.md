# recon-agent

An agentic payment-reconciliation system for ERP accounts receivable, in plain Python (standard library only).

Each bank-statement line goes to an agent. The agent uses tools to find the payer, search open invoices and test invoice combinations. It then either **proposes a match** or **sends the line to a human**. Every proposal passes through guardrails written in code, every step is traced, and every run can be scored against labelled data.

I built it from the work I do every month on Microsoft Dynamics 365 Business Central. Matching receipts to invoices is repetitive and fiddly: aliases, missing references, partial payments, bank charges, one transfer covering several invoices. A mistake there misstates the ledger.

## Quick start

```bash
python -m recon_agent eval                      # offline baseline, no API key needed
python -m recon_agent trace runs/<file>.jsonl --line B10   # replay one decision step by step

export ANTHROPIC_API_KEY=...                    # run the same loop with Claude choosing the tools
python -m recon_agent eval --policy anthropic
python -m unittest discover -s tests -t .
```

## Architecture

```
bank line ──► Agent loop (agent.py) ──► Policy picks next tool (policies.py)
                 │    ▲                    • HeuristicPolicy: deterministic baseline
                 │    │ tool result        • AnthropicPolicy: Claude via tool use (urllib, no SDK)
                 ▼    │
              Tools (tools.py)  ── read: get_bank_line, find_customer, search_open_invoices,
                 │                       find_invoice_combinations
                 │                write: propose_match ─► GUARDRAILS ─► accepted / rejected + reason
                 │                       flag_for_review
                 ▼
           JSONL trace (runs/) ──► evals.py scores vs data/ground_truth.csv
```

Key design decisions:

- **One short episode per bank line.** Each episode's context stays small and independent, so one confusing line can't contaminate the next, and the cost per line is bounded.
- **The model never writes to the ledger.** It can only *propose*. `propose_match` (in `guardrails.py`) enforces the hard rules: the invoices exist, are open, aren't already allocated, belong to one customer; the payment isn't an overpayment; any shortfall is a bank charge (≤ 100) or a single-invoice partial payment; confidence is at least 0.80. A rejection comes back to the agent as a tool result with a reason, so it can fix its proposal or escalate.
- **Deterministic fallback.** The loop has a step budget (8). If the agent loops or never decides, the line is forced to human review. Nothing is left silently unreconciled.
- **Swappable policies.** The heuristic baseline and the LLM share the same tools, loop, guardrails and eval, so you can compare them directly.

## Observability and evaluation

Every run writes `runs/<run_id>-<policy>.jsonl`, one event per step: the model's text, the tool, its arguments, the result, the latency, and the token usage. `python -m recon_agent trace` replays any line's decision.

`evals.py` scores a run against `data/ground_truth.csv`. The 12 lines include exact matches, aliases, a reference without its prefix, a bank charge, a partial payment, a payment covering two invoices, an ambiguous duplicate amount, an unidentified deposit, an overpayment and a bank fee. Metrics are ordered by what they cost a finance team:

| metric | why |
|---|---|
| **wrong auto-posts** | must be 0. A wrong posted match misstates the ledger. The eval exits non-zero if any occur. |
| auto-match precision / recall | how much work is safely automated |
| review rate | how much still lands on an accountant |
| tool calls and tokens per line | cost and latency |

Current baseline (heuristic): 100% accuracy, 0 wrong auto-posts, 33% review rate, 4.0 tool calls per line.

## What I'm still working out

- Calibrating the LLM's self-reported confidence. It's a weak signal today, which is why the hard checks live in code.
- Growing the labelled set from real (anonymised) bank statements, so the eval isn't only testing cases I thought of.
- Connecting the tools to Business Central's OData/REST API instead of CSV files, posting accepted matches as draft cash-receipt journal lines for approval.
