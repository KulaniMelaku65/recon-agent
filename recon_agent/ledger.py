"""In-memory view of the ERP data (bank lines, open invoices, customers) for one run."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class Ledger:
    """In-memory view of the ERP data for one reconciliation run."""

    bank_lines: dict[str, dict]
    invoices: dict[str, dict]
    customers: dict[str, dict]
    allocated: set[str] = field(default_factory=set)
    decisions: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, data_dir: str | Path) -> "Ledger":
        d = Path(data_dir)
        bank = {r["line_id"]: {**r, "amount": float(r["amount"])} for r in _read(d / "bank_statement.csv")}
        inv = {r["invoice_no"]: {**r, "amount": float(r["amount"])} for r in _read(d / "open_invoices.csv")}
        cust = {}
        for r in _read(d / "customers.csv"):
            r["aliases"] = [a.strip() for a in r["aliases"].split(";") if a.strip()]
            cust[r["customer_id"]] = r
        return cls(bank, inv, cust)

    def open_invoices(self) -> list[dict]:
        return [i for n, i in self.invoices.items() if n not in self.allocated]


class ToolError(Exception):
    pass
