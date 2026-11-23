"""Shared types passed between the agent loop and policies."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = ""


@dataclass
class PolicyStep:
    calls: list[ToolCall]
    text: str = ""
    usage: dict = field(default_factory=dict)
