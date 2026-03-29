"""Agent memory systems.

Provides short-term (conversation) and long-term (persistent)
memory for agents, with a sliding-window strategy for context
management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in the conversation."""

    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    tool_name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        d = {"role": self.role.value, "content": self.content}
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


class ConversationMemory:
    """Sliding-window conversation memory.

    Keeps the most recent messages within a token/message budget,
    always preserving the system prompt.
    """

    def __init__(self, max_messages: int = 50) -> None:
        self._messages: list[Message] = []
        self._max_messages = max_messages

    def add(self, role: Role | str, content: str, **kwargs) -> Message:
        """Add a message to memory."""
        if isinstance(role, str):
            role = Role(role)
        msg = Message(role=role, content=content, **kwargs)
        self._messages.append(msg)
        self._trim()
        return msg

    def add_system(self, content: str) -> Message:
        return self.add(Role.SYSTEM, content)

    def add_user(self, content: str) -> Message:
        return self.add(Role.USER, content)

    def add_assistant(self, content: str) -> Message:
        return self.add(Role.ASSISTANT, content)

    def add_tool_result(self, content: str, tool_name: str, tool_call_id: str = "") -> Message:
        return self.add(Role.TOOL, content, tool_name=tool_name, tool_call_id=tool_call_id)

    def get_messages(self) -> list[Message]:
        """Return all messages."""
        return list(self._messages)

    def get_context(self) -> list[dict]:
        """Return messages as dicts for LLM consumption."""
        return [m.to_dict() for m in self._messages]

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def _trim(self) -> None:
        """Trim messages to fit within the budget, preserving system messages."""
        if len(self._messages) <= self._max_messages:
            return

        system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
        other_msgs = [m for m in self._messages if m.role != Role.SYSTEM]

        # Keep system messages + most recent non-system messages
        budget = self._max_messages - len(system_msgs)
        trimmed = other_msgs[-budget:] if budget > 0 else []
        self._messages = system_msgs + trimmed

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def last_message(self) -> Message | None:
        return self._messages[-1] if self._messages else None


class SummaryMemory:
    """Memory that periodically summarizes older messages.

    Useful for long conversations where you want to retain context
    without exceeding token limits. Summarization is pluggable —
    provide your own summarizer function.
    """

    def __init__(
        self,
        summarize_fn: callable | None = None,
        summarize_threshold: int = 20,
    ):
        self._messages: list[Message] = []
        self._summaries: list[str] = []
        self._summarize_fn = summarize_fn or self._default_summarize
        self._threshold = summarize_threshold

    def add(self, role: Role | str, content: str) -> Message:
        if isinstance(role, str):
            role = Role(role)
        msg = Message(role=role, content=content)
        self._messages.append(msg)

        if len(self._messages) > self._threshold:
            self._compress()
        return msg

    def _compress(self) -> None:
        """Summarize older messages and keep recent ones."""
        split = len(self._messages) // 2
        old_msgs = self._messages[:split]
        summary = self._summarize_fn(old_msgs)
        self._summaries.append(summary)
        self._messages = self._messages[split:]

    @staticmethod
    def _default_summarize(messages: list[Message]) -> str:
        """Simple concatenation summarizer (replace with LLM call in prod)."""
        parts = []
        for m in messages:
            parts.append(f"{m.role.value}: {m.content[:100]}")
        return "Summary: " + " | ".join(parts)

    def get_context(self) -> list[dict]:
        """Return summaries + recent messages."""
        context = []
        if self._summaries:
            context.append({
                "role": "system",
                "content": "Previous conversation summary:\n" + "\n".join(self._summaries),
            })
        context.extend(m.to_dict() for m in self._messages)
        return context

    def __len__(self) -> int:
        return len(self._messages) + len(self._summaries)
