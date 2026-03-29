"""Tests for agent memory systems."""

import pytest

from agent_framework.memory import ConversationMemory, SummaryMemory, Role, Message


class TestMessage:
    def test_to_dict(self):
        msg = Message(role=Role.USER, content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_tool_message(self):
        msg = Message(role=Role.TOOL, content="result", tool_name="search")
        d = msg.to_dict()
        assert d["tool_name"] == "search"


class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory()
        mem.add_user("hello")
        mem.add_assistant("hi there")
        assert len(mem) == 2

    def test_get_context(self):
        mem = ConversationMemory()
        mem.add_user("test")
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["role"] == "user"

    def test_sliding_window(self):
        mem = ConversationMemory(max_messages=5)
        for i in range(10):
            mem.add_user(f"message {i}")
        assert len(mem) <= 5

    def test_preserves_system(self):
        mem = ConversationMemory(max_messages=3)
        mem.add_system("I am a helpful assistant")
        for i in range(5):
            mem.add_user(f"msg {i}")
        messages = mem.get_messages()
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_user("test")
        mem.clear()
        assert len(mem) == 0

    def test_last_message(self):
        mem = ConversationMemory()
        assert mem.last_message is None
        mem.add_user("hello")
        assert mem.last_message.content == "hello"

    def test_tool_result(self):
        mem = ConversationMemory()
        mem.add_tool_result("42", tool_name="calculator")
        assert mem.last_message.tool_name == "calculator"


class TestSummaryMemory:
    def test_basic_add(self):
        mem = SummaryMemory(summarize_threshold=100)
        mem.add(Role.USER, "hello")
        assert len(mem) == 1

    def test_compression(self):
        mem = SummaryMemory(summarize_threshold=5)
        for i in range(6):
            mem.add(Role.USER, f"message {i}")
        ctx = mem.get_context()
        assert any("summary" in str(m).lower() for m in ctx)

    def test_get_context(self):
        mem = SummaryMemory()
        mem.add("user", "hello")
        ctx = mem.get_context()
        assert len(ctx) >= 1
