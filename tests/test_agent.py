"""Tests for the core agent."""

import pytest

from agent_framework.agent import Agent, AgentState, AgentStep
from agent_framework.tools import ToolRegistry


class TestAgent:
    def test_basic_run(self):
        agent = Agent(name="test")
        result = agent.run("What is 2+2?")
        assert result.success
        assert result.answer != ""

    def test_with_tools(self):
        registry = ToolRegistry()
        registry.add("calculator", lambda query: "42", "Calculate things")

        agent = Agent(name="math", tools=registry)
        result = agent.run("Calculate something")
        assert result.success
        assert result.num_steps > 0

    def test_max_steps(self):
        def infinite_planner(ctx):
            return AgentStep(thought="thinking...", action="noop", action_input={})

        registry = ToolRegistry()
        registry.add("noop", lambda: "ok")

        agent = Agent(name="stuck", tools=registry, planner=infinite_planner, max_steps=3)
        result = agent.run("do something")
        assert not result.success
        assert result.num_steps == 3

    def test_custom_planner(self):
        def my_planner(ctx):
            return AgentStep(thought="I know the answer: 42", action="finish")

        agent = Agent(name="smart", planner=my_planner)
        result = agent.run("What is the meaning of life?")
        assert result.success
        assert "42" in result.answer

    def test_state_transitions(self):
        agent = Agent(name="test")
        assert agent.state == AgentState.IDLE
        result = agent.run("test task")
        assert agent.state in (AgentState.DONE, AgentState.ERROR)

    def test_system_prompt(self):
        agent = Agent(name="test", system_prompt="You are helpful.")
        messages = agent.memory.get_messages()
        assert any(m.content == "You are helpful." for m in messages)

    def test_repr(self):
        agent = Agent(name="demo")
        r = repr(agent)
        assert "demo" in r
        assert "Agent" in r

    def test_tool_error_handling(self):
        def failing_planner(ctx):
            if not ctx["steps"]:
                return AgentStep(thought="trying", action="bad_tool", action_input={})
            return AgentStep(thought="done", action="finish")

        registry = ToolRegistry()
        registry.add("bad_tool", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        agent = Agent(name="error_test", tools=registry, planner=failing_planner)
        result = agent.run("test")
        assert result.num_steps > 0
