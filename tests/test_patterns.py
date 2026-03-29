"""Tests for agent patterns."""

import pytest

from agent_framework.agent import Agent
from agent_framework.patterns import (
    AgentRouter,
    chain_of_thought_planner,
    create_tool_agent,
)


class TestCreateToolAgent:
    def test_creates_agent(self):
        agent = create_tool_agent(
            "test",
            tools={"echo": lambda query: query},
        )
        assert isinstance(agent, Agent)
        assert len(agent.tools) == 1

    def test_with_system_prompt(self):
        agent = create_tool_agent(
            "test",
            tools={},
            system_prompt="Be helpful",
        )
        messages = agent.memory.get_messages()
        assert any("helpful" in m.content for m in messages)


class TestAgentRouter:
    def test_keyword_routing(self):
        math_agent = Agent(name="math")
        search_agent = Agent(name="search")

        router = AgentRouter()
        router.register("math", math_agent, keywords=["calculate", "compute", "math"])
        router.register("search", search_agent, keywords=["find", "search", "look"])

        result = router.route("calculate 2+2")
        assert result is not None

    def test_default_agent(self):
        default = Agent(name="default")
        router = AgentRouter()
        router.set_default(default)

        result = router.route("something random")
        assert result.success

    def test_no_match_raises(self):
        router = AgentRouter()
        with pytest.raises(ValueError):
            router.route("no agents registered")

    def test_len(self):
        router = AgentRouter()
        assert len(router) == 0
        router.register("a", Agent(name="a"))
        assert len(router) == 1

    def test_agents_property(self):
        router = AgentRouter()
        agent = Agent(name="test")
        router.register("test", agent)
        assert "test" in router.agents


class TestChainOfThoughtPlanner:
    def test_produces_steps(self):
        context = {"task": "test", "steps": [], "tools": [], "memory": []}
        step = chain_of_thought_planner(context)
        assert step.thought != ""
        assert step.action is None
