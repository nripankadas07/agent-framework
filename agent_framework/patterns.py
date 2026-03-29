"""Common agent patterns and compositions.

Ready-to-use agent configurations for common tasks:
chain-of-thought, tool-augmented Q&A, and multi-agent delegation.
"""

from __future__ import annotations

from typing import Any, Callable

from .agent import Agent, AgentResult, AgentStep
from .memory import ConversationMemory
from .tools import ToolRegistry


def chain_of_thought_planner(context: dict) -> AgentStep:
    """Planner that breaks down tasks into reasoning steps.

    Simulates chain-of-thought without an LLM — useful for testing
    and as a template for production planners.
    """
    steps = context["steps"]
    task = context["task"]

    if not steps:
        return AgentStep(
            thought=f"Let me think about this step by step. Task: {task}",
        )

    if len(steps) == 1:
        return AgentStep(
            thought=f"Analyzing the key aspects of: {task}",
        )

    return AgentStep(
        thought=f"Based on my reasoning: The answer to '{task}' involves careful analysis.",
        action="finish",
    )


def create_tool_agent(
    name: str,
    tools: dict[str, Callable],
    system_prompt: str = "",
    max_steps: int = 5,
) -> Agent:
    """Create an agent pre-configured with a set of tools.

    Args:
        name: Agent name.
        tools: Dict mapping tool names to callables.
        system_prompt: Optional system prompt.
        max_steps: Maximum planning steps.

    Returns:
        Configured Agent instance.
    """
    registry = ToolRegistry()
    for tool_name, fn in tools.items():
        registry.add(tool_name, fn)

    return Agent(
        name=name,
        tools=registry,
        system_prompt=system_prompt,
        max_steps=max_steps,
    )


class AgentRouter:
    """Route tasks to specialized sub-agents.

    Implements a simple multi-agent pattern where a router decides
    which specialist agent should handle a given task.

    Usage:
        router = AgentRouter()
        router.register("math", math_agent, keywords=["calculate", "compute"])
        router.register("search", search_agent, keywords=["find", "look up"])

        result = router.route("Calculate 2 + 2")
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._keywords: dict[str, list[str]] = {}
        self._default: Agent | None = None

    def register(self, name: str, agent: Agent, keywords: list[str] | None = None) -> None:
        """Register a specialist agent."""
        self._agents[name] = agent
        if keywords:
            self._keywords[name] = [k.lower() for k in keywords]

    def set_default(self, agent: Agent) -> None:
        """Set the fallback agent for unmatched tasks."""
        self._default = agent

    def route(self, task: str) -> AgentResult:
        """Route a task to the best matching agent."""
        agent = self._match(task)
        if agent is None:
            if self._default:
                agent = self._default
            else:
                raise ValueError(f"No agent matched for task: {task}")
        return agent.run(task)

    def _match(self, task: str) -> Agent | None:
        """Find the best agent for a task using keyword matching."""
        task_lower = task.lower()
        best_agent = None
        best_score = 0

        for name, keywords in self._keywords.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > best_score:
                best_score = score
                best_agent = self._agents[name]

        return best_agent

    @property
    def agents(self) -> dict[str, Agent]:
        return dict(self._agents)

    def __len__(self) -> int:
        return len(self._agents)
