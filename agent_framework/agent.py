"""Core agent with ReAct-style planning loop.

The agent follows a Reason → Act → Observe cycle:
1. Reason about the current state and decide what to do
2. Act by calling a tool
3. Observe the result and decide if the goal is met
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .memory import ConversationMemory
from .tools import ToolRegistry


class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single step in the agent's execution."""

    thought: str
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str | None = None
    error: str | None = None


@dataclass
class AgentResult:
    """Final result of an agent run."""

    answer: str
    steps: list[AgentStep]
    success: bool

    @property
    def num_steps(self) -> int:
        return len(self.steps)


class Agent:
    """ReAct-style agent with tool use and memory.

    The agent uses a planning function to decide actions. In production,
    this would be an LLM call. For testing, you can provide a simple
    rule-based planner.

    Usage:
        registry = ToolRegistry()
        registry.add("search", lambda query: f"Results for {query}")

        agent = Agent(
            name="researcher",
            tools=registry,
            planner=my_planner_fn,
        )
        result = agent.run("Find information about RAG")
    """

    def __init__(
        self,
        name: str = "agent",
        tools: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        planner: Callable | None = None,
        system_prompt: str = "",
        max_steps: int = 10,
    ):
        self.name = name
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory()
        self._planner = planner or self._default_planner
        self.max_steps = max_steps
        self.state = AgentState.IDLE

        if system_prompt:
            self.memory.add_system(system_prompt)

    def run(self, task: str) -> AgentResult:
        """Execute the agent on a task, returning the final result."""
        self.state = AgentState.THINKING
        self.memory.add_user(task)
        steps: list[AgentStep] = []

        for i in range(self.max_steps):
            # Plan next action
            step = self._plan(task, steps)
            steps.append(step)

            if step.error:
                self.state = AgentState.ERROR
                return AgentResult(answer=step.error, steps=steps, success=False)

            # Check if agent wants to finish
            if step.action is None or step.action == "finish":
                self.state = AgentState.DONE
                answer = step.thought
                self.memory.add_assistant(answer)
                return AgentResult(answer=answer, steps=steps, success=True)

            # Execute the tool
            self.state = AgentState.ACTING
            try:
                observation = str(self.tools.execute(step.action, **step.action_input))
                step.observation = observation
                self.memory.add_tool_result(observation, tool_name=step.action)
            except Exception as e:
                step.observation = f"Error: {e}"
                step.error = str(e)

            self.state = AgentState.THINKING

        # Max steps reached
        self.state = AgentState.ERROR
        return AgentResult(
            answer="Max steps reached without completing the task.",
            steps=steps,
            success=False,
        )

    def _plan(self, task: str, previous_steps: list[AgentStep]) -> AgentStep:
        """Use the planner to decide the next action."""
        context = {
            "task": task,
            "steps": previous_steps,
            "tools": self.tools.schemas(),
            "memory": self.memory.get_context(),
        }
        return self._planner(context)

    @staticmethod
    def _default_planner(context: dict) -> AgentStep:
        """Simple rule-based planner for testing.

        In production, replace this with an LLM call that returns
        structured tool-use decisions.
        """
        steps = context["steps"]
        tools = context["tools"]

        # If we have tool results, finish
        if steps and steps[-1].observation:
            return AgentStep(
                thought=f"I have the result: {steps[-1].observation}",
                action="finish",
            )

        # If tools available and no steps yet, use the first tool
        if tools and not steps:
            tool = tools[0]
            return AgentStep(
                thought=f"I'll use the {tool['name']} tool to help with this task.",
                action=tool["name"],
                action_input={"query": context["task"]} if "query" in str(tool.get("parameters", {})) else {},
            )

        # Default: finish with the task description
        return AgentStep(
            thought=f"Based on my analysis: {context['task']}",
            action="finish",
        )

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, tools={len(self.tools)}, "
            f"state={self.state.value})"
        )
