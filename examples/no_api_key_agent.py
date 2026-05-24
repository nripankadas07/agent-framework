"""No-API-key demo: an inspectable tool-using agent with a trace."""

from __future__ import annotations

import ast
import operator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_framework.agent import Agent, AgentStep
from agent_framework.tools import ToolRegistry


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_arithmetic(expr: str) -> str:
    """Evaluate tiny arithmetic expressions without eval()."""
    tree = ast.parse(expr, mode="eval")

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError(f"unsupported expression: {expr!r}")

    value = walk(tree)
    return str(int(value) if value.is_integer() else value)


def planner(context: dict) -> AgentStep:
    if not context["steps"]:
        return AgentStep(
            thought="I need exact arithmetic, so I will call the calculator tool.",
            action="calculate",
            action_input={"expression": "12 * 7 + 5"},
        )
    observation = context["steps"][-1].observation
    return AgentStep(thought=f"The checked answer is {observation}.", action="finish")


def main() -> None:
    tools = ToolRegistry()

    @tools.register("calculate", "Safely evaluate a small arithmetic expression")
    def calculate(expression: str) -> str:
        return safe_arithmetic(expression)

    agent = Agent(name="demo-agent", tools=tools, planner=planner, max_steps=4)
    result = agent.run("What is 12 * 7 + 5?")
    print(result.answer)
    print("\nTrace:")
    for i, step in enumerate(result.steps, 1):
        print(f"{i}. thought={step.thought!r} action={step.action!r} observation={step.observation!r}")


if __name__ == "__main__":
    main()
