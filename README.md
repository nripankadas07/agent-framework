# agent-framework

A lightweight agent orchestration library with tool registration, conversation memory, and ReAct-style planning loops. No heavyweight dependencies — just the core abstractions you need to build LLM-powered agents.

## Why This Exists

Most agent frameworks pull in dozens of dependencies and enforce rigid patterns. This library provides the essential building blocks — tools, memory, and a planning loop — so you can build agents that fit your architecture instead of the other way around.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from agent_framework.agent import Agent
from agent_framework.tools import ToolRegistry

# Register tools
tools = ToolRegistry()

@tools.register("search", "Search for information")
def search(query: str) -> str:
    return f"Results for: {query}"

@tools.register("calculator", "Do math")
def calculator(expression: str) -> str:
    return str(eval(expression))

# Create and run an agent
agent = Agent(name="assistant", tools=tools)
result = agent.run("Find the population of France")

print(result.answer)
print(f"Completed in {result.num_steps} steps")
```

## Components

### Tool Registry

Type-safe tool registration with automatic parameter inference:

```python
from agent_framework.tools import ToolRegistry, ToolParam

registry = ToolRegistry()

# Decorator style
@registry.register("greet", "Greet someone by name")
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Programmatic style
registry.add("double", lambda x: x * 2, "Double a number")

# Export schemas for LLM consumption
schemas = registry.schemas()
```

### Memory

Conversation memory with sliding-window management:

```python
from agent_framework.memory import ConversationMemory, SummaryMemory

# Sliding window — keeps recent messages, preserves system prompt
memory = ConversationMemory(max_messages=50)
memory.add_system("You are a helpful assistant.")
memory.add_user("Hello!")
memory.add_assistant("Hi there!")

# Summary memory — compresses old messages into summaries
summary_mem = SummaryMemory(summarize_threshold=20)
```

### Agent Patterns

Pre-built patterns for common agent architectures:

```python
from agent_framework.patterns import create_tool_agent, AgentRouter

# Quick tool agent
agent = create_tool_agent("math", tools={"calc": lambda expr: eval(expr)})

# Multi-agent routing
router = AgentRouter()
router.register("math", math_agent, keywords=["calculate", "compute"])
router.register("search", search_agent, keywords=["find", "look up"])
result = router.route("Calculate the area of a circle with radius 5")
```

## Project Structure

```
agent-framework/
├── agent_framework/
│   ├── __init__.py
│   ├── tools.py            # Tool registration and execution
│   ├── memory.py           # Conversation and summary memory
│   ├── agent.py            # Core ReAct agent loop
│   └── patterns.py         # Common agent patterns
├── tests/
│   ├── test_tools.py
│   ├── test_memory.py
│   ├── test_agent.py
│   └── test_patterns.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
