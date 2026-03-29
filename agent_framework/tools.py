"""Tool registration and execution system.

Tools are the atomic capabilities an agent can invoke. Each tool is a
callable with a name, description, and typed parameter schema — enough
for an LLM to decide when and how to call it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolParam:
    """Schema for a single tool parameter."""

    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    """A registered tool that an agent can invoke."""

    name: str
    description: str
    fn: Callable[..., Any]
    params: list[ToolParam] = field(default_factory=list)

    def execute(self, **kwargs: Any) -> Any:
        """Run the tool with the given arguments."""
        return self.fn(**kwargs)

    def to_schema(self) -> dict[str, Any]:
        """Export as a JSON-serializable schema (LLM-friendly)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                p.name: {
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.params
            },
        }


class ToolRegistry:
    """Registry for managing available tools.

    Usage:
        registry = ToolRegistry()

        @registry.register("calculator", "Perform arithmetic")
        def calc(expression: str) -> str:
            return str(eval(expression))

        result = registry.execute("calculator", expression="2 + 2")
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str = "",
        params: list[ToolParam] | None = None,
    ) -> Callable:
        """Decorator to register a function as a tool."""

        def decorator(fn: Callable) -> Callable:
            tool_params = params or self._infer_params(fn)
            desc = description or fn.__doc__ or ""
            self._tools[name] = Tool(
                name=name,
                description=desc.strip(),
                fn=fn,
                params=tool_params,
            )
            return fn

        return decorator

    def add(self, name: str, fn: Callable, description: str = "", params: list[ToolParam] | None = None) -> None:
        """Programmatically add a tool."""
        tool_params = params or self._infer_params(fn)
        desc = description or fn.__doc__ or ""
        self._tools[name] = Tool(name=name, description=desc.strip(), fn=fn, params=tool_params)

    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a tool by name."""
        return self.get(name).execute(**kwargs)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        """Export all tool schemas for LLM consumption."""
        return [t.to_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @staticmethod
    def _infer_params(fn: Callable) -> list[ToolParam]:
        """Infer parameter schema from function signature."""
        sig = inspect.signature(fn)
        params = []
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            ptype = "string"
            if param.annotation != inspect.Parameter.empty:
                type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
                ptype = type_map.get(param.annotation, "string")
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            params.append(ToolParam(name=pname, type=ptype, required=required, default=default))
        return params
