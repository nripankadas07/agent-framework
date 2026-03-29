"""Tests for tool registration and execution."""

import pytest

from agent_framework.tools import Tool, ToolParam, ToolRegistry


class TestTool:
    def test_execute(self):
        tool = Tool(name="add", description="Add two numbers", fn=lambda a, b: a + b)
        assert tool.execute(a=2, b=3) == 5

    def test_schema(self):
        tool = Tool(
            name="search",
            description="Search the web",
            fn=lambda q: q,
            params=[ToolParam(name="q", type="string", description="Query")],
        )
        schema = tool.to_schema()
        assert schema["name"] == "search"
        assert "q" in schema["parameters"]


class TestToolRegistry:
    def test_decorator_registration(self):
        registry = ToolRegistry()

        @registry.register("greet", "Say hello")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert "greet" in registry
        assert registry.execute("greet", name="World") == "Hello, World!"

    def test_add_method(self):
        registry = ToolRegistry()
        registry.add("double", lambda x: x * 2, "Double a number")
        assert registry.execute("double", x=5) == 10

    def test_unknown_tool(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.add("a", lambda: None)
        registry.add("b", lambda: None)
        assert len(registry.list_tools()) == 2

    def test_schemas(self):
        registry = ToolRegistry()
        registry.add("test", lambda: "ok", "A test tool")
        schemas = registry.schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "test"

    def test_param_inference(self):
        registry = ToolRegistry()
        registry.add("typed", lambda x: x)
        tool = registry.get("typed")
        assert len(tool.params) == 1
        assert tool.params[0].name == "x"

    def test_len(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.add("a", lambda: None)
        assert len(registry) == 1
