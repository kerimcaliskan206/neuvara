"""
Tool registry — placeholder for future function-calling integrations.

When the AI needs to actually run code (look up a patient record, re-run
inference, query a knowledge base), tools are registered here and resolved
by name. For now the registry is a stub so the chat service can call
``list_tools()`` without crashing and downstream code can be developed
against the final shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol


class Tool(Protocol):
    name: str
    description: str

    async def run(self, **kwargs) -> dict:
        ...


@dataclass
class RegisteredTool:
    name: str
    description: str
    handler: Callable[..., Awaitable[dict]]
    parameters_schema: dict


class ToolRegistry:
    """In-memory tool registry. Wire real tools here in a future phase."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        handler: Callable[..., Awaitable[dict]],
        parameters_schema: dict | None = None,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            handler=handler,
            parameters_schema=parameters_schema or {},
        )

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema,
            }
            for t in self._tools.values()
        ]

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)