"""
Prompt template helpers.

Templates use Python's ``str.format`` syntax so they remain readable and
behave predictably (no Jinja installation required). Each template is
declared once and rendered many times — never concatenate strings in code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """A named, parameterised prompt template."""

    name: str
    template: str

    def render(self, **kwargs) -> str:
        """Render the template, raising ``KeyError`` for any unmet placeholder."""
        try:
            return self.template.format(**kwargs)
        except KeyError as exc:
            raise KeyError(
                f"PromptTemplate '{self.name}' missing required value: {exc.args[0]!r}"
            ) from exc

    def required_keys(self) -> set[str]:
        """Best-effort extraction of {placeholders} declared in the template."""
        import string
        return {
            field for _, field, _, _ in string.Formatter().parse(self.template)
            if field
        }