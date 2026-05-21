import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` instead of ZeroDivisionError."""
    return numerator / denominator if denominator != 0 else default


def format_pct(count: int, total: int) -> str:
    """Returns a formatted percentage string, e.g. '23.4%'."""
    return f"{safe_divide(count * 100, total):.1f}%"


def ensure_dir(path: Path | str) -> Path:
    """Creates a directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def flatten_dict(
    d: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """
    Flattens a nested dict into a single-level dict with dotted keys.

    Example:
        {"a": {"b": 1, "c": 2}} → {"a.b": 1, "a.c": 2}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamps a value to [lo, hi]."""
    return max(lo, min(hi, value))


@contextmanager
def timer(label: str = "operation") -> Generator[None, None, None]:
    """Context manager that logs elapsed time for any block of code."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("[timer] %s: %.1fms", label, elapsed_ms)
