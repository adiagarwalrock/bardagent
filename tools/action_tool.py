from typing import List

from langchain_community.tools import ShellTool
import sympy as sp
from langchain_core.tools import tool


def shell_tool() -> List:
    """Return the Shell Tool."""
    return [ShellTool()]


@tool(
    "math",
    return_direct=True,
    description="""
    A tool for evaluating arithmetic and mid-level math expressions (fractions, exponents, roots, trig).
    Input should be a valid mathematical expression.
    Examples of valid expressions: "2 + 2", "sin(pi/4) + cos(pi/4)", "sqrt(16) + 3^2"
    """,
)
def math_evaluator(expression: str) -> str:
    """Evaluate a math expression and return the result as a string."""
    try:
        return str(sp.simplify(sp.sympify(expression)))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not evaluate expression: {exc}") from exc


def math_tool() -> List:
    """Return the Math Tool."""
    return [math_evaluator]
