"""Tool definitions for MathNexus agents.

Tools provided:
1. Python REPL — execute Python code for numerical verification
2. SymPy Math — symbolic computation (solve, simplify, calculus)
3. Web Search — search for math theorems and concepts via DuckDuckGo
4. Calculator — evaluate arithmetic expressions safely
"""

import io
import re
import sys
import ast
import math
import contextlib
from typing import Optional, Any

from langchain_core.tools import tool
from sympy import (
    symbols, solve, simplify, expand, factor, diff, integrate,
    limit, series, Matrix, Symbol, Eq, sympify, latex, pi, E, oo,
)

# ── Tool 1: Python REPL ──────────────────────────────────────────────────────


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output.

    Use this tool to run Python code for numerical computations, verification,
    or any calculation that requires programming. The code can use standard
    libraries (math, numpy, etc.). Use print() to display results.

    Args:
        code: Python code string to execute.
    """
    # Capture stdout
    stdout = io.StringIO()
    stderr = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, {"__builtins__": __builtins__, "math": math})

        output = stdout.getvalue()
        errors = stderr.getvalue()

        if errors:
            return f"[stderr]:\n{errors}\n[stdout]:\n{output}"
        return output if output else "(code executed successfully with no output)"
    except Exception as e:
        return f"Error executing code: {type(e).__name__}: {e}"


# ── Tool 2: SymPy Symbolic Math ───────────────────────────────────────────────


@tool
def sympy_math(expression: str) -> str:
    """Perform symbolic mathematical computation using SymPy.

    Supports:
    - Solving equations: solve(x**2 - 4, x) → [-2, 2]
    - Simplifying: simplify((x**2 - 1)/(x - 1)) → x + 1
    - Expanding: expand((x+1)**3)
    - Factoring: factor(x**2 - 4)
    - Differentiation: diff(x**3, x)
    - Integration: integrate(x**2, x)
    - Limits: limit(sin(x)/x, x, 0)
    - Matrix operations: Matrix([[1,2],[3,4]]).inv()
    - Series expansion: series(sin(x), x, 0, 5)

    Args:
        expression: A SymPy expression to evaluate. Use 'x', 'y', 'z' as symbols.
                    For multi-symbol expressions, declare them explicitly.
    """
    try:
        # Create common symbols
        x, y, z, n, t, a, b, c, d = symbols("x y z n t a b c d")

        # Parse the expression
        local_ns = {
            "x": x, "y": y, "z": z, "n": n, "t": t, "a": a, "b": b, "c": c, "d": d,
            "solve": solve, "simplify": simplify, "expand": expand, "factor": factor,
            "diff": diff, "integrate": integrate, "limit": limit, "series": series,
            "Matrix": Matrix, "Symbol": Symbol, "Eq": Eq, "sympify": sympify,
            "pi": pi, "E": E, "oo": oo, "latex": latex,
        }

        # Check if it's a plain expression or a function call
        cleaned = expression.strip()
        result = eval(cleaned, {"__builtins__": {}}, local_ns)

        # Format the result nicely
        if hasattr(result, "__str__"):
            return str(result)
        return repr(result)

    except Exception as e:
        return f"SymPy error: {type(e).__name__}: {e}"


# ── Tool 3: Web Search ────────────────────────────────────────────────────────


@tool
def web_search(query: str) -> str:
    """Search the web for mathematical concepts, theorems, formulas, or references.

    Use this to look up mathematical theorems, definitions, or approaches
    that may help solve the problem.

    Args:
        query: Search query string.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No search results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "No description")
            formatted.append(f"{i}. {title}\n   {body}\n")

        return "\n".join(formatted)

    except ImportError:
        return "Web search unavailable: duckduckgo-search not installed."
    except Exception as e:
        return f"Web search error: {type(e).__name__}: {e}"


# ── Tool 4: Calculator ────────────────────────────────────────────────────────


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical arithmetic expression.

    Supports: +, -, *, /, ** (power), % (modulo), // (floor division),
    abs(), round(), order of operations with parentheses.

    Args:
        expression: A mathematical expression to evaluate (e.g., "3.14 * 2 ** 5").
    """
    # Only allow safe operations
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "int": int, "float": float, "pow": pow, "sqrt": math.sqrt,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "log2": math.log2,
        "exp": math.exp, "pi": math.pi, "e": math.e,
        "factorial": math.factorial, "gcd": math.gcd,
    }

    try:
        tree = ast.parse(expression.strip(), mode="eval")

        # Safety check: only allow simple operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                return f"Error: '{node.id}' is not allowed in calculator."
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                return "Error: only simple function calls are allowed."

        result = eval(
            compile(tree, "<calculator>", "eval"),
            {"__builtins__": {}},
            allowed_names,
        )
        return str(result)

    except SyntaxError as e:
        return f"Syntax error: {e}"
    except Exception as e:
        return f"Calculator error: {type(e).__name__}: {e}"


# ── Tool Registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [python_repl, sympy_math, web_search, calculator]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
