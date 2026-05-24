"""Deep Research Agent — backward-compatible entry point.

Delegates to the modular implementation under agent/core/loop.py.
"""

from .core.loop import run_agent_loop

__all__ = ["run_agent_loop"]
