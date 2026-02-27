"""Analysis, metrics, and visualization."""

# Re-export adaptive analysis functions at package level so that
# ``from scripts.lib.analysis import analyze_adaptive_order`` keeps working.
from .adaptive import *  # noqa: F401,F403
