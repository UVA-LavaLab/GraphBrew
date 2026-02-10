#!/usr/bin/env python3
"""Thin wrapper — delegates to scripts.lib.eval_weights.main()."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.lib.eval_weights import main
main()
