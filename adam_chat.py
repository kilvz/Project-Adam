#!/usr/bin/env python3
"""Backward-compat shim — delegates to the project_adam package."""
import sys
sys.path.insert(0, "src")
from project_adam.__main__ import main

if __name__ == "__main__":
    main()
