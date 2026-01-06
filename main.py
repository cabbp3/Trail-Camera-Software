#!/usr/bin/env python3
"""
Trail Cam Organizer - Main Entry Point

Launches the full-featured Trail Cam app.
"""

import sys
import os

# Ensure we can find our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.label_tool import main

if __name__ == "__main__":
    main()
