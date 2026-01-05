#!/usr/bin/env python3
"""
Trail Cam Organizer - Entry Point

Simple photo browser for end users.
Run this for the product/consumer app.
"""

import sys
import os

# Ensure we can find our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from organizer_ui import main

if __name__ == "__main__":
    main()
