#!/usr/bin/env python3
"""
Main launcher for Nova Voice Assistant.

Simple entry point that starts the assistant pipeline.
"""

from assistant import main
import asyncio
import sys

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 