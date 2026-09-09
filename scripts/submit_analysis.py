#!/usr/bin/env python3
"""Submit checkpoint shards and collect their results on the login node."""

import sys

from tiny_llm.cli import main

if __name__ == "__main__":
    main(["submit-analysis", *sys.argv[1:]])
