"""Entry point for running the package as a module (python -m deterministic_inference)."""

import sys
from deterministic_inference.cli import main

if __name__ == "__main__":
    sys.exit(main())
