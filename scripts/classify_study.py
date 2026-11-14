#!/usr/bin/env python3

import sys
from pathlib import Path

try:
    from dcm_classifier.classify_study import main
except Exception as e:
    print(f"Missing module import {e}")
    print(
        f"Try setting export PYTHONPATH={Path(__file__).parent.parent.as_posix()}/src"
    )
    sys.exit(255)


if __name__ == "__main__":
    main()
