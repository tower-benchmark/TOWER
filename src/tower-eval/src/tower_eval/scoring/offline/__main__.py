# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for paper workbook scoring."""

import sys

from tower_eval.scoring import _paper_cli as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
