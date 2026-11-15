# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

# test_eef_delivery.py reuses tower-eval's test fakes; keep it collectable alone.
_EVAL_TESTS = Path(__file__).resolve().parents[2] / "src" / "tower-eval" / "tests"
if str(_EVAL_TESTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_TESTS))
