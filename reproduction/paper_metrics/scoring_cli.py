# SPDX-License-Identifier: Apache-2.0
"""python -m reproduction.paper_metrics.scoring_cli --help"""

import sys

from tower_eval.scoring import _paper_cli

if __name__ == "__main__":
    raise SystemExit(_paper_cli.main())
sys.modules[__name__] = _paper_cli
