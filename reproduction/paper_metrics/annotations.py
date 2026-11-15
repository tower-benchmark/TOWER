# SPDX-License-Identifier: Apache-2.0
"""Paper workbook reader; implemented in tower_eval so installed wheels need no repo root."""

import sys

from tower_eval.scoring import _paper_annotations

sys.modules[__name__] = _paper_annotations
