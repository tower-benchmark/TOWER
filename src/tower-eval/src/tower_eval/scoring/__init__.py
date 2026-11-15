# SPDX-License-Identifier: Apache-2.0
"""Scoring APIs and compatibility exports for online geometry helpers."""

from .geometry import angle as angle
from .geometry import area as area
from .geometry import geometry as geometry
from .geometry import inside as inside
from .geometry import intersection as intersection
from .geometry import polygon as polygon
from .metrics import BenchmarkMetrics as BenchmarkMetrics
from .metrics import EpisodeRecord as EpisodeRecord
from .online import StackScorer as StackScorer

__all__ = ["BenchmarkMetrics", "EpisodeRecord", "StackScorer"]

from .api import score_recording as score_recording
from .api import score_run as score_run
