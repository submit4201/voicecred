from __future__ import annotations

"""Scoring module wrapper.

The repo historically kept `Scorer` at `src/voicecred/scorer.py`.
To align with Phase 5 layout we re-export the existing implementation here
so callers can import from `src.voicecred.scoring.scorer`.
"""

from src.voicecred.scorer import Scorer, ScoreResult

__all__ = ["Scorer", "ScoreResult"]
