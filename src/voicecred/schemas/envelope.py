from __future__ import annotations

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.voicecred.utils.logger_util import get_logger, logging
logger=get_logger(__name__,logging.DEBUG)
class AcousticNamed(BaseModel):
    f0_mean: Optional[float] = Field(None)
    f0_median: Optional[float] = Field(None)
    f0_std: Optional[float] = Field(None)
    rms: float = Field(..., description="Root mean square energy")
    zcr: float = Field(..., description="Zero crossing rate")


class QC(BaseModel):
    snr_db: Optional[float] = Field(None)
    speech_ratio: float = Field(..., ge=0.0, le=1.0)
    voiced_seconds: float = Field(..., ge=0.0)
    words_in_window: Optional[int] = Field(None, ge=0)

    @field_validator("snr_db")
    def _snr_reasonable(cls, v: Optional[float]):
        # Allow None but if provided roughly bound to realistic SNR values
        if v is None:
            return v
        if v < -100.0 or v > 100.0:
            raise ValueError("snr_db out of realistic range")
        return v


class STTWord(BaseModel):
    word: str
    start_ms: int
    end_ms: int
    confidence: float = Field(..., ge=0.0, le=1.0)


class STTResult(BaseModel):
    raw: Optional[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    words: Optional[List[STTWord]] = None
    is_final: Optional[bool] = False


class SpeakerSegment(BaseModel):
    speaker: str
    start_ms: int
    end_ms: int


class LinguisticFeatures(BaseModel):
    # keep this flexible but typed so downstream validators can refer to keys
    pronoun_ratio: Optional[float]
    article_ratio: Optional[float]
    ttr: Optional[float]
    avg_tokens_per_sentence: Optional[float]
    tokens: Optional[int]
    speaking_rate: Optional[float]


class ScoringPayload(BaseModel):
    score: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    # Optional breakdown helpful for UI and diagnostics
    contributions: Optional[Dict[str, float]] = None
    explain: Optional[Dict[str, Any]] = None


class EnvelopeV1(BaseModel):
    """Authoritative envelope for a windowed frame or event payload.

    This model is intentionally conservative: prefer named acoustic metrics
    (acoustic_named) over raw acoustic vector lists. QC is required for safety.
    """

    version: Literal["v1"] = "v1"
    session_id: str
    window_id: Optional[str]
    timestamp_ms: int
    # prefer a named acoustic mapping for robustness
    acoustic_named: Optional[AcousticNamed] = None
    # legacy raw vector is explicitly disallowed in strict contract usage
    # acoustic: Optional[List[float]] = None
    qc: QC
    stt: Optional[STTResult] = None
    speaker: Optional[List[SpeakerSegment]] = None
    linguistic: Optional[LinguisticFeatures] = None
    scoring: Optional[ScoringPayload] = None
    meta: Optional[Dict[str, object]] = None

    @field_validator("acoustic_named", mode="before")
    def require_named_acoustic(cls, v):
        # If a caller provides a raw vector, require it to be converted upstream
        # into the named mapping to keep contract stable.
        if v is None:
            return v
        # pydantic will validate structure; accept dicts that can be parsed
        return v

    @field_validator("qc")
    def qc_presence(cls, v: QC):
        if v is None:
            raise ValueError("qc block is required for EnvelopeV1")
        # speech_ratio already bounded; further sanity checks can be added here
        if v.speech_ratio is None:
            raise ValueError("qc.speech_ratio required")
        return v
