from __future__ import annotations

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from src.voicecred.utils.logger_util import get_logger, logging
logger=get_logger(__name__,logging.DEBUG)
from .envelope import EnvelopeV1, STTResult, SpeakerSegment, ScoringPayload


EventType = Literal[
    "FrameReceived",
    "AcousticFeaturesReady",
    "STTPartial",
    "STTFinal",
    "SpeakerSegmentsReady",
    "LinguisticFeaturesReady",
    "FeatureFrameAssembled",
    "BaselineReady",
    "ScoreUpdated",
]


class BaseEvent(BaseModel):
    type: EventType
    # the envelope is optional for some events (e.g. acoustic-only), but present
    # for assembled/score events.
    envelope: Optional[EnvelopeV1] = None


class FrameReceived(BaseEvent):
    type: Literal["FrameReceived"] = "FrameReceived"
    # raw frame payload may contain a transcript_override (tests) and timestamp
    pcm_length: int


class AcousticFeaturesReady(BaseEvent):
    type: Literal["AcousticFeaturesReady"] = "AcousticFeaturesReady"
    envelope: EnvelopeV1


class STTPartial(BaseEvent):
    type: Literal["STTPartial"] = "STTPartial"
    envelope: EnvelopeV1
    partial: STTResult


class STTFinal(BaseEvent):
    type: Literal["STTFinal"] = "STTFinal"
    envelope: EnvelopeV1
    final: STTResult


class SpeakerSegmentsReady(BaseEvent):
    type: Literal["SpeakerSegmentsReady"] = "SpeakerSegmentsReady"
    envelope: EnvelopeV1
    segments: List[SpeakerSegment]


class LinguisticFeaturesReady(BaseEvent):
    type: Literal["LinguisticFeaturesReady"] = "LinguisticFeaturesReady"
    envelope: EnvelopeV1


class FeatureFrameAssembled(BaseEvent):
    type: Literal["FeatureFrameAssembled"] = "FeatureFrameAssembled"
    envelope: EnvelopeV1


class BaselineReady(BaseEvent):
    type: Literal["BaselineReady"] = "BaselineReady"
    envelope: EnvelopeV1
    baseline_id: str


class ScoreUpdated(BaseEvent):
    type: Literal["ScoreUpdated"] = "ScoreUpdated"
    envelope: EnvelopeV1
    scoring: ScoringPayload
