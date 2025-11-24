from __future__ import annotations

from enum import Enum
from typing import Set


class SessionStates(str, Enum):
    INITIALIZING = "initializing"
    CALIBRATING = "calibrating"
    ACTIVE = "active"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"


ALLOWED_TRANSITIONS = {
    SessionStates.INITIALIZING: {SessionStates.CALIBRATING, SessionStates.FINALIZED},
    SessionStates.CALIBRATING: {SessionStates.ACTIVE, SessionStates.FINALIZED},
    SessionStates.ACTIVE: {SessionStates.FINALIZING, SessionStates.FINALIZED},
    SessionStates.FINALIZING: {SessionStates.FINALIZED},
    SessionStates.FINALIZED: set(),
}


def can_transition(from_state: SessionStates | str, to_state: SessionStates | str) -> bool:
    """Return True if a transition from from_state -> to_state is allowed."""
    f = SessionStates(from_state) if not isinstance(from_state, SessionStates) else from_state
    t = SessionStates(to_state) if not isinstance(to_state, SessionStates) else to_state
    return t in ALLOWED_TRANSITIONS.get(f, set())
