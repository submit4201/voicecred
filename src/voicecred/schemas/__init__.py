"""Schemas package for contract-first models (Envelope/Event definitions).

These models are intentionally strict — they enforce required fields and safe
defaults for the protocol used by the ingress/event-bus/assembler pipeline.
"""

__all__ = ["envelope", "events"]
