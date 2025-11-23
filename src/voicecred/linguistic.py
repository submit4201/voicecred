from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import math
import logging
import time

logger = logging.getLogger(__name__)

try:
    import spacy
except Exception:
    spacy = None


@dataclass
class LinguisticResult:
    timestamp_ms: int
    linguistic: Dict[str, float]
    asr_quality: float


class LinguisticEngine:
    """Extract linguistic features using spaCy (or fallback simple tokenization).

    Accepts either a raw transcript string or a transcript structure from STT adapter
    (see `stt.MockSTTAdapter` output). The engine computes features per window such as:
    - pronoun_ratio (pronouns / tokens)
    - article_ratio (a/an/the / tokens)
    - ttr (type-token ratio)
    - avg_tokens_per_sentence
    - asr_conf (provided/inferred)

    If spaCy is not available, the engine will use a light-weight tokenizer.
    """

    def __init__(self, nlp: Any | None = None):
        if nlp is not None:
            self._nlp = nlp
        elif spacy is not None:
            try:
                # use a blank English pipeline for minimal dependencies
                self._nlp = spacy.blank("en")
                # ensure sentence boundaries are available for doc.sents in blank pipelines
                if "sentencizer" not in getattr(self._nlp, "pipe_names", []):
                    try:
                        self._nlp.add_pipe("sentencizer")
                    except Exception:
                        # older spaCy versions may require SentenceRecognizer name
                        try:
                            self._nlp.add_pipe("sentencizer", name="sentencizer")
                        except Exception:
                            # If adding sentencizer fails, leave nlp as-is and handle downstream
                            pass
            except Exception:
                self._nlp = None
        else:
            self._nlp = None

    def analyze(self, transcript_data: Dict[str, Any], timestamp_ms: int = 0) -> LinguisticResult:
        logger.debug("LinguisticEngine.analyze called: transcript_len=%s asr_conf=%s", len(str(transcript_data.get('raw') or '')), transcript_data.get('confidence'))
        start = time.time()
        """Analyze transcript_data (dict), returns LinguisticResult.

        transcript_data: {"words": [...], "confidence": float, "raw": str}
        """
        raw = transcript_data.get("raw") or " ".join([w.get("word", "") for w in transcript_data.get("words", [])])
        asr_conf = float(transcript_data.get("confidence", 0.0))

        tokens = []
        sentences = [raw]

        if self._nlp is not None:
            doc = self._nlp(raw)
            tokens = [t.text for t in doc if not t.is_space]
            sentences = [sent.text for sent in doc.sents] if doc.sents else [raw]
            pos = [t.pos_ for t in doc]
        else:
            # naive whitespace tokenizer
            tokens = raw.split()
            pos = ["NOUN" for _ in tokens]

        n_tokens = max(1, len(tokens))
        types = set(tokens)
        ttr = len(types) / n_tokens

        pronouns = sum(1 for t in tokens if t.lower() in {"i", "you", "he", "she", "they", "we", "me", "him", "her", "them"})
        articles = sum(1 for t in tokens if t.lower() in {"a", "an", "the"})

        avg_tokens_per_sentence = float(sum(len(s.split()) for s in sentences) / max(1, len(sentences)))

        # compute some additional surface-level metrics
        avg_word_len = float(sum(len(t) for t in tokens) / max(1, n_tokens))
        # simple lexical density: non-stop tokens / tokens (if spaCy provides is_stop)
        if self._nlp is not None:
            try:
                doc = self._nlp(raw)
                non_stop = sum(1 for t in doc if not getattr(t, "is_stop", False) and not t.is_space)
            except Exception:
                non_stop = n_tokens
        else:
            non_stop = n_tokens

        lexical_density = float(non_stop) / n_tokens

        # noun/verb approximations from POS tags (fallback to naive checks)
        noun_count = sum(1 for p in pos if p and p.startswith("N"))
        verb_count = sum(1 for p in pos if p and p.startswith("V"))
        noun_ratio = float(noun_count) / n_tokens
        verb_ratio = float(verb_count) / n_tokens

        features = {
            "pronoun_ratio": pronouns / n_tokens,
            "article_ratio": articles / n_tokens,
            "ttr": ttr,
            "avg_tokens_per_sentence": avg_tokens_per_sentence,
            # speaking_rate: tokens per second estimated from word timestamps when available
            # falls back to tokens/sec based on sentence token count if timing isn't available
            "tokens": int(n_tokens),
            "speaking_rate": None,
            "avg_word_length": avg_word_len,
            "lexical_density": lexical_density,
            "noun_ratio": noun_ratio,
            "verb_ratio": verb_ratio,
        }

        # attempt to infer speaking rate from timestamped word list (if provided)
        try:
            words = transcript_data.get("words") or []
            if words and isinstance(words, list):
                start_times = [w.get("start_ms") for w in words if isinstance(w, dict) and w.get("start_ms") is not None]
                end_times = [w.get("end_ms") for w in words if isinstance(w, dict) and w.get("end_ms") is not None]
                if start_times and end_times:
                    duration_ms = max(end_times) - min(start_times)
                    if duration_ms <= 0:
                        duration_ms = 1
                    features["speaking_rate"] = float(n_tokens) / (duration_ms / 1000.0)
        except Exception:
            # Don't fail the pipeline; speaking_rate remains None if we can't compute it
            pass

        dur_ms = (time.time() - start) * 1000
        logger.debug("LinguisticEngine.analyze finished in %.3fms tokens=%s sentences=%s", dur_ms, len(tokens), len(sentences))
        return LinguisticResult(timestamp_ms=timestamp_ms, linguistic=features, asr_quality=asr_conf)
