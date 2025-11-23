from __future__ import annotations

from typing import Dict, Any


def assemble_feature_frame(session_id: str, acoustic: Dict[str, Any], linguistic: Dict[str, Any] | None, timestamp_ms: int) -> Dict[str, Any]:
    """Combine acoustic and linguistic results into a unified feature frame dict.

    Keeps schema compatible with feature_version v1.
    """
    feature = {
        "feature_version": 1,
        "session_id": session_id,
        "timestamp_ms": timestamp_ms,
        "acoustic": acoustic.get("acoustic") if isinstance(acoustic, dict) else acoustic,
        # a named mapping of the acoustic vector for deterministic keys (f0_mean,f0_median,f0_std,rms,zcr)
        "acoustic_named": None,
        # include linguistic features when available
        "linguistic": linguistic if linguistic is not None else None,
        "derived": [],
        # ensure qc includes acoustic QC and ASR metadata where available
        "qc": {}
    }
    # populate QC from acoustic and ASR/linguistic if available
    if isinstance(acoustic, dict):
        feature["qc"].update(acoustic.get("qc", {}))
        # if acoustic is a dict containing an 'acoustic' list vector, provide named mapping too
        if isinstance(acoustic.get("acoustic"), list):
            names = ["f0_mean", "f0_median", "f0_std", "rms", "zcr"]
            named = {names[i]: acoustic.get("acoustic")[i] if i < len(acoustic.get("acoustic")) else None for i in range(len(names))}
            feature["acoustic_named"] = named
    elif isinstance(acoustic, list):
        # provide a named mapping to make downstream baselines & scoring deterministic
        names = ["f0_mean", "f0_median", "f0_std", "rms", "zcr"]
        named = {names[i]: acoustic[i] if i < len(acoustic) else None for i in range(len(names))}
        feature["acoustic_named"] = named
    # Add a small derived example
    if isinstance(linguistic, dict):
        # include asr_quality in QC (if provided by linguistic analysis)
        if "asr_quality" in linguistic:
            feature["qc"]["asr_conf"] = float(linguistic.get("asr_quality"))
        # copy a few convenient linguistic keys into derived (for easy downstream inspection)
        for k in ("pronoun_ratio", "article_ratio", "ttr", "avg_tokens_per_sentence", "avg_word_length", "lexical_density", "noun_ratio", "verb_ratio"):
            if k in linguistic:
                feature["derived"].append({k: linguistic[k]})
        # annotate word counts if present
        if "tokens" in linguistic:
            feature["qc"]["words_in_window"] = int(linguistic.get("tokens", 0))
        # also give a brief combined measure if present
        if isinstance(linguistic.get("asr_quality"), (int, float)):
            feature["derived"].append({"combined_asr_quality": float(linguistic.get("asr_quality"))})
        # include speaking_rate when available from the linguistic analysis (tokens/sec)
        if isinstance(linguistic.get("speaking_rate"), (int, float)):
            feature["derived"].append({"speaking_rate": float(linguistic.get("speaking_rate"))})
    # synthesize a pause_ratio derived feature from acoustic QC when available
    if isinstance(feature.get("qc"), dict):
        try:
            speech_ratio = feature["qc"].get("speech_ratio")
            if isinstance(speech_ratio, (int, float)):
                pause_ratio = 1.0 - float(speech_ratio)
                feature["derived"].append({"pause_ratio": float(pause_ratio)})
        except Exception:
            # best effort — don't fail if qc contains unexpected types
            pass
    return feature
