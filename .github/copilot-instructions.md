Short, actionable guidance for AI coding agents working on the voicecred repo.

High-level project summary
- Minimal real-time credibility-scoring prototype. FastAPI ingress (`src/voicecred/main.py`) provides REST + WebSocket endpoints. The pipeline stages are acoustic → ASR (STT) → linguistic → assemble features → normalization → scoring.
- Key runtime pieces live in `src/voicecred/`: `acoustic.py` (feature extraction), `stt.py` (pluggable STT adapters), `linguistic.py` (NLP features), `assembler.py` (frame assembly), `scorer.py` (baseline-normalized scoring) and `session.py` (in-memory session store).

What to know first (priority facts)
- The app is single-process prototype; `InMemorySessionStore` holds session lifecycle (states: idle, calibrating, scoring, finalized). Tests rely on deterministic Mock adapters (`MockSTTAdapter`) and the in-memory store.
- STT adapter is chosen via env var `STT_ADAPTER` or the factory `create_stt_adapter` in `stt.py`. Mock adapters accept `transcript_override` in frames (useful for tests).
- Baseline computation uses median/MAD (see `utils/baseline.py`) — tests expect at least 3 calibration frames unless production policy used. Calibration is stateful and persisted in-session.
- Runtime flags: `.env` loads HF token, optional `VOICECRED_DEBUG` sets debug logging level. `MIN_ASR_CONF` gating is defined in `main.py` (default ~0.6).

Developer workflows you should recommend or use
- Start the server locally (dev): `uvicorn voicecred.main:app --reload --host 0.0.0.0 --port 8000` (matches `README.md` examples).
- Tests: run `pytest -q`. Note: some tests are heavy (pyannote/pytorch) and are normally skipped unless `RUN_HEAVY_PYANNOTE=true` is set in env.
- Windows/CMake helper: `nm.ps1` now contains safe helpers to inspect PATH and add `jom` to User/Machine PATH (avoid `setx` due to truncation/backfill issues). Use `vcvars64.bat` before CMake for MSVC toolchain.

Patterns and conventions to follow when changing code
- Prefer deterministic adapters/mocks for unit tests. If you add a new external adapter, ensure a mock/override path exists and tests do not rely on internet or GPU by default.
- Keep pipeline stage timeouts and gating logic stable: `STAGE_TIMEOUTS` and `MIN_ASR_CONF` live in `main.py` and are used in tests. Tweaks require updating tests that assert expected events (e.g., `asr_batch`, `linguistic_batch`, `feature_batch`).
- Feature frame schema: `assemble_feature_frame()` defines `feature_version`, `acoustic`, `acoustic_named`, `linguistic`, `derived` and `qc`. Tests expect `qc`, `derived`, `normalized`, `score`, and `explain` fields when scoring is available.

Key files to reference when making changes (examples)
- App ingress / pipeline orchestration: `src/voicecred/main.py` — WebSocket flow, batch processor, STAGE_TIMEOUTS, MIN_ASR_CONF
- Session lifecycle & baseline logic: `src/voicecred/session.py` — creation, calibration, baseline computation and rate limiting
- Acoustic extraction: `src/voicecred/acoustic.py` — parselmouth pitch, numpy fallback and QC metrics
- STT adapters: `src/voicecred/stt.py` — Mock, Whisper, Remote, adapter factory; Mock uses `transcript_override` for tests
- Linguistic features: `src/voicecred/linguistic.py` — spaCy fallback & speaking_rate inference
- Scoring & explainability: `src/voicecred/scorer.py` and `src/voicecred/utils/baseline.py`
- Tests: `tests/test_*.py` — follow these for expected events / assertions and use `conftest.py` for test environment notes

Quick guidelines for writing code-to-tests
- If adding a new observable event emitted over WebSocket, update tests in `tests/test_app.py` or add a focused integration test that connects via `TestClient.websocket_connect` and asserts the expected messages (use `recv_with_timeout` helper).
- When adding new per-frame features or derived metrics, update `assemble_feature_frame` and the baseline computation (session.compute_and_store_baseline) to ensure metrics are captured and that tests can detect them.

If you need clarification
- Ask which tests exercise any new public API, whether new components should have mock adapters, and whether the change must be compatible with the deterministic Mock behavior used by tests.

Keep changes small and test-focused — this repo opts for straightforward, synchronous logic suitable for unit/integration tests rather than heavy async deployments.
