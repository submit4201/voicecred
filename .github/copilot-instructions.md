# Copilot / AI helper guidance for contributors

Quick, focused instructions so an AI coding assistant can be productive right away.

1. Project overview
   - Purpose: a lightweight realtime "credibility scoring" prototype (ingress + in-memory pipeline). See `README.md` for developer notes.
   - Key entry points: `src/voicecred/main.py` (FastAPI app), `src/voicecred/session.py` (InMemorySessionStore), `src/voicecred/acoustic.py`, `src/voicecred/stt.py`, `src/voicecred/linguistic.py`, `src/voicecred/assembler.py`, `src/voicecred/scorer.py`.

2. How the pipeline flows (high-level)
   - Clients POST `/sessions/start` -> receive `session_id` and short-lived `token`.
   - Client connects to WebSocket `/ws/{session_id}?token={token}` and sends PCM frames as messages (see `scripts/client_harness.py` and `tests/test_app.py`).
   - Background batch processor: acoustic feature extraction -> STT -> linguistic -> assemble features -> baseline normalization -> scoring. This flow is implemented and orchestrated in `main.py`.

3. Important runtime/config knobs
   - STT adapter selection: env var `STT_ADAPTER` (e.g. `mock` or `whisper`) via `create_stt_adapter()` in `stt.py` (default: MockSTTAdapter). Tests rely on Mock adapter behavior.
   - ASR gating: `MIN_ASR_CONF` in `main.py` controls whether linguistic analysis runs for a batch (default 0.6).
   - Debugging: set `VOICECRED_DEBUG=1` for DEBUG level logs.
   - External integrations are optional: `parselmouth`, `numpy`, `spacy`, `pyannote.audio` are used when installed — code has fallbacks for tests.
   - Hugging Face token: `HF_API_KEY` or `HUGGINGFACE_HUB_TOKEN` are used by the pyannote adapter.

4. Baseline & calibration behavior (tests and production differences)
   - `InMemorySessionStore.compute_and_store_baseline()` requires at least `min_frames` (default 3). QC gating applies (speech_ratio, voiced_seconds, SNR) to accept frames for baseline.
   - There's a stricter production mode (`use_production_policy=True`) which requires more frames and/or longer voiced audio (see `session.py`). Tests use smaller thresholds.

5. Testing & CI notes
   - Run tests: `pytest -q` (see `pytest.ini` for logs).
   - Some tests load heavy models (pyannote) and are skipped by default; enable them with `RUN_HEAVY_PYANNOTE=true`. CI workflow uses the same flag: see `.github/workflows/python-tests.yml`.
   - FastAPI tests use `fastapi.TestClient` and `ws.receive_json()` with a small helper `recv_with_timeout()` pattern in `tests/test_app.py` — copy that for new WebSocket tests.

6. Common project-specific patterns to follow
   - Use mock adapters for deterministic tests (e.g. `MockSTTAdapter.transcribe()` supports `transcript_override` on frames).
   - Feature frames are assembled using `assemble_feature_frame()` (fields: `feature_version`, `acoustic`, `acoustic_named`, `linguistic`, `derived`, `qc`, `normalized`, `score`, `explain`). Tests assert presence of these fields.
   - Scoring uses `Scorer` with `DEFAULT_WEIGHTS`. Metric lookup handles `acoustic_named` and `derived` list entries (see `_lookup_metric()` in `scorer.py`). Use that when adding new rules.
   - Session lifecycle: `idle` → `calibrating` → `scoring` → `finalized`. Use `store.set_state()` and `store.finalize()` when manipulating sessions.

7. Concrete examples an AI assistant can suggest or generate
   - Example to start dev server: `conda activate C:/workspace/voicecred/.conda; uvicorn voicecred.main:app --reload --host 0.0.0.0 --port 8000` (repo README). Add `STT_ADAPTER` override if needed.
   - Example test skeleton for WebSocket flows: follow `tests/test_app.py` structure (use `TestClient`, `client.websocket_connect()`, `recv_with_timeout()` helper and `transcript_override` frames to force deterministic ASR).
   - When proposing changes in inference adapters, preserve the `create_stt_adapter()` factory shape and ensure tests remain deterministic by falling back to `MockSTTAdapter`.

8. Where to look for more context
   - CI: `.github/workflows/python-tests.yml`
   - Test examples: `tests/test_app.py`, `tests/test_stt_and_linguistic.py` (for integration patterns)
   - Feature assembly/scoring: `src/voicecred/assembler.py`, `src/voicecred/scorer.py`

Note: keep fixes small and obvious. Update unit tests (preferred) and match the repository's test-first patterns when modifying behavior.

If you'd like, I can iterate on wording or include a short example PR checklist tailored to the repo's CI and heavy-test gating.