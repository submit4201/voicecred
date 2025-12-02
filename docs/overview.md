<!-- User comments removed: Use the document to communicate changes; any inline markers were removed to keep docs clean. -->

# VoiceCred — Detailed Project Overview

This overview documents the high-level design, runtime architecture, developer workflows, testing patterns and relevant implementation notes for the VoiceCred repository. It's intended to help maintainers and AI coding agents move quickly and make safe changes.

---

## Table of Contents
- Project Goals & High-Level Architecture
- Pipeline & Data Flow (Detailed)
- Key Components & Files
- Runtime Configuration & Environment
- Developer Workflows (Run, Build, Tests)
- Data Models & Feature Schema
- Baseline, Normalization & Scoring
- Extension Points & Conventions for Contributions
- Testing Patterns & Examples
- Troubleshooting & Known Caveats
- Quick Commands & Shortcuts
- FAQ & Notes for Future Work

---

## Project Goals & High-Level Architecture
VoiceCred is a lightweight, single-process prototype that demonstrates a realtime credibility scoring pipeline. It exposes REST and WebSocket ingress endpoints via FastAPI and processes audio frames through an acoustic stage, ASR (STT), linguistic analysis, feature assembly, baseline normalization, and scoring.

The design choices prioritize testability (deterministic mocks), a minimal dependency footprint for local development, and straightforward, synchronous logic for unit/integration testing. For production, the modular adapters (STT, speaker recognition) are pluggable and can be swapped for remote or GPU-backed models.

---

## Pipeline & Data Flow (Detailed)

1) Client connects to the WebSocket endpoint and provides a short-lived JWT token obtained from `/sessions/start`. If token doesn't validate, connection is closed.

2) Frames are sent via `{"type":"frame","pcm": [..], "ts": <ms>, "transcript_override": <str>?}`. If the session is in the `calibrating` state frames are stored as calib frames; otherwise they are stored as `last_frames`.

3) The background batch processor runs periodically (0.5s):
   - Pop frames (either non-calibration or calibration frames) from `InMemorySessionStore`.
   - Acoustic extraction: runs `acoustic.AcousticEngine.process_batch_async()` to extract low-dim acoustic features and QC metrics.
   - Send `acoustic_batch` websocket event.
   
  - Run STT: call `stt_adapter.transcribe()` (adapter selected by `STT_ADAPTER` environment variable; default is `mock` in tests; set `STT_ADAPTER=whisper` to use `WhisperSTTAdapter` in integration setups — note that Whisper may require heavy dependencies and model downloads).
  ; results saved in session store and sent as `asr_batch`.
   
   - Optionally gate linguistic analysis by ASR confidence (`MIN_ASR_CONF`) — run `linguistic.LinguisticEngine.analyze()` and emit `linguistic_batch`.
   - Assemble feature frames: use `assembler.assemble_feature_frame(session_id, acoustic, linguistics, timestamp)`.
   - Baseline computation: during `calibrating` state compute baseline using `session.compute_and_store_baseline(..)` when enough quality frames available (default minimum 3 in tests, stricter defaults for production policy).
   - Normalization: z-score metrics in `feature_frame['normalized']` using baseline stats.
   - Scoring: use `scorer.Scorer.compute()` to produce a score, CI, and `explain` dict, persisted into session state and optionally stored persistently (opt-in flag per session).
   - Emit `feature_batch` event containing assembled frames with normalized metrics, score, ci and explain.

Diagram (ASCII):
```
Client (WS) -> /ws/session_id?token=token
  -> SessionStore.add_frame / add_calib_frame
  -> AcousticEngine.process_batch_async -> acoustic batch
  -> STT adapter -> asr_batch
  -> LinguisticEngine.analyze -> linguistic_batch
  -> assemble_feature_frame -> baseline & normalization -> scoring -> feature_batch
```

---

## Key Components & Files
- `src/voicecred/main.py` — Ingress API, WebSocket orchestration, overall batch loop logic, gating and event flow; stage timeouts and `MIN_ASR_CONF` defined here.
- `src/voicecred/session.py` — `InMemorySessionStore` & `SessionState`; implements creation, lifecycle state transitions, baseline computation, calibration policies, naive rate-limiting and persistence hooks.
- `src/voicecred/acoustic.py` — Acoustic extraction, using Parselmouth if available; produces vector features and `qc` metrics: `snr_db`, `speech_ratio`, `voiced_seconds`, with numpy fallback for testing.
- `src/voicecred/stt.py` — STT adapter interfaces and implementations: `MockSTTAdapter`, `WhisperSTTAdapter`, `RemoteSTTAdapter`, `create_stt_adapter()`.
- `src/voicecred/linguistic.py` — Linguistic analysis implementation using spaCy if available or fallback tokenization; extracts features like `ttr`, `pronoun_ratio`, `speaking_rate`.
- `src/voicecred/assembler.py` — `assemble_feature_frame()` which composes acoustic, linguistic features into a consistent frame schema (with `derived` features and `qc` fields).
- `src/voicecred/scorer.py` — Scoring logic using weight tables, median/MAD normalization and optional CI estimation.
- `src/voicecred/utils/baseline.py` — robust median/MAD helpers, `z_score`, `normalize_sequence`, small rolling-window helper.
- `src/voicecred/utils/logger_util.py` — Helper for creating named loggers.
- `scripts/` — CLI/test harness & diagnostic helpers, plus `nm.ps1` for developer convenience on Windows.
- `tests/` — Ex: `test_app.py`, `test_stt_and_linguistic.py`, `test_session.py`, `test_scorer.py`, etc.

---

## Runtime Configuration & Environment
- `.env` loads secrets (e.g. `HF_API_KEY`), and optional `VOICECRED_DEBUG` toggles debug logging.
- `STT_ADAPTER` environment var selects STT adapter: `mock`, `whisper`, or `remote`.
- `RUN_HEAVY_PYANNOTE=true` toggles heavy tests that require `pyannote` and other GPU/PyTorch dependencies.

---

## Developer Workflows: Install, Run, Build & Tests

Common commands (shell examples):

- Create virtual env (recommended) and install dependencies:
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

- Run the server for local development:
```bash
uvicorn voicecred.main:app --reload --host 127.0.0.1 --port 8000
# or to test via CLI harness scripts (e.g. scripts/client_harness.py)
```

- Run tests (default):
```bash
pytest -q
```

- Run tests enabling heavy Pyannote tests (if you have access & env setup):
```bash
# Example (POSIX or PowerShell environment variation):
$env:RUN_HEAVY_PYANNOTE = 'true'  # Windows PowerShell
pytest -q
```

- Windows Build helper (CMake, JOM for `NMake Makefiles JOM` generator):
```powershell
# Run inside a Visual Studio developer prompt (vcvars64.bat) and ensure jom in PATH
cmake -S . -B build -G "NMake Makefiles JOM" -T v141
cmake --build build --config Release  # or Push-Location build; jom; Pop-Location
```

Notes:
- The repo includes `nm.ps1` as a helper to inspect and repair PATH and safely add `jom`.
- Do not use `setx PATH` because it historically truncates PATH values; prefer PowerShell's `[Environment]::SetEnvironmentVariable`.

---

## Data Models & Feature Schema (Important for Changes)
A `feature_frame` produced by `assemble_feature_frame` will (v1) include keys:
- `feature_version`: integer
- `session_id`: string
- `timestamp_ms`: int
- `acoustic`: list of floats [f0_mean,f0_median,f0_std,rms,zcr] or a dictionary
- `acoustic_named` : dict with named components (ex: `f0_mean`, `rms`)
- `linguistic`: dict with linguistic features
- `derived`: list of small dicts like `{"pause_ratio": 0.4}`
- `qc`: dict containing QC fields: `snr_db`, `speech_ratio`, `words_in_window`, etc.

When scoring results are computed, additional keys are appended:
- `normalized`: mapping from metric name -> z-score
- `score`: float 0..100
- `ci`: [float, float]
- `explain`: detailed contributions and weights

---

## Baseline, Normalization & Scoring
Key implementation details to be aware of if you're changing scoring behaviour:

- Baseline is computed from `calib_frames` using median/MAD per metric. See `session.compute_and_store_baseline` and `utils/baseline.compute_median_mad`.
- QC gating: frames are filtered by `speech_ratio`, `voiced_seconds`, or `snr_db` before baseline accumulation. The pipeline also allows ASR-overrides (`words_in_window`) to rescue otherwise empty audio frames for tests.
- There are two baseline policies: `test` and `production` with stricter voiced-second requirements for production.
- Scorer: `Scorer.compute(frame, baseline)` uses a weight table (`Scorer.DEFAULT_WEIGHTS`) to build raw aggregated value from z-scores; it then maps `raw -> 0..100` using scaled tanh.
- CI & EMA: the scorer supports analytic CI estimation (fast) and a bootstrap style fallback; EMA smoothing is updated per session via `Scorer.update_ema`.

If you add a new metric you should:
1) Ensure it's present in `assemble_feature_frame` under `acoustic`, `linguistic` or `derived`.
2) Ensure the baseline collects it in `InMemorySessionStore.compute_and_store_baseline` by recognizing the metric path.
3) Update `Scorer.DEFAULT_WEIGHTS` if it should affect scoring and add unit tests that assert the new metric appears in `explain` / `z_scores`.

---

## Extension Points & Conventions for Contributions
- STT adapters: Add new adapters to `stt.py` and return the standard structure: `{"words": [...], 'confidence': float, 'raw': text}`. Also add a deterministic override path for tests (e.g., `transcript_override`).
- Speaker/diarization adapters: Provide `create_speaker_adapter(name)` and a fallback `MockSpeakerRecognitionAdapter` for tests.
- When adding heavy dependencies (pyannote, Whisper, GPU libs) ensure tests remain optional; add gates like `RUN_HEAVY_PYANNOTE=true` and mocks for CI.
- Avoid modifying `STAGE_TIMEOUTS` and `MIN_ASR_CONF` without updating tests that assert pipeline events or gating behaviour.

---

## Testing Patterns & Examples
- Use `TestClient` in `tests/test_app.py` to connect to the WebSocket endpoints and assert message events (e.g. `acoustic_batch`, `asr_batch`, `linguistic_batch`, `feature_batch`).
- Use `MockSTTAdapter` with `transcript_override` to force deterministic ASR outputs and test full pipeline behaviours in deterministic ways.
- Use `conftest.py` to configure `sys.path` and optionally add a local venv site-packages for heavy dependencies.
- `recv_with_timeout(ws, timeout)` helper runs `ws.receive_json()` in a thread pool and prevents the test thread from blocking forever; use it when expecting background events from the batch processor.

Example test snippets (from `tests/test_app.py`):
```python
with client.websocket_connect(f"/ws/{sid}?token={token}") as ws:
    ack = ws.receive_json()
    assert ack["session_id"] == sid
    ws.send_json({"type": "control", "cmd": "reset"})
    out = ws.receive_json()
    assert out.get("status") == "calibrating"

    # send frames with transcript override to trigger pipeline
    for i in range(4):
        ws.send_json({"type": "frame", "ts": i, "pcm": [], "transcript_override": "this is a test"})

    # wait for messages and validate events
```

---

## Troubleshooting & Known Caveats
- Windows PATH issues: avoid `setx PATH` (truncates values); use `nm.ps1` helpers to inspect and fix `PATH` using `[Environment]::SetEnvironmentVariable`.
- Running heavy tests: `pyannote` requires a HF token and system GPU resources; heavy tests are gated by `RUN_HEAVY_PYANNOTE=true`.
- The app is a prototype — `InMemorySessionStore` is not durable. Persistent baselines only exist if `persist_baseline` is set on the session.
- Single-process constraints: long-running operations should not block the event loop (processor uses `asyncio.to_thread` where necessary).

---

## Quick Commands & Shortcuts
- Start server:
```bash
pip install -r requirements.txt
uvicorn voicecred.main:app --reload
```
- Run local tests (fast tests):
```bash
pytest -q
```
- Enable heavy tests (if you have HF tokens & GPU):
```bash
# PowerShell
$env:RUN_HEAVY_PYANNOTE = 'true'
pytest -q
```
- Add `jom` to your PATH without using setx on Windows (see nm.ps1):
```powershell
.
m.ps1  # load functions
Add-ToPath -NewPath 'C:\tools\jom' -Scope User
```

---

## FAQ & Notes for Future Work
- Where should I add new features? Add new features to `assemble_feature_frame`, baseline computation and `Scorer` if they must affect scores.
- Should we maintain `SessionState` backwards compatibility? Yes — test and maintain `feature_version` and schema compatibility in `assemble_feature_frame`.
- Where to add new WebSocket event types? Update `tests/test_app.py` and add a new test to assert the new event type is emitted.

---

If you'd like I can also:
- Add a `docs/architecture.md` with an expanded sequence diagram and a per-file map to help new contributors.
- Add a short contributing guide with recommended PR schema and test patterns.
- Add a GitHub Actions workflow for the Windows CMake build using `NMake Makefiles JOM`.

Please tell me what else you'd like added or expanded in this overview.
