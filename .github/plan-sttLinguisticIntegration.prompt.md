<|---IGNORE---|> ingore below it is an old plan <|---IGNORE---|>
# STT Adapter + Linguistic Engine — Integration Plan
TL;DR — Add a pluggable STT adapter layer plus a spaCy-backed (with lightweight fallback) Linguistic Engine. Integrate into the worker per-frame and per-batch flows (non-blocking async → threadpool) and gate linguistic features by ASR confidence. Deliver unit + integration tests, an end-to-end mocked WS test, and metrics/traces for observability.

## Steps (high-level)
1. Add adapter injection points and configuration for STT (use `src/voicecred/stt.py`): supports Mock + production adapters (Whisper/VOSK/Cloud).
2. Integrate STT into per-batch processing in `src/voicecred/main.py` background batch processor (call `STTAdapter.transcribe()` off the event loop).
3. Add `LinguisticEngine` calls (`src/voicecred/linguistic.py`) to transform ASR results into linguistic features and store in session state.
4. Add per-window feature assembly & gating by ASR confidence, update session snapshots & tests.
5. Create unit tests (mock STT, spaCy blank pipeline) and integration WS tests (using `transcript_override`) validating the end-to-end pipeline.

---

## Module-by-module decomposition (concise)

### 1) Ingress / Session Manager (existing)
- Responsibilities: WS/REST auth, create/route session state, rate-limits, issue tokens.
- Inputs: session start, frames, control commands.
- Outputs: ack, tokens, routing.
- Files: `src/voicecred/main.py`, `src/voicecred/session.py`.

### 2) Layer‑2 Acoustic Engine (existing)
- Responsibilities: PCM → acoustic features & QC.
- Notes: synchronous CPU work run via `asyncio.to_thread()` or threadpool; support `process_frame` + `process_batch`.
- Files: `src/voicecred/acoustic.py`.

### 3) STT Adapter Layer (new/exists)
- Responsibilities: expose pluggable adapter interface and adapters.
- Inputs: PCM (batch or short-window), optional transcript_override from clients.
- Outputs: timestamped words, per-word confidences, overall ASR confidence, raw text.
- Implementation notes:
  - Protocol: `STTAdapter.transcribe(frames)` returns {words[], confidence, raw}.
  - Provide `MockSTTAdapter` for tests; production adapters to be added separately behind the interface.
  - Run transcription in a threadpool (non-blocking).
- Files: `src/voicecred/stt.py` (already present).

### 4) Linguistic Engine (new/exists)
- Responsibilities: compute per-window linguistic features (pronoun_ratio, article_ratio, TTR, sentence complexity, avg_tokens).
- Inputs: STT result or client-provided transcript.
- Outputs: linguistic feature vector + ASR-quality score.
- Notes:
  - Use spaCy when available; fallback tokenizer if not.
  - Gate output if ASR confidence < threshold (configurable, e.g., 0.6).
- Files: `src/voicecred/linguistic.py` (already present).

### 5) Feature Assembler
- Responsibilities: join acoustic + linguistic + qc + metadata into a unified feature frame.
- Inputs: acoustic result, linguistic result, timestamps.
- Outputs: feature frame (see schema).

### 6) Baseline Calibrator / Credibility Scorer / QC
- Responsibilities: calibrate, compute baseline stats, Z-scores, EMA smoothing, confidence intervals, top_factors, QC gating.

### 7) Session State Store
- Responsibilities: store last N frames, calib frames, transcripts, and scorer state; snapshot on finalize.
- Files: `src/voicecred/session.py`.

---

## Interface contracts / message schemas (concise)

Feature Frame (v1)
- Keys: feature_version:int, session_id:str, timestamp_ms:int, acoustic:float[], linguistic:float[]|null, derived:[], qc:{snr_db, speech_ratio, asr_conf, num_speakers, words_in_window}

Scorer Output (server → client)
- Keys: timestamp_ms, score (0–100), ci:[low, high], label, z_scores:{...}, top_factors:[{metric,z,weight,explanation}], qc:{score,reason}

ASR Event (ws server -> client)
- Keys: type:"asr", timestamp_ms, raw, confidence, words:[{word,start_ms,end_ms,confidence}]
- Example: emitted per-window or per-batch when a transcription is available

Linguistic Event (ws server -> client)
- Keys: type:"linguistic", timestamp_ms, linguistic:{pronoun_ratio,article_ratio,ttr,avg_tokens_per_sentence}, asr_quality

Control messages
- Client → server: {"type":"control","cmd":"reset"} or {"type":"control","cmd":"finalize"}
- Server responses: {"status":"calibrating", "duration_sec":X}, {"status":"finalized","calib_quality":N}

Session snapshot (finalize)
- Include number of frames, calibration_profile summary, persisted transcripts (only if opted-in)

---

## Sequence diagram (mermaid + ASCII fallback)

Mermaid
```
sequenceDiagram
  participant Client
  participant Ingress
  participant Worker
  participant Acoustic
  participant STT
  participant Linguistic
  participant Assembler
  participant Calibrator
  participant Scorer
  participant UI

  Client->>Ingress: open ws(session_id)
  Client->>Ingress: send {"cmd":"reset"}
  Ingress->>Worker: create session state
  Client->>Ingress: stream audio frames
  Ingress->>Worker: forward frames
  Worker->>Acoustic: submit pcm frames (threadpool)
  Worker->>STT: submit pcm batch (async to threadpool)
  STT-->>Worker: transcript + word timestamps + conf
  Worker->>Linguistic: analyze transcript → linguistic features
  Acoustic-->>Worker: acoustic features + QC
  Worker->>Assembler: assemble feature frame
  Assembler->>Calibrator: if calibrating->store
  Assembler->>Scorer: if scoring -> compute score
  Scorer-->>Worker: score + CI + explain
  Worker->>Ingress: send score event
  Ingress->>Client: ws send (score event)
  Client->>UI: render gauge & explain
```

ASCII fallback
```
Client -> Ingress (WS) -> Worker
Worker -> Acoustic (parselmouth)
Worker -> STT (batch async)
STT -> Worker -> Linguistic Engine -> Worker
Worker -> Assembler -> Calibrator/Scorer -> Worker -> Ingress -> Client
```

---

## Deployment & scaling architecture (no Redis required)
- Minimal dev: single host with uvicorn, background threadpool. DB optional (SQLite).
- Production (recommended): multiple worker nodes behind a load balancer with sticky sessions to preserve in-memory session state. Use Postgres for metadata and S3 for artifacts.

---

## Observability & logging
- Metrics (Prometheus): scorer_latency_ms, frames_processed_total, sessions_active_count, acoustic_latency_ms, stt_latency_ms, linguistic_latency_ms, qc_inconclusive_count
- Tracing (OpenTelemetry): include model_version, session_id, top_factors, ASR_conf
- Logs: structured JSON, redact transcripts unless consented
- Alerts: p95 latency > 300ms, QC inconclusive rate > 20%, error-rate spike > 1%

---

## Testing & validation
- Unit tests (STT adapter/mock, LinguisticEngine, Feature Assembler, Calibrator, Scorer)
- Integration tests (WS flow: frames -> acoustic -> stt -> linguistic -> feature-frame -> scorer)
- E2E tests using prerecorded audio with expected labels

---

## Short-term Sprint checklist (next 2 weeks)
Week 1
- Add STT adapter configuration & injection (Mock + adapter skeleton). (files: `src/voicecred/stt.py`, `src/voicecred/main.py`)
- Hook STT into the background batch processor with a non-blocking threadpool call; produce `asr_batch` events.
- Add session state fields for transcript storage & add snapshot inclusion on finalize.

Week 2
- Add LinguisticEngine integration with gating by ASR confidence and emit `linguistic_batch` events.
- Wire linguistic features into Feature Assembler to create full feature frames.
- Add unit/integration tests: end-to-end WS flow (synthetic frames -> MockSTT -> Linguistic -> Assembler) and expand CI.

Deliverables
- Deterministic Mock STT + unit-tests
- ASR batch & linguistic batch events in WS (integration tests)
- Updated Feature Assembler and tests
- Dev docs + README notes for optional audio deps

---

If you prefer, I can now implement step 1: "STT → background batch → emit `asr_batch`" (mock-first) with tests; or implement step 2 (LinguisticEngine gating & feature assembler) — tell me which and I’ll proceed.



<|---IGNORE---|>




---

## Plan: Research-grade stress‑tension scoring (high-level)
TL;DR — Build a baseline-calibrated, explainable scoring pipeline that converts acoustic + linguistic signals into a smoothed, uncertainty-aware stress‑tension index (0–100). Start with baseline calibration + normalization, then scoring & integrator, then evaluation/CI, finalizing with observability and deployment readiness.

### Major phases (priority order)
1. Baseline & normalization (core foundation) ✅
2. Scorer / integrator + explainability (weights, EMA, CI) ✅
3. Robust QC + gating + instrumentation (noise, ASR conf, multi‑speaker) ✅
4. Evaluation & dataset / labelling strategy (controlled deception studies & confounder analysis)
5. Production polishing: metrics, tracing, privacy, docs

---

## Steps (3–12 actionable items, prioritized)
1. Implement Per-Session Baseline + Robust Normalizers (z-score & MAD) — core foundation
   - Files: session.py (extend SessionState), acoustic.py (expose raw features), linguistic.py (ensure tokens/numeric outputs), utils (new helper baseline module).
   - Actions:
     - Add `SessionState.baseline` metadata (median, MAD per metric, counts).
     - Add `store.add_calib_frame(session_id, frame)` usage for calibration windows (already present).
     - Implement baseline computation: per metric median and MAD, rolling baseline updates, baseline TTL.
     - Add normalization helpers: z = (x - median)/MAD_norm (MAD->sigma equiv ≈ 1.4826*MAD).
   - Outcome: All downstream code can request normalized features for scoring — essential, difficult (data correctness) and unlocks scoring.

2. Add Scorer Module & Real‑Time Integrator — the scoring core
   - Files: `src/voicecred/scorer.py` (new), assembler.py (include normalized features), main.py (invoke scorer for batches).
   - Actions:
     - Define score schema: per-frame metrics normalized, per-feature weight table + directionality (higher = more tension or opposite).
     - Implement flexible weight config and explainability metadata (which metrics contributed, z-scores, per-feature contribution).
     - Implement smoothing: EMA with configurable alpha per-session; persistent state kept in SessionState.
     - Implement short-term CI estimation: bootstrap sample of recent window or analytic propagation across z-scores → simple approximate CI.
     - Add "inconclusive" gates: if QC fails or ASR conf < threshold OR insufficient baseline data.
   - Outcome: Real-time, explainable stress‑tension index + per-frame contributions.

3. Rigorous QC and gating improvements
   - Files: main.py (stronger gates), session.py.
   - Actions:
     - Expand QC metrics (ambient noise estimate, energy floor, overlap detection).
     - Fail fast: mark frames/sessions as 'inconclusive' when noise, multi-speaker, or low ASR confidence.
     - Expose QC reason codes in pipeline_status / assembled frames.
   - Outcome: Safer, more defensible outputs and fewer false signals.

4. Validation / Dataset + Evaluation Harness
   - Files: `scripts/evaluation/` (new), `tests/fixtures/` (data), `tests/test_scoring.py`.
   - Actions:
     - Define controlled tasks for evaluation (mock deception tasks with ground-truth labels for stress proxies).
     - Build synthetic data + small recorded sample set for unit tests and benchmark.
     - Validate metrics: AUC, balanced accuracy, calibration curves (Brier), CI coverage, false alarm analysis.
   - Outcome: Research-grade evaluation metrics and reproducible experiments.

5. Uncertainty & Explainability polishing
   - Files: `src/voicecred/scorer.py`, `src/voicecred/utils/explainability.py`.
   - Actions:
     - Offer per-frame explanation: per-feature z-score, contribution, top-3 drivers.
     - CI calibration via bootstrap or analytic variance propagation.
     - Emphasize uncertainty in outputs and classify into Green/Yellow/Red/Inconclusive.
   - Outcome: Defensible, transparent output suitable for research.

6. Observability, reproducibility & privacy
   - Files: main.py (metrics), `src/voicecred/utils/tracing` (new), CI config updates.
   - Actions:
     - Add stage timing metrics and counters (Prometheus).
     - Log minimal redacted telemetry (no raw transcripts unless opted-in).
     - Add privacy docs + session opt-in/out + data retention.
   - Outcome: Production-grade observability and compliance.

---

## Engineering-level sub‑tasks (short checklist)
- Baseline & normalizers
  - [ ] Add SessionState.baseline structure; implement compute_baseline(session_id)
  - [ ] Add normalize_metric()/denormalize helpers + tests
- Scorer
  - [ ] Design weight config (YAML/JSON) and default weights (acoustic high weight for jitter/shimmer, pauses, prosodic congruence; linguistic medium/low)
  - [ ] Implement Scorer.run(frame_or_batch, baseline) -> ScoreResult {score, ci, contributions}
  - [ ] Implement EMA smoothers persisted per session
  - [ ] Add `feature_batch` enrichment with `score` & `explain`
- QC
  - [ ] Expand QC detection (SNR threshold, ASR words count, multi-speaker gating)
  - [ ] Add `pipeline_status` inconclusive reasons
- Tests
  - [ ] Unit tests for baseline utils
  - [ ] Integration tests: simulate frames, baseline warm-up, ensure normalized values & assembled feature have 'score' and 'explain' keys
  - [ ] Eval harness: reproducible sample data + evaluation script
- Docs & privacy
  - [ ] Add methodology docs explaining caveats (no claim of truth detection, focus on stress‑tension signals, baseline necessity)
  - [ ] Add README with steps to run local experiments, opt-in flags for transcript storage, and retention policy

---

## Research & ethical considerations (must include in docs)
- Avoid claims of lie detection or truth — explicitly state scope: stress‑tension SIGNALS correlated with deceit in controlled conditions; NOT a classifier of truth.
- Baselines and controlled settings: require subject consent and explicit warning about limitations, potential confounds (caffeine, illness, accent, language), fairness checks and demographic analyses.
- Avoid automated decisions; surface outputs as signals requiring human interpretation.
- Data minimization: default to not storing transcripts; make storage explicit & auditable.

---

## Deliverables & timeline suggestions (rough)
- Week 1–2: Baseline + normalization + unit tests + small end‑to‑end smoke tests. (Hardest foundations done early.)
- Week 2–4: Scorer and real‑time integrator + smoothing and CI + feature explainability + integration tests.
- Week 4–6: QC upgrades, evaluation harness, synthetic/recorded datasets + metrics and reproducible experiments.
- Week 6+: Productionization: metrics/export, docs, privacy/consent/ops integration.

---

## Why start with baseline/calibration
- Baseline normalization is the single most important technical enabler for defensible, explainable outputs. Without robust per-subject normalization (median/MAD, warm-up, rolling update), feature comparisons are meaningless across individuals and environments. Doing it first reduces downstream model/human interpretation errors and makes the scoring stage simpler & mathematically sound.

---

If you want I can:
- Draft a concrete file-by-file implementation plan now (no code), or
- Start implementing the baseline + normalization code (create new tests and interfaces) — which is my recommended next move.

Which would you prefer next? ✅

per-session baseline statistics (median + MAD), robust normalization helpers, and a real-time, explainable scorer with EMA smoothing and CI. Changes are small, testable, and designed to integrate into the existing acoustic → STT → linguistic → assembler → session pipeline.

### Steps (file-by-file, high level)
1. Add baseline helpers utilities
   - Files: `src/voicecred/utils/baseline.py` (new)
   - What: Implement robust helpers: compute_median_mad(), mad_to_sigma conversion (1.4826), z_score(x, median, mad), rolling/online median & MAD helpers (optional).
   - Tests: `tests/test_baseline.py` — unit tests for median/MAD accuracy, z-scores, and rolling update behavior.

2. Extend session storage for baseline & scoring state
   - Files: session.py (modify)
   - What: Add `SessionState.baseline` structure (per-metric: median, mad, count, last_updated), `SessionState.scoring_state` (EMA state, last_scores, smoothing_alpha). Add helper methods: `add_calib_frame()`, `compute_and_store_baseline()`, `has_valid_baseline(metric)` and TTL / warmup tracking.
   - Tests: Update test_session.py — ensure baseline accumulation, compute_baseline result shape, TTL/expiry semantics, persisted smoothing state.

3. Ensure raw features are consistently emitted
   - Files: acoustic.py and linguistic.py (modify)
   - What: Standardize and explicitly expose numeric feature keys (names and types) returned by per-frame/batch functions to make downstream normalization deterministic. Add small feature(s) useful to scoring (e.g., speaking_rate, pause_ratio if available).
   - Tests: Update test_acoustic.py and test_stt_and_linguistic.py to assert stable naming and numeric types.

4. Add scorer module (core scoring + explainability + CI)
   - Files: `src/voicecred/scorer.py` (new)
   - What: Implement Scorer class and ScoreResult model:
     - Accept normalized features or raw features + baseline.
     - Compute per-feature z-scores, apply sign/weight table to compute contributions, sum into a normalized index (0–100).
     - Provide per-frame explain: top contributing features and contributions.
     - Implement EMA smoothing per-session (persisted in SessionState) and simple CI estimate (e.g., analytic variance across weighted z-scores or bootstrap over recent window).
     - Expose configuration (weights/directionality) with sensible defaults and JSON/YAML override support.
   - Tests: `tests/test_scorer.py` — unit tests for contributions, combined score scaling, EMA smoothing, CI intervals, config overrides.

5. Wire scoring into assembly & pipeline
   - Files: assembler.py (modify), main.py (modify)
   - What: 
     - `assembler`: include `normalized` block in assembled frames and a `score`/`explain` placeholder.
     - `main`: after assembler step and QC gating, call Scorer using session baseline (or mark "inconclusive" if baseline insufficient). Persist EMA state into `SessionState.scoring_state`. Add `score` and `explain` fields to `feature_batch` events.
     - Add clear "inconclusive" reasons to `pipeline_status` (e.g., insufficient_calibration, low_asr_conf, high_noise).
   - Tests:
     - Update test_app.py to expect `feature_batch` includes `score` and `explain` and that low-confidence gates produce `inconclusive` pipeline_status.
     - Add integration tests for baseline warmup → normalized features → score present.

6. Add evaluation harness & scripts
   - Files: `scripts/eval_pipeline.py` (new), `scripts/evaluation/` (optional sample data)
   - What: Small reproducible harness to run prerecorded/synthetic audio through pipeline, generate per-session score traces, and compute evaluation metrics (AUC, Brier score, calibration curves) against provided labels.
   - Tests: Add smoke tests ensuring the harness runs on sample fixtures (low-cost synthetic data).

7. Documentation & configuration
   - Files: README.md (update), `docs/methodology.md` (new)
   - What: Document baseline rationale, normalization math (median/MAD), scoring defaults, explainability semantics, caveats (not lie detection), consent & data retention defaults, test instructions and sample evaluation guidance.

### Further Considerations (open design choices)
1. Warm-up / calibration policy: How many calibration frames constitute a valid baseline? (Default option: 60–120 seconds worth of voiced frames; configurable per-session.)
2. Baseline persistence: Should baselines persist across sessions for a subject (requires identifier & privacy opt-in) or be session-scoped only? (Consider opt-in persistent baseline for research.)
3. CI estimation method: Analytic propagation from weighted independent z-scores (fast) vs bootstrap over window (more robust but heavier). Decide based on target latency and compute budget.

---

✅ Next step: Review this file-by-file plan and tell me which choices you prefer for warm-up length, baseline persistence, and CI approach. Then I’ll refine or expand the plan into exact test names and method signatures (still no code).

<IMPORTANT: all above is old code you may review for completeness but we're moving on do a double check and proceed if all is complete.>
</|---IGNORE---|>
<|START|>
report - is all implemented and complete and finished?
