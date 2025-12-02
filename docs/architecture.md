# VoiceCred — Architecture & UML

This document provides UML diagrams and sequence views to help contributors and AI agents understand the runtime flow and the main components of VoiceCred.

The content includes both Mermaid diagrams (renderable on GitHub & many editors) and PlantUML sources for classic UML tooling.

---

## Component Diagram (Mermaid)
```mermaid
graph LR
  Client --> Connect --> App[fastapi.FastAPI]
  App --> SessionStore[InMemorySessionStore]
  App --> Acoustic[AcousticEngine]
  App --> STTAdapter[STTAdapter]
  App --> Linguistic[LinguisticEngine]
  App --> Assembler[assembler.assemble_feature_frame]
  App --> Scorer[Scorer]
  App --> Tests[tests]
```

### Notes
- `App` (FastAPI) orchestrates the pipeline: accepts frames via WS, persists them in `SessionStore`, runs acoustic extraction, STT, linguistic analysis, assembles features and triggers scoring.
- STT adapters are pluggable — tests default to `MockSTTAdapter` which supports `transcript_override` to provide deterministic transcripts for tests.

---

<!-- [MermaidChart: 76c27082-f609-4aa3-b222-d71a4855be6e] -->
<!-- [MermaidChart: 76c27082-f609-4aa3-b222-d71a4855be6e] -->
## High-level Sequence Diagram (Mermaid)
```mermaid
sequenceDiagram
  participant Client
  participant WS as FastAPI WS
  participant Store as InMemorySessionStore
  participant Acoustic as AcousticEngine
  participant STT as STTAdapter
  participant Linguistic as LinguisticEngine
  participant Assembler as assemble_feature_frame
  participant Scorer as Scorer

  Client->>WS: websocket connect /?token=...
  WS-->>Client: ack {session_id, state}
  Client->>WS: frame (pcm, ts, transcript_override?)
  WS->>Store: add_frame/add_calib_frame(session_id, frame)
  Note over WS: Background batch processor wakes periodically
  WS->>Store: pop_frames(session_id)
  WS->>Acoustic: process_batch_async(frames)
  Acoustic-->>WS: acoustic results (per frame)
  WS-->>Client: acoustic_batch
  WS->>STT: transcribe(frames)
  STT-->>WS: asr_result (words, confidence, raw)
  WS-->>Client: asr_batch
  alt asr_conf >= MIN_ASR_CONF
    WS->>Linguistic: analyze(asr_result)
    Linguistic-->>WS: linguistic_result
    WS-->>Client: linguistic_batch
    WS->>Assembler: assemble_feature_frame(acoustic, linguistic)
    Assembler-->>WS: feature_frames
    WS->>Store: compute_and_store_baseline(session_id) [if calibrating]
    WS->>Scorer: compute(feature_frame, baseline)
    Scorer-->>WS: score + explain
    WS-->>Client: feature_batch (with normalized + score + explain)
  else asr_conf low
    WS-->>Client: pipeline_status (linguistic skipped)
  end

```

---

## PlantUML Class Diagram (source)
Below is a PlantUML class diagram source for key classes and their main public methods. You can paste it into an online PlantUML editor or generate PNG/SVG with PlantUML locally.

```mermaid
classDiagram
class InMemorySessionStore {
  +create_session(session_id=None): SessionState
  +get(session_id): SessionState
  +add_frame(session_id, frame)
  +add_calib_frame(session_id, frame)
  +pop_frames(session_id): list
  +pop_calib_frames(session_id, max_n=None)
  +compute_and_store_baseline(session_id, min_frames=3): dict
  +add_asr_result(session_id, asr)
  +add_feature_frame(session_id, frame)
  +finalize(session_id): dict
}

class SessionState {
  +session_id: str
  +state: str  # idle|calibrating|scoring|finalized
  +calib_frames: list
  +last_frames: list
  +asr_results: list
  +feature_frames: list
  +baseline: dict
  +scoring_state: dict
}

class AcousticEngine {
  +process_frame(pcm, timestamp_ms)
  +process_batch(frames): list
  +process_batch_async(frames): list
}

class STTAdapter <<Interface>> {
  +transcribe(frames)
}

class MockSTTAdapter {
  +transcribe(frames)
}

class WhisperSTTAdapter {
  +transcribe(frames)
}

class LinguisticEngine {
  +analyze(asr_result, timestamp_ms)
}

class Scorer {
  +compute(frame, baseline)
  +update_ema(scoring_state, raw)
}

class Assembler {
  +assemble_feature_frame(session_id, acoustic, linguistic, timestamp_ms)
}

InMemorySessionStore "1" *-- "*" SessionState
WSApp ..> InMemorySessionStore : uses
WSApp ..> AcousticEngine : uses
WSApp ..> STTAdapter : delegates
WSApp ..> LinguisticEngine : uses
WSApp ..> Assembler : uses
WSApp ..> Scorer : uses

AcousticEngine -- Assembler : returns acoustic features
STTAdapter -- LinguisticEngine : provides transcripts
LinguisticEngine -- Assembler : returns linguistic features
Assembler -- Scorer : normalized & scoring inputs
```

Replace `WSApp` with `FastAPI main` when rendering to annotate the orchestrating application.

---

## How to generate diagrams locally
If you want to render PlantUML diagrams locally, install PlantUML and Graphviz (for PNG/SVG output) and run:

```bash
# Example: generate PNG from PlantUML text file
plantuml -tpng docs/architecture.puml
```

### PNG / SVG exports
![Architecture Diagram](../docs/diagrams/architecture.png)
<!-- For SVG viewing: -->
<img src="../docs/diagrams/architecture.svg" alt="Architecture Diagram" width="800" />

For Mermaid diagrams, many Markdown editors (including GitHub) can automatically render them when enclosed in ` ```mermaid ` blocks.

---

## Notes for contributors & AI agents
- Use these diagrams when modeling interactions for new features or when changing message event types over WebSocket. Update the sequence diagram when changing the batch loop or gating logic in `main.py`.
- Keep `assemble_feature_frame` deterministic (feature names and `feature_version`) so baseline computation is stable across runs and tests.
- If you add a new STT or speaker adapter, update the component and class diagrams to keep documentation accurate.

---

If you'd like, I can add automated rendering to the repo (e.g., a `docs/diagrams/` folder with PNG exports, or a step in CI that generates diagram artifacts) — let me know which format you prefer (SVG/PNG/PlantUML/mermaid).
