# VoiceCred — Credibility Scoring Prototype

Lightweight skeleton implementation for the real-time credibility scoring pipeline.

This repository includes an ingress/service skeleton (FastAPI) and in-memory session store to bootstrap the architecture described by the user.

Goals implemented here (initial):

- FastAPI + uvicorn skeleton with REST & WebSocket endpoints
- In-memory session store for session lifecycle (reset, calibrate, finalize)
- JWT token issuance for ephemeral WS tokens
- Simple client harness to send PCM frames (test) and control commands
- Unit tests for session lifecycle

Run locally (dev):

1) Install dependencies

   pip install -r requirements.txtpip install

2) Start the server

   uvicorn voicecred.main:app --reload --host 0.0.0.0 --port 8000

3) Run tests

   pytest -q

Client harness

1) Ensure server is running (uvicorn)
2) Run the Python harness:

   python scripts/client_harness.py

 python for this venv = C:/workspace/voicecred/.conda/python
 must run
 conda activate  C:/workspace/voicecred/.conda
 to start venv before anthing in a new term

 example
 *tests*

 ```bash
 conda activate  C:/workspace/voicecred/.conda/
 pytest -q
 ```

 or

 ```bash
  C:/workspace/voicecred/.conda/python.exe pytest -q
```

## *Starting venv*

```ps
conda activate C:/workspace/voicecred/.conda
```

Testing notes
------------

Some tests are intentionally heavy (they load the pyannote.audio pipeline and may require a Hugging Face token or additional dependencies). These tests are skipped by default in CI and local runs. To enable them locally set the environment variable:

```bash
export RUN_HEAVY_PYANNOTE=true  # or on Windows PowerShell: $env:RUN_HEAVY_PYANNOTE = 'true'
pytest -q
```
