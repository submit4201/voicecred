# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Scorer: include contributions and explain metadata in `ScoreUpdated` events so UI and downstream consumers can show top contributors and explainability data.
- Add `score_events` channel for deterministic testing and internal consumers separate from `ui_out`.
- Add `scorer_ready` readiness marker to avoid race conditions when publishing ops events to the scorer loop.
- Harden tests and test flow for the scorer pipeline; add tests to validate ScoreUpdated payload shape.
