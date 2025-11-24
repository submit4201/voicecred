Title: scorer: include contributions & explain in ScoreUpdated events; add score_events channel + tests

Summary
-------
This PR does three related things to improve scoring instrumentation and test stability:

- Ensure ScoreUpdated events include the Scorer's detailed metadata: `contributions` (top scoring contributors) and `explain` (diagnostics / breakdown). This makes scoring results more useful to UI and downstream consumers.
- Add a dedicated `score_events` channel (in the in-process EventBus) for deterministic test/consumer inspection separate from the `ui_out` queue that may be consumed by websocket forwarders.
- Harden background-scorer startup/testing by adding a `scorer_ready` marker, better logging, and safer test patterns so the CI/test harness is less likely to encounter race conditions.

Files (high-level)
-------------------
- src/voicecred/main.py — add `scorer_ready`, publish contributions + explain fields, create pre-registered `score_events` channel and publish to it
- src/voicecred/schemas/envelope.py — extend ScoringPayload to accept contributions & explain fields
- src/voicecred/assembler/* / src/voicecred/session/* / src/voicecred/bus/* / tests/* — many related refactors and tests to support the event-bus and deterministic test harness
- CHANGELOG.md — new change entry

Why
---
Providing the contributions/explain metadata in ScoreUpdated events enables richer UI displays (top contributing features) and debugging. Adding an internal `score_events` channel avoids timing races with UI-forwarders and makes the scoring pipeline easier to test and introspect.

Backward compatibility
----------------------
ScoreUpdated events still include the prior `score` and CI fields. Consumers that rely on the older event shape will still get them; this PR *adds* non-breaking, optional fields (`contributions` and `explain`).

Testing
-------
- Unit and integration tests were added and updated to validate the new payload shape and to avoid background-run race conditions.
- Ran the full test suite locally: 63 tests passed, 1 skipped (heavy test), 19 warnings.

Suggested reviewers
-------------------
- @submit4201

Labels / Notes
--------------
- label: enhancement
- label: tests
