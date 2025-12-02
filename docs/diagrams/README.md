# Diagrams — Architecture Visuals

This directory contains PlantUML and (optionally) produced PNG/SVG diagrams that visualize the VoiceCred architecture.

Files:
- `architecture.puml` — the PlantUML source for the class diagram included in `docs/architecture.md`.
- `architecture.png` and `architecture.svg` — generated rendering of `architecture.puml` (added to this repo for immediate viewing).

Generate diagrams locally (requires Java + PlantUML + Graphviz):
```bash
# Example (install via apt / brew / choco as appropriate):
# Linux: apt-get install default-jre graphviz
# macOS: brew install plantuml graphviz
# Windows: install Java + graphviz + plantuml jar

# generate PNG
plantuml -tpng docs/diagrams/architecture.puml -o docs/diagrams/
# generate SVG
plantuml -tsvg docs/diagrams/architecture.puml -o docs/diagrams/
```

CI step (GitHub Actions) below will generate HTML/PNG/SVG on push and commit artifacts back to the repo when enabled.
