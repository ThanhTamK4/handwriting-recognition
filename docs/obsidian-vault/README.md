# Handwriting Project — Obsidian Vault

Knowledge graph of the handwriting recognition project. Open this folder as an Obsidian vault to navigate the architecture visually.

## Open in Obsidian

1. Install [Obsidian](https://obsidian.md).
2. `Open folder as vault…` → select `docs/obsidian-vault/`.
3. Trust the vault when prompted (needed for community plugins).

## Enable Graphify (recommended)

The built-in Graph view works out of the box. For richer per-tag coloring and force tuning, install **Graphify**:

1. `Settings → Community plugins → Browse`
2. Search **Graphify**, install, enable.
3. Open via `Ctrl+P → Graphify: Open graph`. The `.obsidian/graph.json` in this vault already defines color groups per `type/*` tag (modules = blue, concepts = magenta, pipelines = green, dataset = orange, UI = purple, bugs = red).

## Structure

```
index.md                   — MOC (start here)
modules/                   — one note per src/ and training/ file
concepts/                  — ML, UI, bug, and backend concepts
pipelines/                 — Training + Inference end-to-end flows
data/                      — Dataset notes
ui/                        — UI surface notes
.obsidian/graph.json       — color groups + force params
```

Every note uses YAML frontmatter (`type`, `tags`, `aliases`) and `[[wiki-links]]`. The graph renders those links as edges.

## Entry points

- [[index]] — manual table of contents
- [[Inference Pipeline]] — how a user request becomes text
- [[Training Pipeline]] — how raw IAM data becomes `model.onnx`
- [[TrOCR vs mltu]] — why we ship both backends
