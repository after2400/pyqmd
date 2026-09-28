---
name: pyqmd-librarian
description: Maintain a pyqmd index — add collections, refresh, embed, clean up. Use when the user asks to index, update, or maintain indexed knowledge. Never runs without explicit user confirmation.
---

# pyqmd-librarian

You are the only skill allowed to mutate the pyqmd index. The read-only
`pyqmd` skill searches and retrieves; you add, refresh, and clean up —
and only when the user explicitly asked.

## Rule: confirm-before-write

Never run `collection add`, `update`, `embed`, or `cleanup`
speculatively. Every mutation follows this sequence:

1. **Inspect** — `pyqmd collection list` / `collection show <name>` /
   `pyqmd status`. State what you see.
2. **Preview** — for `cleanup`, run `pyqmd cleanup --dry-run` first and
   show the numbers.
3. **Confirm** — state the exact commands you will run and wait for an
   explicit yes.
4. **Execute, then verify** — run the commands, then `pyqmd status`
   again and report the before/after.

## Intake checklist (new material)

- Convert non-markdown first: PDFs/ebooks/Docs → one `.md` per doc
  (e.g. `pandoc` or `marker`). Skip scanned/image-only PDFs until
  OCR exists — empty docs pollute vectors.
- One file per note/meeting; keep `YYYY-MM-DD-*.md` naming.
- Exclude noise at add time with repeatable `--exclude <glob>` on
  `pyqmd collection add` (e.g. `Archive/**`, `**/drafts/**`,
  `node_modules/**`, `.git/**`, `dist/**`, `build/**`, `.venv/**`).
- Add a context for every new collection and significant sub-path:
  `pyqmd context add qmd://<collection>/<path> "<description>"`
  (`pyqmd context list` to review). Contexts are returned with hits
  and drive agent relevance.

## Hook review

A collection's `update` hook runs via `bash -c` before every
`pyqmd update` and aborts the whole run on non-zero exit. Before the
first run on a new checkout, print the hook
(`collection show <name>`) and confirm it with the user. Never
blind-run someone else's hook.

## Health readout

`pyqmd status` shows per-collection `Files` / `Updated X ago`,
orphaned-content hints (`run 'pyqmd cleanup'`), and tips about missing
contexts or update commands — report these, don't just paste output.
