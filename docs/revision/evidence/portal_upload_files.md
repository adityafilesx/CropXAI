# Files for the portal upload (computed 2026-10-23, after the final build)

| File | Size | SHA-256 | Note |
|---|---|---|---|
| `output/pdf/445_revised_version.pdf` | 682,220 bytes | `4328414e293a0754b3f480082081d7219be9e4537662efdd2f528c397f62703a` | manuscript + response (required) |
| `output/pdf/reviewer_response_sheet.pdf` | 101,962 bytes | `62175fe30dd8ca0b40365cd20ab41b3282e6d673e3ff836fa3b2962ded4fec61` | only if the portal requires a separate sheet |
| `output/pdf/similarity_report.pdf` | - | - | MISSING - must be supplied by the authors |

Recompute with `shasum -a 256` after any rebuild; the PDF changes whenever `build_manuscript.py` runs.
