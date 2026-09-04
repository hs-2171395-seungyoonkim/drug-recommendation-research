# SafeDrug visit-principal-condition audit

This private repository contains source code, tests, and design documents only.
It intentionally excludes MIMIC-III source data, derived row-level clinical data,
model outputs, and credentials.

## Local data required

Place the following files in the repository root after obtaining or securely
transferring them under the applicable PhysioNet credentialed-data agreement:

- `ADMISSIONS.csv`
- `DIAGNOSES_ICD.csv`
- `D_ICD_DIAGNOSES.csv`
- `NOTEEVENTS.csv`
- `data4LLM_with_note.csv`
- `umls_kb_2022ab.jsonl`

Generated row-level artifacts belong under `out/` and must not be committed.

## Environment check

The current development environment uses Python 3.12.

```powershell
py -3.12 -m pytest tests -q
py -3.12 --version
```

The current implementation plan is
`docs/plans/2026-08-24-visit-principal-condition-clustering.md`.

