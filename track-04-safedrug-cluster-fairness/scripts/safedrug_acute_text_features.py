"""SafeDrug acute-attention text query features (R1).

For every cohort HADM_ID, builds the admission-time query text (normalised
ADMISSIONS.DIAGNOSIS + scrubbed chief-complaint text) and its frozen
Bio_ClinicalBERT embedding. Run with `py -3.14 -X utf8` for the real
embedding step; the pure text functions below are also imported and
unit-tested under `py -3.12` (no `transformers` import above function scope
-- see `load_clinicalbert`).

See docs/superpowers/specs/2026-09-07-safedrug-acute-attention-design.md
("R1") for the full contract.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import torch

TEXT_DIM = 768
_SEPARATOR_RE = re.compile(r"[\\;/]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalise(text) -> str:
    """Lowercase; replace backslash/semicolon/slash separators with ", ";
    collapse whitespace. Keeps "/SDA"-style suffixes as separate words
    (e.g. "GRAFT/SDA" -> "graft, sda") rather than deleting them."""
    if not text:
        return ""
    lowered = str(text).lower()
    replaced = _SEPARATOR_RE.sub(", ", lowered)
    return _WHITESPACE_RE.sub(" ", replaced).strip()


def scrub(text, drug_names) -> str:
    """Replace whole-word, case-insensitive matches of any name in
    drug_names with the literal token "[DRUG]". Longer names are tried
    first (regex alternation is first-match, not longest-match) so a
    multi-word generic name is not shadowed by a shorter name also present
    in the list. Known limitation (accepted, see design doc R1): a name
    starting/ending in a non-word character can behave oddly at that one
    boundary -- acceptable for a redaction pass, not a guarantee of zero
    leakage."""
    if not text:
        return ""
    names = sorted({n for n in drug_names if n}, key=len, reverse=True)
    if not names:
        return str(text)
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
    return pattern.sub("[DRUG]", str(text))


def compose_text(admission_diagnosis, cc_demog_stripped, drug_names) -> tuple[str, bool]:
    """R1's final query text and has_text flag for one visit. Missing
    DIAGNOSIS -> CC only; missing CC -> DIAGNOSIS only; both missing ->
    ("", False) -- the caller emits a zero embedding for it, never calls
    the model on it."""
    diag_part = normalise(admission_diagnosis) if admission_diagnosis else ""
    cc_part = scrub(cc_demog_stripped, drug_names).strip() if cc_demog_stripped else ""
    if diag_part and cc_part:
        return f"{diag_part} [SEP] {cc_part}", True
    if diag_part:
        return diag_part, True
    if cc_part:
        return cc_part, True
    return "", False


def mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool a transformer's last hidden state over non-padding tokens,
    then L2-normalise. hidden_state: (batch, seq, dim). attention_mask:
    (batch, seq) 0/1. Pure torch -- no transformers import -- so this is
    unit-testable under py -3.12 with a fake tensor pair."""
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    mean = summed / counts
    return torch.nn.functional.normalize(mean, p=2, dim=1)


def flatten_hadm_ids(hadm_ids: list[list[int]]) -> list[int]:
    return [h for patient in hadm_ids for h in patient]


def default_device_for(cuda_available: bool) -> str:
    """Pure resolution logic mirroring safedrug_train_dump.select_device():
    "cuda" if CUDA is available, else "cpu". Unit-testable under py -3.12
    without importing torch or safedrug_train_dump. The printed CPU-fallback
    warning itself is produced by select_device() (called from main() via a
    lazy import, so importing this module never pulls in torch_train_dump's
    heavier SOTA/SafeDrug model imports at module scope)."""
    return "cuda" if cuda_available else "cpu"


def build_drug_name_list(prescriptions_csv, min_rows: int = 50, min_len: int = 4) -> list[str]:
    df = pd.read_csv(prescriptions_csv, usecols=["DRUG_NAME_GENERIC"])
    names = df["DRUG_NAME_GENERIC"].dropna().astype(str).str.strip().str.lower()
    counts = names.value_counts()
    keep = counts[counts >= min_rows]
    keep = keep[keep.index.str.len() >= min_len]
    return sorted(keep.index.tolist())


def load_or_build_drug_name_list(cache_path, prescriptions_csv, min_rows: int = 50, min_len: int = 4) -> list[str]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return [line.rstrip("\n") for line in cache_path.open("r", encoding="utf-8") if line.strip()]
    names = build_drug_name_list(prescriptions_csv, min_rows, min_len)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def load_admission_diagnosis(admissions_csv) -> dict[int, str]:
    df = pd.read_csv(admissions_csv, usecols=["HADM_ID", "DIAGNOSIS"])
    out = {}
    for row in df.itertuples(index=False):
        if pd.notna(row.DIAGNOSIS):
            out[int(row.HADM_ID)] = str(row.DIAGNOSIS)
    return out


def load_cc_text(cc_pickle) -> dict[int, str]:
    df = pd.read_pickle(cc_pickle)
    out = {}
    for row in df.itertuples(index=False):
        if bool(row.has_cc) and pd.notna(row.cc_demog_stripped):
            out[int(row.HADM_ID)] = str(row.cc_demog_stripped)
    return out


def load_clinicalbert(device: str):
    """Lazy import: transformers is only installed under py -3.14.
    Importing this module (for its pure text/mean_pool functions, tested
    under py -3.12) must never require transformers -- so the import lives
    here, called only from main()'s embedding step."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", local_files_only=True)
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", local_files_only=True)
    model.eval()
    model.to(device)
    return tokenizer, model


def embed_texts(texts, tokenizer, model, device, max_length: int = 128, batch_size: int = 64) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            rows.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SafeDrug acute-attention text query features (R1)")
    parser.add_argument("--admissions-csv", type=str, default="ADMISSIONS.csv")
    parser.add_argument("--cc-pickle", type=str, default="out/36_cc_text.pkl")
    parser.add_argument(
        "--hadm-ids-pkl", type=str,
        default="out/acute_driver_audit/safedrug_mimic3_cohort/records_final_hadm_ids.pkl",
    )
    parser.add_argument(
        "--prescriptions-csv", type=str,
        default=r"C:\Users\Administrator\Desktop\MIMIC-III_v1.4\PRESCRIPTIONS.csv",
    )
    parser.add_argument("--drug-name-cache", type=str, default=None,
                         help="defaults to <out-dir>/drug_name_list.txt")
    parser.add_argument("--min-drug-rows", type=int, default=50)
    parser.add_argument("--min-drug-len", type=int, default=4)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="defaults to auto-selection via safedrug_train_dump.select_device() "
             "(cuda if available, else cpu with a printed warning)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.device is None:
        # Lazy import: avoids pulling safedrug_train_dump's heavier
        # SOTA/SafeDrug model imports (and torch) into this module's
        # import-time footprint, so `py -3.12` test collection stays cheap.
        from safedrug_train_dump import select_device

        device = str(select_device())
    else:
        device = args.device
    args.device = device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.drug_name_cache) if args.drug_name_cache else out_dir / "drug_name_list.txt"

    with open(args.hadm_ids_pkl, "rb") as fh:
        hadm_ids_nested = dill.load(fh, encoding="latin1")
    cohort_hadm_ids = flatten_hadm_ids(hadm_ids_nested)

    drug_names = load_or_build_drug_name_list(cache_path, args.prescriptions_csv, args.min_drug_rows, args.min_drug_len)
    diagnosis_by_hadm = load_admission_diagnosis(args.admissions_csv)
    cc_by_hadm = load_cc_text(args.cc_pickle)

    texts, has_text_flags = [], []
    n_diag_only = n_cc_only = n_both = n_neither = 0
    for hadm_id in cohort_hadm_ids:
        diag = diagnosis_by_hadm.get(hadm_id)
        cc = cc_by_hadm.get(hadm_id)
        text, has_text = compose_text(diag, cc, drug_names)
        texts.append(text)
        has_text_flags.append(has_text)
        if diag and cc:
            n_both += 1
        elif diag:
            n_diag_only += 1
        elif cc:
            n_cc_only += 1
        else:
            n_neither += 1

    n = len(cohort_hadm_ids)
    emb = np.zeros((n, TEXT_DIM), dtype=np.float32)
    nonzero_idx = [i for i, h in enumerate(has_text_flags) if h]
    if nonzero_idx:
        tokenizer, model = load_clinicalbert(args.device)
        nonzero_texts = [texts[i] for i in nonzero_idx]
        nonzero_emb = embed_texts(nonzero_texts, tokenizer, model, args.device, args.max_length, args.batch_size)
        for row, i in enumerate(nonzero_idx):
            emb[i] = nonzero_emb[row]

    np.savez(
        out_dir / "text_query.npz",
        HADM_ID=np.array(cohort_hadm_ids, dtype=np.int64),
        emb=emb,
        has_text=np.array(has_text_flags, dtype=np.int8),
        text=np.array(texts, dtype=object),
    )
    meta = {
        "n_visits": n,
        "n_has_text": int(sum(has_text_flags)),
        "n_diagnosis_and_cc": n_both,
        "n_diagnosis_only": n_diag_only,
        "n_cc_only": n_cc_only,
        "n_neither": n_neither,
        "model_id": "emilyalsentzer/Bio_ClinicalBERT",
        "text_dim": TEXT_DIM,
        "max_length": args.max_length,
        "n_drug_names": len(drug_names),
        "mean_words_per_text": float(np.mean([len(t.split()) for t in texts if t])) if any(texts) else 0.0,
    }
    (out_dir / "text_query_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[+] wrote text_query.npz for {n} visits ({sum(has_text_flags)} with text) to {out_dir}")


if __name__ == "__main__":
    main()
