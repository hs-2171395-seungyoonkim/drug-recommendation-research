"""Acute/chronic diagnosis labels per transition and their association with
medication change (proposal section 3.3-B).

Label sources (all local, no download):
  ICD-9-CM  chronic flag : HCUP CCI 2015  (icdmappings data_files/cci2015.csv)
  ICD-10-CM chronic flag : HCUP CCIR v2023.1 (icdmappings data_files/CCIR_v2023-1.csv)
  ICD-9-CM  category     : HCUP single-level CCS 2015 (unfair/ref/ccs_dxref_2015.csv)
  ICD-10-CM category     : HCUP CCSR default category (dx_cat1_mapping.json; codes only)

"acute" here means CCI/CCIR = 0 (not chronic). Neither tool has an acute
category: CCIR's beta 4-way split was withdrawn in v2023.1, so this is the
only externally standardised binary available for both code systems.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILES = Path(r"C:\Python314\Lib\site-packages\icdmappings\data_files")
CCI9_PATH = DATA_FILES / "cci2015.csv"
CCIR10_PATH = DATA_FILES / "CCIR_v2023-1.csv"
CCSR10_PATH = DATA_FILES / "ICD10_CM_CCSR" / "dx_cat1_mapping.json"
CCS9_REF = Path(r"C:\Users\Administrator\Desktop\unfair\ref\ccs_dxref_2015.csv")

_CCI9_LINE = re.compile(r"^'([^']*)',(.*),'([01])','(\d+)'\s*$")
_CCIR_LINE = re.compile(r"^'([^']*)',(.*),([019])\s*$")


def _strip(s: str) -> str:
    return s.strip().strip("'").strip('"').strip()


def load_cci9(path: Path = CCI9_PATH) -> dict[str, tuple[int, str]]:
    """ICD-9-CM code (no dot) -> (chronic 0/1, description)."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f)  # header
        for line in f:
            m = _CCI9_LINE.match(line.rstrip("\n"))
            if not m:
                continue
            out[_strip(m.group(1))] = (int(m.group(3)), _strip(m.group(2)))
    if len(out) < 10000:
        raise ValueError(f"cci2015 parse looks wrong: {len(out)} codes")
    return out


def load_ccir10(path: Path = CCIR10_PATH) -> dict[str, tuple[int | None, str]]:
    """ICD-10-CM code (no dot) -> (chronic 0/1 or None for '9', description)."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _CCIR_LINE.match(line.rstrip("\n"))
            if not m or m.group(1).startswith("ICD-10"):
                continue
            flag = m.group(3)
            out[_strip(m.group(1))] = (None if flag == "9" else int(flag), _strip(m.group(2)))
    if len(out) < 50000:
        raise ValueError(f"CCIR parse looks wrong: {len(out)} codes")
    return out


def load_ccs9(path: Path = CCS9_REF) -> dict[str, tuple[str, str]]:
    """ICD-9-CM code -> (CCS category id, CCS category description)."""
    out = {}
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f, quotechar="'"):
            if len(row) < 3 or not row[0].strip() or row[0].startswith("NOTE") or row[0].startswith("ICD-9"):
                continue
            out[_strip(row[0])] = (_strip(row[1]), _strip(row[2]))
    if len(out) < 10000:
        raise ValueError(f"ccs dxref parse looks wrong: {len(out)} codes")
    return out


def load_ccsr10(path: Path = CCSR10_PATH) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class CodeLabeler:
    """chronic(code, version) -> 1 / 0 / None ; category(...) -> str ; describe(...) -> str."""

    def __init__(self):
        self.cci9 = load_cci9()
        self.ccir10 = load_ccir10()
        self.ccs9 = load_ccs9()
        self.ccsr10 = load_ccsr10()

    def chronic(self, code: str, version: int):
        table = self.cci9 if version == 9 else self.ccir10
        hit = table.get(code)
        return None if hit is None else hit[0]

    def describe(self, code: str, version: int) -> str:
        table = self.cci9 if version == 9 else self.ccir10
        hit = table.get(code)
        return hit[1] if hit else ""

    def category(self, code: str, version: int) -> str:
        if version == 9:
            hit = self.ccs9.get(code)
            return f"CCS {hit[0]} {hit[1]}" if hit else f"ICD9 {code[:3]}"
        cat = self.ccsr10.get(code)
        return f"CCSR {cat}" if cat else f"ICD10 {code[:3]}"


def split_code(token: str) -> tuple[str, int]:
    """'4019' -> ('4019', 9); '4019_9' -> ('4019', 9); 'I10_10' -> ('I10', 10)."""
    if "_" in token:
        code, ver = token.rsplit("_", 1)
        return code, int(ver)
    return token, 9


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def is_status_code(code: str, version: int) -> bool:
    """Status / history / external-cause codes that CCI marks 'not chronic' but
    that are not acute events either: ICD-9 V- and E-codes, ICD-10 Z-codes."""
    head = code[:1].upper()
    return (version == 9 and head in ("V", "E")) or (version == 10 and head == "Z")


def transition_features(visits: list[dict], labeler: CodeLabeler) -> list[dict]:
    """One row per transition (visit i >= 1) of one patient.

    Each visit dict: {"hadm": id, "dx": set[(code, version)], "meds": set[int],
    "principal": (code, version) | None, "admit": Timestamp | None,
    "emergency": bool | None}. Visits must be in the order the caller wants
    "previous" to mean (chronological, unless testing the SafeDrug order).
    """
    rows = []
    history: set = set()
    for i, v in enumerate(visits):
        if i >= 1:
            prev = visits[i - 1]
            cur_dx, prev_dx = v["dx"], prev["dx"]
            flags = {c: labeler.chronic(*c) for c in cur_dx}
            new_prev = cur_dx - prev_dx
            new_hist = cur_dx - history
            added = v["meds"] - prev["meds"]
            stopped = prev["meds"] - v["meds"]
            pflag = labeler.chronic(*v["principal"]) if v["principal"] else None
            gap = None
            if v.get("admit") is not None and prev.get("admit") is not None:
                gap = (v["admit"] - prev["admit"]).total_seconds() / 86400.0
            status = {c for c in cur_dx if is_status_code(*c)}
            rows.append({
                "hadm": v["hadm"],
                "visit_pos": i,
                "n_dx": len(cur_dx),
                "n_status": len(status),
                "n_acute_status": sum(1 for c in status if flags[c] == 0),
                "n_new_status_prev": len(new_prev & status),
                "n_new_acute_excl_status_prev": sum(1 for c in new_prev - status if flags[c] == 0),
                "n_new_acute_excl_status_hist": sum(1 for c in new_hist - status if flags[c] == 0),
                "n_acute": sum(1 for c in cur_dx if flags[c] == 0),
                "n_chronic": sum(1 for c in cur_dx if flags[c] == 1),
                "n_unknown": sum(1 for c in cur_dx if flags[c] is None),
                "n_new_prev": len(new_prev),
                "n_new_acute_prev": sum(1 for c in new_prev if flags[c] == 0),
                "n_new_chronic_prev": sum(1 for c in new_prev if flags[c] == 1),
                "n_new_unknown_prev": sum(1 for c in new_prev if flags[c] is None),
                "n_new_hist": len(new_hist),
                "n_new_acute_hist": sum(1 for c in new_hist if flags[c] == 0),
                "n_new_chronic_hist": sum(1 for c in new_hist if flags[c] == 1),
                "n_dropped_prev": len(prev_dx - cur_dx),
                "principal_chronic": pflag,
                "principal_new_prev": (v["principal"] not in prev_dx) if v["principal"] else None,
                "principal_new_hist": (v["principal"] not in history) if v["principal"] else None,
                "n_cur_meds": len(v["meds"]),
                "n_prev_meds": len(prev["meds"]),
                "n_added": len(added),
                "n_stopped": len(stopped),
                "prev_cur_jaccard": jaccard(prev["meds"], v["meds"]),
                "gap_days": gap,
                "emergency": v.get("emergency"),
                "new_acute_codes": sorted(c for c in new_prev - status if flags[c] == 0),
                "added_meds": sorted(added),
            })
        history |= v["dx"]
    return rows


# ---------------------------------------------------------------- statistics

def spearman(x, y) -> float:
    x = pd.Series(np.asarray(x, dtype=float)).rank()
    y = pd.Series(np.asarray(y, dtype=float)).rank()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ols_standardized(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Standardised betas (columns of X and y z-scored) and R^2 with intercept."""
    Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    ys = (y - y.mean()) / (y.std() if y.std() else 1)
    A = np.column_stack([np.ones(len(ys)), Xs])
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ beta
    r2 = 1 - float(resid @ resid) / float(ys @ ys) if float(ys @ ys) else float("nan")
    return beta[1:], r2


def cluster_bootstrap(stat_fn, df: pd.DataFrame, cluster_col: str, n_boot: int = 200, seed: int = 0):
    """Resample clusters (patients) with replacement; stat_fn(df) -> scalar or 1-D array."""
    rng = np.random.default_rng(seed)
    groups = df.groupby(cluster_col, sort=False).indices
    keys = list(groups)
    point = np.atleast_1d(np.asarray(stat_fn(df), dtype=float))
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        draws.append(np.atleast_1d(np.asarray(stat_fn(df.iloc[idx]), dtype=float)))
    draws = np.vstack(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {"point": point.tolist(), "ci_low": lo.tolist(), "ci_high": hi.tolist(),
            "n_boot": n_boot, "n_clusters": len(keys)}
