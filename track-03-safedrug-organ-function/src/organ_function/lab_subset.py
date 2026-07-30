"""
Streams labevents.csv.gz (2.09GB, 122.1M rows) in chunks so the whole file
is never held in memory at once. hadm_id is intentionally NOT kept in the
output: ~41% of rows for these itemids have no hadm_id (outpatient/ED
draws) — see plan Background on why lookups join on (subject_id, charttime)
instead.
"""
import pandas as pd

from organ_function.lab_config import LAB_ITEMIDS

LAB_SUBSET_COLUMNS = [
    "subject_id",
    "itemid",
    "charttime",
    "valuenum",
    "ref_range_lower",
    "ref_range_upper",
]


def extract_lab_subset(csv_path: str, chunksize: int = 2_000_000) -> pd.DataFrame:
    """csv_path: path to labevents.csv or labevents.csv.gz (pandas infers
    gzip from the .gz suffix). Returns a single concatenated DataFrame with
    LAB_SUBSET_COLUMNS, sorted by (subject_id, charttime), keeping only rows
    whose itemid is a target lab and whose valuenum is not null."""
    target_itemids = set(LAB_ITEMIDS.values())
    kept_chunks = []
    reader = pd.read_csv(
        csv_path,
        usecols=["subject_id", "itemid", "charttime", "valuenum", "ref_range_lower", "ref_range_upper"],
        chunksize=chunksize,
    )
    for chunk in reader:
        filtered = chunk[chunk["itemid"].isin(target_itemids) & chunk["valuenum"].notna()]
        if len(filtered):
            kept_chunks.append(filtered)

    if not kept_chunks:
        return pd.DataFrame(columns=LAB_SUBSET_COLUMNS)

    result = pd.concat(kept_chunks, ignore_index=True)
    result["charttime"] = pd.to_datetime(result["charttime"])
    return (
        result[LAB_SUBSET_COLUMNS]
        .sort_values(["subject_id", "charttime"])
        .reset_index(drop=True)
    )
