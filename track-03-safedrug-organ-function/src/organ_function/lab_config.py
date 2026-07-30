"""
Laboratory-derived Organ Function feature definitions.

itemids below were confirmed against the real 2.09GB
data/raw_labs/labevents.csv.gz (122,103,668 rows) during planning — see
docs/plans/2026-07-30-organ-function-lab-pipeline.md,
"Confirmed lab itemids" table, for the row-count evidence behind each choice
over its duplicate/legacy itemid.
"""

LAB_ITEMIDS: dict[str, int] = {
    "creatinine": 50912,
    "bun": 51006,
    "alt": 50861,
    "ast": 50878,
    "bilirubin_total": 50885,
    "albumin": 50862,
    "inr": 51237,
    "pt": 51274,
    "potassium": 50971,
    "platelet": 51265,
    "wbc": 51301,
    "glucose": 50931,
}

LAB_NAMES: list[str] = list(LAB_ITEMIDS.keys())

# Exported for the downstream model/evaluation project (Plan B): label display
# when inspecting raw labevents rows. Nothing in this package consumes it.
ITEMID_TO_NAME: dict[int, str] = {v: k for k, v in LAB_ITEMIDS.items()}

# Kidney function markers. Exported for the downstream model/evaluation
# project's Renal Dysfunction subgroup definition; unused inside this package.
KIDNEY_FUNCTION_LABS: list[str] = ["creatinine", "bun"]

# Liver-related markers. Exported for the downstream model/evaluation project's
# Liver Dysfunction subgroup definition; unused inside this package. ALT/AST specifically reflect hepatocellular injury
# (not liver synthetic/excretory function) — bilirubin and albumin reflect
# liver function more directly. Kept in one list because both groups are
# used identically downstream (see Plan B's Liver Dysfunction subgroup).
LIVER_RELATED_LABS: list[str] = ["bilirubin_total", "albumin", "alt", "ast"]
