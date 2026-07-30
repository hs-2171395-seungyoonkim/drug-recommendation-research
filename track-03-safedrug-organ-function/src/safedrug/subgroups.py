"""
Renal/Liver Dysfunction subgroup membership, computed from
organ_function_features.pkl's RAW (pre-standardization) _deviation values -
not vectorize.py's standardized output - so imputation/standardization
never shifts subgroup boundaries (design spec §4/§6).
"""


def _is_nan(x) -> bool:
    return x != x  # NaN is the only float that is not equal to itself


def is_renal_dysfunction(visit_features: dict) -> bool:
    deviation = visit_features["creatinine_deviation"]
    return not _is_nan(deviation) and deviation > 0


def is_liver_dysfunction(visit_features: dict) -> bool:
    albumin_deviation = visit_features["albumin_deviation"]
    bilirubin_deviation = visit_features["bilirubin_total_deviation"]
    albumin_low = not _is_nan(albumin_deviation) and albumin_deviation < 0
    bilirubin_high = not _is_nan(bilirubin_deviation) and bilirubin_deviation > 0
    return albumin_low or bilirubin_high
