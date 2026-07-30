from organ_function.lab_config import (
    LAB_ITEMIDS,
    LAB_NAMES,
    ITEMID_TO_NAME,
    KIDNEY_FUNCTION_LABS,
    LIVER_RELATED_LABS,
)


def test_lab_itemids_match_verified_labevents_counts():
    assert LAB_ITEMIDS == {
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


def test_lab_names_matches_dict_key_order():
    assert LAB_NAMES == list(LAB_ITEMIDS.keys())


def test_itemid_to_name_is_exact_inverse():
    assert ITEMID_TO_NAME == {v: k for k, v in LAB_ITEMIDS.items()}


def test_kidney_and_liver_groupings_are_disjoint_and_use_organ_function_terms():
    assert KIDNEY_FUNCTION_LABS == ["creatinine", "bun"]
    assert LIVER_RELATED_LABS == ["bilirubin_total", "albumin", "alt", "ast"]
    assert set(KIDNEY_FUNCTION_LABS).isdisjoint(LIVER_RELATED_LABS)
    assert set(KIDNEY_FUNCTION_LABS) | set(LIVER_RELATED_LABS) <= set(LAB_NAMES)
