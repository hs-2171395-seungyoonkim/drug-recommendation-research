from safedrug.subgroups import is_liver_dysfunction, is_renal_dysfunction


def _visit(**overrides):
    base = {
        "creatinine_deviation": 0.0,
        "albumin_deviation": 0.0,
        "bilirubin_total_deviation": 0.0,
    }
    base.update(overrides)
    return base


def test_renal_dysfunction_true_when_creatinine_above_range():
    assert is_renal_dysfunction(_visit(creatinine_deviation=0.5)) is True


def test_renal_dysfunction_false_when_within_or_below_range():
    assert is_renal_dysfunction(_visit(creatinine_deviation=0.0)) is False
    assert is_renal_dysfunction(_visit(creatinine_deviation=-0.3)) is False


def test_renal_dysfunction_false_when_deviation_is_nan_missing():
    assert is_renal_dysfunction(_visit(creatinine_deviation=float("nan"))) is False


def test_liver_dysfunction_true_when_albumin_low_or_bilirubin_high():
    assert is_liver_dysfunction(_visit(albumin_deviation=-0.2)) is True
    assert is_liver_dysfunction(_visit(bilirubin_total_deviation=0.4)) is True


def test_liver_dysfunction_false_when_both_normal():
    assert is_liver_dysfunction(_visit()) is False


def test_liver_dysfunction_false_when_both_nan():
    assert (
        is_liver_dysfunction(
            _visit(albumin_deviation=float("nan"), bilirubin_total_deviation=float("nan"))
        )
        is False
    )
