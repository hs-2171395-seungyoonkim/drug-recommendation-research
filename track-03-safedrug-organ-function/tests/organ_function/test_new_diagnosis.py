from organ_function.new_diagnosis import has_new_diagnosis


def test_new_code_detected():
    assert has_new_diagnosis([5, 6], {1, 2, 5}) is True


def test_no_new_code():
    assert has_new_diagnosis([1, 2], {1, 2, 5}) is False


def test_first_visit_empty_history_is_not_flagged_new():
    # No prior visits to compare against yet — defined as False, not True,
    # so the flag means "newly appeared relative to this patient's own
    # history", not "trivially true because history is empty".
    assert has_new_diagnosis([1, 2], set()) is False
