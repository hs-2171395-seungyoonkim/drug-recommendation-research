def has_new_diagnosis(current_diag_ids: list[int], prior_diag_ids_seen: set[int]) -> bool:
    """current_diag_ids: current visit's diag_voc ids (adm[0] in
    records_final2.pkl). prior_diag_ids_seen: union of diag_voc ids from
    ALL strictly earlier visits of this patient. Empty set (patient's first
    visit) always returns False — there is no baseline yet to compare
    against."""
    if not prior_diag_ids_seen:
        return False
    return any(d not in prior_diag_ids_seen for d in current_diag_ids)
