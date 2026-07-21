import textwrap

from HEIDR.service_personalization.ndc_atc_mapping import (
    normalize_ndc,
    load_ndc2rxcui,
    load_rxcui2atc3,
    ndc_to_atc3,
)


def test_normalize_ndc_strips_dashes_and_pads():
    assert normalize_ndc("0002-1433-61") == "00002143361"
    assert normalize_ndc("00591083960") == "00591083960"
    assert normalize_ndc(591083960) == "00591083960"


def test_load_ndc2rxcui_parses_dict_literal(tmp_path):
    p = tmp_path / "ndc2rxnorm_mapping.txt"
    p.write_text("{'00591083960': u'12345', '00904404073': u'67890'}")
    mapping = load_ndc2rxcui(str(p))
    assert mapping["00591083960"] == "12345"
    assert mapping["00904404073"] == "67890"


def test_load_rxcui2atc3_takes_first_four_chars(tmp_path):
    p = tmp_path / "ndc2atc_level4.csv"
    p.write_text(textwrap.dedent("""\
        YEAR,MONTH,NDC,RXCUI,ATC4
        2014,11,0002-1433-61,12345,A10BJ01
        2014,11,0002-1434-61,67890,C01BA02
    """))
    mapping = load_rxcui2atc3(str(p))
    assert mapping["12345"] == "A10B"
    assert mapping["67890"] == "C01B"


def test_ndc_to_atc3_end_to_end():
    ndc2rxcui = {"00591083960": "12345"}
    rxcui2atc3 = {"12345": "A10B"}
    assert ndc_to_atc3("00591083960", ndc2rxcui, rxcui2atc3) == "A10B"


def test_ndc_to_atc3_returns_none_when_unmapped():
    assert ndc_to_atc3("99999999999", {}, {}) is None
