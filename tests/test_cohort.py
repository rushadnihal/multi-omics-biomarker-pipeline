from mosurv.cohort import collapse_stage, pick_one_per_patient


def test_pick_one_per_patient():
    ids = ["TCGA-AA-0001-01B", "TCGA-AA-0001-01A", "TCGA-AA-0002-11A", "TCGA-AA-0003-01Z",
           "TCGA-AA-0004-02A", "TCGA-AA-0005-01A"]
    ch, ndup = pick_one_per_patient(ids, "01", ["Z"])
    assert ch == {"TCGA-AA-0001": "TCGA-AA-0001-01A", "TCGA-AA-0005": "TCGA-AA-0005-01A"}
    assert ndup == 1


def test_collapse_stage():
    assert collapse_stage("Stage IA") == "I"
    assert collapse_stage("Stage IIB") == "II"
    assert collapse_stage("Stage IIIA") == "III"
    assert collapse_stage("Stage IV") == "IV"
    assert collapse_stage("[Discrepancy]") is None
    assert collapse_stage(None) is None
