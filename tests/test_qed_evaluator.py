from pathlib import Path


def test_qed_script_exists_and_uses_raw_valid_cohort():
    text = Path("scripts/evaluate_qed.py").read_text(encoding="utf-8")
    assert "metric_molecule_source" in text
    assert "raw_valid" in text
    assert "QED.WEIGHT_MEAN" in text
