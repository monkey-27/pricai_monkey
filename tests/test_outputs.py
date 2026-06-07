from pathlib import Path

from pact.run_eval import OUTPUT_DIR, run


def test_output_files_are_created():
    run(methods="PACTFull", audit=False)
    assert (OUTPUT_DIR / "predictions.csv").exists()
    assert (OUTPUT_DIR / "metrics.json").exists()
    assert (OUTPUT_DIR / "audit_report.md").exists()
    assert Path(OUTPUT_DIR).is_dir()

