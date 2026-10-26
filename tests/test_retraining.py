"""Integrity checks for the completed independent retraining record."""
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/retraining"

class RetrainingRecordTests(unittest.TestCase):
    def test_selection_and_training_provenance(self):
        run = json.loads((ROOT / "pipeline.json").read_text())
        self.assertEqual(run["status"], "completed")
        self.assertTrue(run["distributed_assets_unchanged"])
        names = [stage["name"] for stage in run["stages"]]
        self.assertLess(names.index("select-reaction"), names.index("download-test"))
        self.assertFalse(run["selection"]["test_data_used"])
        training = json.loads((ROOT / "training-summary.json").read_text())
        for row in training["stages"]:
            self.assertFalse(row["initialization"]["loaded_trained_head"])
        self.assertEqual(training["stages"][0]["optimizer_updates"], 1505)
        for row in training["stages"][1:]:
            self.assertEqual(row["source_updates"], 1003)
            self.assertEqual(row["trajectory_updates"], 503)

    def test_selected_weights_reach_full_evaluation(self):
        run = json.loads((ROOT / "pipeline.json").read_text())
        result = json.loads((ROOT / "evaluation.json").read_text())
        source = result["model_sources"]["retrained"]
        self.assertEqual(source["transport_sha256"],
                         run["selected_weights"]["transport"]["sha256"])
        self.assertEqual(source["reaction_sha256"],
                         run["selected_weights"]["reaction"]["sha256"])
        self.assertLessEqual(result["reference_metric_max_difference"], 1e-8)
        self.assertEqual(set(result["forward"]), {"C1", "GG", "retrained"})
        for layout in result["inverse"].values():
            self.assertEqual(len(layout["retrained"]["cases"]), 32)

    def test_agreement_record(self):
        comparison = json.loads((ROOT / "comparison.json").read_text())
        self.assertTrue(comparison["within_tolerance"])
        self.assertLessEqual(comparison["maximum_metric_difference"],
                             comparison["tolerance_absolute_relative_L2"])

if __name__ == "__main__":
    unittest.main()
