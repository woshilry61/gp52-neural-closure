"""Integrity checks for the independently executed raw-input reproduction."""
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "benchmarks/raw-reproduction"

def read(name):
    return json.loads((RECORDS / name).read_text())

class RawReproductionTests(unittest.TestCase):
    def test_complete_dependency_chain(self):
        run = read("pipeline.json")
        self.assertEqual(run["status"], "completed")
        stages = run["stages"]
        names = [row["name"] for row in stages]
        for before, after in [
            ("fetch-sen", "extract"), ("extract", "cfd-flow"),
            ("cfd-flow", "extend-flow"), ("extend-flow", "blood-observations"),
            ("blood-observations", "train-blood"), ("train-blood", "prepare-assets"),
            ("cfd-labels", "export-data"), ("prepare-assets", "export-data"),
            ("export-data", "biochemical-retraining"), ("biochemical-retraining", "validate")]:
            self.assertLess(names.index(before), names.index(after))
        self.assertTrue(all(row["returncode"] == 0 for row in stages))
        self.assertTrue(read("validation.json")["passed"])

    def test_generated_inputs_and_physics(self):
        report = read("validation.json")
        self.assertEqual(report["cfd_chemistry_cases"], 32)
        self.assertTrue(report["original_assets_unchanged"])
        self.assertLessEqual(report["maximum_data_difference"],
                             report["thresholds"]["data_absolute"])
        self.assertLessEqual(report["maximum_prepared_input_difference"],
                             report["thresholds"]["prepared_inputs_absolute"])
        self.assertLess(report["blood_continuity_residual"], 1e-10)
        check = read("cfd.json")["poiseuille_check"]
        self.assertLess(check["velocity_relative_l2"], 1e-10)
        self.assertLess(check["pressure_relative_l2"], 1e-8)
        partitions = read("data-manifest.json")
        self.assertEqual([partitions[n + ".npz"]["cases"]
                          for n in ["train", "development", "test"]], [16, 8, 8])

    def test_new_weights_and_held_out_evaluation(self):
        run = read("biochemical-pipeline.json")
        self.assertEqual(run["data_source"], "regenerated")
        names = [row["name"] for row in run["stages"]]
        self.assertLess(names.index("select-reaction"), names.index("stage-test"))
        self.assertFalse(run["selection"]["test_data_used"])
        result = read("evaluation.json")
        source = result["model_sources"]["retrained"]
        for kind in ["transport", "reaction"]:
            self.assertEqual(source[kind + "_sha256"],
                             run["selected_weights"][kind]["sha256"])
        for group in result["inverse"].values():
            self.assertEqual(len(group["retrained"]["cases"]), 32)
        for row in read("training-summary.json")["stages"]:
            self.assertFalse(row["initialization"]["loaded_trained_head"])
        report = read("validation.json")
        self.assertLessEqual(report["maximum_metric_difference"],
                             report["thresholds"]["relative_L2_metric_absolute"])

if __name__ == "__main__":
    unittest.main()
