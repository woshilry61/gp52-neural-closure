from pathlib import Path
import hashlib,json,re,unittest
ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
    def test_assets(self):
        manifest=json.loads((ROOT/"benchmarks/artifact-manifest.json").read_text())
        for name,row in manifest["assets"].items():
            with self.subTest(path=name):
                p=ROOT/name
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row["sha256"])
    def test_splits(self):
        p=json.loads((ROOT/"src/gp52/assets/splits.json").read_text())
        groups=[set(p[k])for k in ["train_seeds","development_seeds","fresh_seeds"]]
        self.assertEqual([len(g)for g in groups],[16,8,8])
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    def test_document_links(self):
        for p in [ROOT/"README.md",*list((ROOT/"docs").glob("*.md"))]:
            for link in re.findall(r"\]\(([^)]+)\)",p.read_text()):
                if "://" in link or link.startswith("#"):continue
                with self.subTest(document=str(p),link=link):self.assertTrue((p.parent/link.split("#")[0]).exists())
    def test_reproduction(self):
        p=json.loads((ROOT/"benchmarks/reproduction.json").read_text())
        self.assertEqual(p["status"],"complete")
        self.assertLess(p["reference_metric_max_difference"],1e-8)
if __name__=="__main__":unittest.main()
