"""Integrity contracts for the current public experiment."""
from pathlib import Path
import csv,json,unittest
ROOT=Path(__file__).resolve().parents[1]
class UnifiedReleaseTests(unittest.TestCase):
 def test_shared_split(self):
  p=json.loads((ROOT/'configs/unified-cfd-baselines-v1.json').read_text())
  groups=[set(p[k]) for k in ['train','validation','test']]
  self.assertEqual([len(s) for s in groups],[64,16,32])
  self.assertEqual(set.union(*groups),set(range(112)))
  self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
 def test_asset_manifest(self):
  p=json.loads((ROOT/'data/unified-manifest.json').read_text())
  self.assertEqual(set(p['assets']),{'single','three','seven'})
  for row in p['assets'].values():
   self.assertEqual(len(row['sha256']),64)
   self.assertGreater(row['bytes'],0)
   self.assertLess(row['bytes'],2_000_000_000)
   entries=row.get('parts',[row])
   self.assertEqual(sum(e['bytes'] for e in entries),row['bytes'])
   for e in entries:self.assertTrue(e['url'].endswith('/'+e['file']))
 def test_replay_coverage(self):
  p=json.loads((ROOT/'benchmarks/unified-release-validation.json').read_text())
  self.assertEqual(p['status'],'passed')
  self.assertEqual(len(p['records']),12)
  self.assertEqual(sum(r['forward_cases'] for r in p['records']),384)
  self.assertEqual(sum(r['inverse_fits'] for r in p['records']),144)
  for r in p['records']:
   self.assertLessEqual(r['forward_metric_max_abs_difference'],p['forward_tolerance'])
   self.assertLessEqual(r['inverse_metric_max_abs_difference'],p['inverse_tolerance'])
if __name__=='__main__':unittest.main()
