"""Read-only hash verification of original raw/prepared data after audit."""
import csv,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R=Path(__file__).resolve().parents[1]
S=R/'research/thesis_assets/phase4_duplicate_audit/v20261116T230847Z_34d5abf2'
N=R/'research/thesis_assets/phase4_duplicate_audit_revision/v20261116T233232Z_86ade2d0'
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def check(t):
 p,h=t;f=Path(p);f=f if f.is_absolute() else R/f
 try:
  digest=hashlib.sha256()
  with open(f,'rb') as src:
   for b in iter(lambda:src.read(1024*1024),b''):digest.update(b)
  return None if digest.hexdigest()==h else str(p)
 except Exception as e:return str(p)+': '+str(e)
sets={'raw_images':[(r['source'],r['sha256']) for r in read(R/'dataset/audit/raw_manifest.csv')],'prepared_images':[(r['path'],r['sha256']) for r in read(S/'prepared_image_preservation.csv')],'original_metadata_manifests':list(json.loads((S/'preservation_snapshot.json').read_text()).items())}
result={}
for name,items in sets.items():
 with ThreadPoolExecutor(max_workers=4) as pool:errors=[e for e in pool.map(check,items,buffersize=16) if e]
 result[name]={'checked':len(items),'mismatches':errors};print(name,len(items),'mismatches',len(errors),flush=True)
(N/'full_source_preservation_verification.json').write_text(json.dumps(result,indent=2)+'\n')
assert not any(r['mismatches'] for r in result.values())
