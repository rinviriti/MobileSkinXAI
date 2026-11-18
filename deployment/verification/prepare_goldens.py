"""Materialize the four licensed, deidentified development fixtures locally.

Images are supplied once in test-only golden_cases.dart, never production assets.
No external download, held-out image or patient metadata is involved.
"""
from pathlib import Path
import json,base64,hashlib,argparse
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'.private');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 source=(ROOT/'app/flutter/integration_test/golden_cases.dart').read_text();payload=source.split('const goldenCases = ',1)[1];rows,_=json.JSONDecoder().raw_decode(payload);result=[]
 for r in rows:
  raw=base64.b64decode(r['jpeg_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==r['sha256'];dest=a.output/(r['image']+'.jpg')
  if dest.exists():assert dest.read_bytes()==raw
  else:dest.write_bytes(raw)
  result.append({'image':r['image'],'source':str(dest.resolve()),'source_sha256':r['sha256'],'original_logits':r['logits'],'split':'development_validation'})
 (a.output/'golden_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(a.output/'golden_manifest.json')
if __name__=='__main__':main()
