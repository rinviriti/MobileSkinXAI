"""Join public frozen image digests with YOUR licensed local audit index.

No split regeneration and no medical metadata publication. Output is private.
This reconstructs membership semantics, not original byte-for-byte JSON seals.
"""
from pathlib import Path
import csv,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--local-index',type=Path,required=True);a=p.parse_args()
 rows=list(csv.DictReader(a.local_index.open()));index={hashlib.sha256((r['dataset']+':'+r['image']).encode()).hexdigest():r for r in rows};public=list(csv.DictReader((ROOT/'training/configs/frozen_split_checksums.csv').open()));result=[]
 for r in public:
  q=index.get(r['image_key_sha256']);assert q is not None,'Missing licensed local image; do not regenerate splits'
  source=Path(q['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_sha256'],'Original image checksum mismatch'
  result.append({**q,'study_split':r['split'],'expected_processed_sha256':r['processed_sha256'],'use_in_lesion_probability_mean':r['use_in_lesion_probability_mean']})
 out=ROOT/'.private';out.mkdir(exist_ok=True);target=out/'restored_membership.csv'
 if target.exists():raise FileExistsError('Private manifest exists; preserve/version it before rerunning')
 with target.open('w',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in result for k in r)));w.writeheader();w.writerows(result)
 print(json.dumps({'images':len(result),'private_output':str(target),'split_regenerated':False,'data_approval_granted':False},indent=2))
if __name__=='__main__':main()
