"""Fetch exactly two official metadata CSVs; never fetch images/thumbnails."""
from pathlib import Path
import urllib.request,json,hashlib,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/sys.argv[1];data=out/'source_data';data.mkdir(parents=True,exist_ok=True)
urls={'hiba_collection251_metadata.csv':'https://api.isic-archive.com/collections/251/metadata/','hiba_author_supplementary.csv':'https://raw.githubusercontent.com/piashiba/HIBASkinLesionsDataset/main/Data/SupplementaryData.csv'}
records=[]
for name,url in urls.items():
 p=data/name;assert not p.exists()
 req=urllib.request.Request(url,headers={'User-Agent':'MobileSkinXAI research metadata audit'})
 with urllib.request.urlopen(req,timeout=45) as r:
  assert not r.headers.get('Content-Type','').startswith('image/');b=r.read(5000001);assert len(b)<5000000;assert b',' in b[:10000] and not b.lstrip().startswith(b'<');p.write_bytes(b)
  records.append({'url':url,'saved':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'content_type':r.headers.get('Content-Type'),'created_UTC':datetime.now(timezone.utc).isoformat()})
(data/'hiba_download_receipt.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
