import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/camacl-mpl')
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib,random,sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'final_training').is_dir() and (p/'dataset').is_dir())
RELEASE=ROOT/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8'
ARCH='mobilenetv4_conv_small.e2400_r224_in1k';CLASSES=['NV','MEL','BCC','SCC']
PRETRAIN=ROOT/'final_training/pretrained/M3'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())
def seed_all(s):
 import torch
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
