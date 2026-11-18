import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mobileskinxai-ablation-mpl')
import csv,json,hashlib,sys,time,copy
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'final_training').is_dir() and (p/'dataset').is_dir())
PREP=ROOT/'final_training/phase5a/v20261117T061722Z_f5055bd9';RELEASE=ROOT/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8'
PHASE6=ROOT/'research/thesis_assets/final_test_results/v20261117T103000Z_phase6_final'
CLASSES=['NV','MEL','BCC','SCC'];KEYS=['B1','B2','B3']
NAMES={'M3':'MobileNetV4-Conv-Small','M5':'MobileSkinXAI-Enhanced','B1':'MobileSkinXAI-Enhanced without reliability gate','B2':'MobileNetV4 with student-only consistency','B3':'Domain-robust MobileNetV4-Conv-Small'}
COLORS={'M3':'#F28E2B','M5':'#B07AA1','B1':'#0072B2','B2':'#009E73','B3':'#CC79A7'}
TITLE='MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.'
sys.path.insert(0,str(PREP/'code'));import core

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())
def write(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def checkfiles(folder,name):
 for line in (folder/name).read_text().splitlines():
  h,p=line.split('  ',1);assert sha(folder/p)==h,p

def sealcheck(out):
 receipt=load(out/'00_Protocol/PROTOCOL_LOCK.json')
 for p,h in receipt['sha256'].items():assert sha(out/p)==h,p
 return receipt
