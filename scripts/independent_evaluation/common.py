import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mobileskinxai-phase6-mpl')
import csv,json,hashlib,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RELEASE=ROOT/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8'
PACKAGE=ROOT/'research/thesis_assets/final_training_results/v20261117T091430Z_704ed3cc'
PREP=ROOT/'final_training/phase5a/v20261117T061722Z_f5055bd9'
CONTROL=ROOT/'final_training/m5_lr_control/v20261117T075145Z_df7c70a4'
CLASSES=['NV','MEL','BCC','SCC']
KEYS=['M1','M2','M3','M4','M5','M5-LR']
NAMES=dict(zip(KEYS,['MobileNetV3-Small','EfficientNet-Lite0','MobileNetV4-Conv-Small','ConvNeXt-Tiny','MobileSkinXAI-Enhanced','MobileSkinXAI-Enhanced (LR Control)']))
COLORS=dict(zip(KEYS,['#4E79A7','#76B7B2','#F28E2B','#9C755F','#B07AA1','#E15759']))
ROLES=dict(zip(KEYS,['lightweight comparator','lightweight comparator','main lightweight baseline','larger reference','primary enhanced candidate','secondary enhanced candidate']))
TITLE='MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())
def check_list(root,name):
 lines=(root/name).read_text().splitlines()
 for line in lines:
  h,p=line.split('  ',1);assert sha(root/p)==h,root/p
 return len(lines)
def lockcheck(out):
 out=Path(out);receipt=load(out/'LOCK_RECEIPT.json');assert sha(out/'MODEL_SELECTION_LOCK.json')==receipt['sha256'];assert (out/'MODEL_SELECTION_LOCK.json').stat().st_mode&0o222==0
 return load(out/'MODEL_SELECTION_LOCK.json')
def model_info():
 rows=read(PACKAGE/'11_Source_Data/model_metrics.csv');out={}
 for r in rows:
  k=r['model'];cfg=CONTROL/'configs/M5.json' if k=='M5-LR' else PREP/f'configs/{k}.json'
  out[k]={'name':NAMES[k],'role':ROLES[k],'checkpoint':r['original_checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],'configuration':str(cfg.relative_to(ROOT)),'configuration_sha256':sha(cfg),'config':load(cfg),'best_epoch':int(r['best_epoch'])}
 return out
