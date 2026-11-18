"""Shared definitions for a validation-only, versioned publication package."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mobileskinxai-publication-matplotlib')
import csv,hashlib,io,json,shutil,sys,textwrap
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
ROOT=Path(os.environ['MOBILESKINXAI_PROJECT_ROOT']).resolve() if os.environ.get('MOBILESKINXAI_PROJECT_ROOT') else next(p for p in Path(__file__).resolve().parents if (p/'final_training').is_dir() and (p/'dataset').is_dir())
PREP=ROOT/'final_training/phase5a/v20261117T061722Z_f5055bd9'
BASE=ROOT/'final_training/phase5b/v20261117T063340Z_d20cbfaa'
CONTROL=ROOT/'final_training/m5_lr_control/v20261117T075145Z_df7c70a4'
RELEASE=ROOT/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8'
TITLE='MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.'
CLASSES=['NV','MEL','BCC','SCC'];KEYS=['M1','M2','M3','M4','M5','M5-LR']
NAMES=dict(zip(KEYS,['MobileNetV3-Small','EfficientNet-Lite0','MobileNetV4-Conv-Small','ConvNeXt-Tiny','MobileSkinXAI-Enhanced','MobileSkinXAI-Enhanced (LR Control)']))
COLORS=dict(zip(KEYS,['#4E79A7','#76B7B2','#F28E2B','#9C755F','#B07AA1','#E15759']))
CLASS_COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
CATEGORIES=['00_Executive_Summary','01_Dataset_Analysis','02_Model_Architectures','03_Training_Curves','04_Confusion_Matrices','05_ROC_PR_Curves','06_Classwise_Performance','07_Model_Comparisons','08_Model_Complexity','09_M5_Enhancement_Analysis','10_Explainability','11_Source_Data','12_Publication_Ready']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p):return str(Path(p).relative_to(ROOT))
def read_csv(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows);fields=fields or list(dict.fromkeys(k for row in rows for k in row));stream=io.StringIO(newline='');w=csv.DictWriter(stream,fields);w.writeheader();w.writerows(rows);data=stream.getvalue()
 if p.exists():
  assert p.read_bytes()==data.encode(),f'Existing source data differs: {p}'
 else:p.write_bytes(data.encode())
 return p
def write_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(x,indent=2,allow_nan=False)+'\n'
 if p.exists():assert p.read_text()==data,f'Existing JSON differs: {p}'
 else:p.write_text(data)
 return p
def load_context(out):
 out=Path(out);out=out if out.is_absolute() else ROOT/out
 original=json.loads((BASE/'final_results.json').read_text());control=json.loads((CONTROL/'comparison_results.json').read_text());models={r['model']:r.copy() for r in original['models']};models['M5-LR']=next(r.copy() for r in control['models'] if r['model_key']=='LR')
 for k,r in models.items():
  r['name']=NAMES[k];r['run']=CONTROL/'run' if k=='M5-LR' else PREP/f'runs/{k}_seed42';r['config_path']=CONTROL/'configs/M5.json' if k=='M5-LR' else PREP/f'configs/{k}.json';r['config']=json.loads(r['config_path'].read_text());r['best_epoch']=r.get('best_epoch',r.get('best_additional_or_total_epoch'));r['checkpoint']=ROOT/r['checkpoint'];r['history']=[json.loads(l) for l in (r['run']/'history.jsonl').read_text().splitlines()]
 manifest=read_csv(RELEASE/'dataset_manifest.csv');valid={r['image']:r for r in manifest if r['release_eligible']=='True' and r['split']=='validation'}
 return {'out':out,'data':out/'11_Source_Data','models':models,'manifest':manifest,'validation':valid}
def source(c,name):return c['data']/name
def full_name(k,width=34):return '\n'.join(textwrap.wrap(NAMES[k],width=width))
def horizontal(c,values,title,xlabel,percent=False):
 fig,ax=plt.subplots(figsize=(11,5.7),layout='constrained');y=np.arange(6);bars=ax.barh(y,values,color=[COLORS[k] for k in KEYS]);ax.set_yticks(y,[full_name(k) for k in KEYS]);ax.invert_yaxis();ax.set(xlabel=xlabel,title=title)
 if percent:
  ax.set_xlim(0,1.08);ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=4,fontsize=9)
 else:
  ax.set_xlim(0,max(values)*1.22 if max(values)>0 else 1);ax.bar_label(bars,labels=[f'{v:,.2f}' for v in values],padding=4,fontsize=9)
 return fig,ax
def save_figure(c,category,name,fig,title,caption,sources,split='validation',models=None,method='',script='plots.py'):
 folder=c['out']/category;stem=f'{category[:2]}_{name}';paths={ext:folder/f'{stem}.{ext}' for ext in ['png','svg','pdf']};log=c['data']/'figure_records.jsonl'
 records=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else [];existing=next((r for r in records if r['figure_id']==stem),None)
 if existing:
  assert all(p.exists() for p in paths.values());plt.close(fig);return existing
 assert not any(p.exists() for p in paths.values()),f'Partial figure exists: {stem}'
 for ext,p in paths.items():fig.savefig(p,dpi=300,bbox_inches='tight');shutil.copyfile(p,c['out']/'12_Publication_Ready'/p.name)
 plt.close(fig)
 srclist=[str(Path(s).relative_to(c['out'])) for s in sources]
 assert all((c['out']/s).is_file() for s in srclist),srclist
 row={'figure_id':stem,'category':category,'title':title,'caption':caption,'dataset_split':split,'models':';'.join(NAMES[k] for k in (models or [])),'source_data':';'.join(srclist),'source_data_sha256':json.dumps({s:sha(c['out']/s) for s in srclist},sort_keys=True),'generation_script':'11_Source_Data/generation_scripts/'+script,'generation_method':method,'png_path':str(paths['png'].relative_to(c['out'])),'svg_path':str(paths['svg'].relative_to(c['out'])),'pdf_path':str(paths['pdf'].relative_to(c['out'])),'publication_png':'12_Publication_Ready/'+paths['png'].name,'publication_svg':'12_Publication_Ready/'+paths['svg'].name,'publication_pdf':'12_Publication_Ready/'+paths['pdf'].name,'png_dpi':300,'heldout_test_PAD_evaluated':False}
 with log.open('a') as f:f.write(json.dumps(row)+'\n')
 print('FIGURE '+stem,flush=True);return row
