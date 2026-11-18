"""Offline publication integrity checks; no ML training or private data access."""
from pathlib import Path
import ast,csv,json,hashlib,re,math,sys
ROOT=Path(__file__).resolve().parents[1]
MODEL='app/flutter/assets/models/e2_lr_control_v20261118/model_float32.tflite'
SHA='8b6e19f3d8d0dd9c3b798a2cf99c0525079b9de4ea2f5d1708dd928a718412f3'
def rows(p):
 with (ROOT/p).open(newline='') as f:return list(csv.DictReader(f))
def main():
 errors=[];python_count=0
 for p in ROOT.rglob('*.py'):
  if any(x in p.parts for x in ['.private','.dart_tool','build','Pods','.git']):continue
  try:ast.parse(p.read_text(),filename=str(p))
  except SyntaxError as e:errors.append(f'Python syntax: {p.relative_to(ROOT)}:{e.lineno}')
  python_count+=1
 model=ROOT/MODEL;assert hashlib.sha256(model.read_bytes()).hexdigest()==SHA
 assert model.read_bytes()[4:8]==b'TFL3' and model.stat().st_size==10073864
 metadata=json.loads((model.parent/'model_metadata.json').read_text());assert metadata['classes']==['NV','MEL','BCC','SCC'] and metadata['output']=='logits' and metadata['input_shape']==[1,224,224,3]
 pub=(ROOT/'app/flutter/pubspec.yaml').read_text();assert '    - assets/models/e2_lr_control_v20261118/' in pub
 provenance=json.loads((ROOT/'deployment/verification/MODEL_PROVENANCE.json').read_text());assert hashlib.sha256((ROOT/'deployment/verification/student_reference.pt').read_bytes()).hexdigest()==provenance['pytorch_reference_sha256']
 cms={}
 for r in rows('evaluation/source_data/CONFUSION_COUNTS.csv'):
  key=(r['model'],r['condition'],r['dataset'],r['analysis_level']);a=cms.setdefault(key,[[0]*4 for _ in range(4)]);labels=['NV','MEL','BCC','SCC'];a[labels.index(r['true_class'])][labels.index(r['predicted_class'])]=int(r['count'])
 metrics=rows('evaluation/source_data/ALL_SETTING_METRICS.csv')
 for r in metrics:
  a=cms[(r['model'],r['condition'],r['dataset'],r['analysis_level'])];supports=[sum(row) for row in a];preds=[sum(a[i][j] for i in range(4)) for j in range(4)];n=sum(supports);rec=[a[j][j]/supports[j] for j in range(4)];pre=[a[j][j]/preds[j] if preds[j] else 0 for j in range(4)];f=[2*x*y/(x+y) if x+y else 0 for x,y in zip(rec,pre)]
  derived={'accuracy':sum(a[j][j] for j in range(4))/n,'macro_f1':sum(f)/4,'balanced_accuracy':sum(rec)/4,'macro_precision':sum(pre)/4,'MEL_recall':rec[1],'SCC_recall':rec[3]}
  for k,v in derived.items():assert abs(v-float(r[k]))<1e-12,(r['model'],k)
  if r['dataset']=='PAD' and r['analysis_level']=='lesion_primary':assert supports==[41,7,129,28]
 for p in ROOT.rglob('*'):
  if not p.is_file() or any(x in p.parts for x in ['.git','.private','.dart_tool','build','Pods','ephemeral','.symlinks']):continue
  rel=str(p.relative_to(ROOT));assert p.stat().st_size<50*1024**2,(rel,'Unexpected large artifact')
  if p.suffix in ['.csv','.json','.jsonl','.ipynb','.md','.py','.yaml','.dart']:
   try:text=p.read_text()
   except UnicodeDecodeError:continue
   if re.search(r'PAT_\d',text):errors.append('Private group identifier in '+rel)
   # High-confidence token/key patterns only; values never printed.
   for pattern in [r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{50,}',r'AKIA[0-9A-Z]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']:
    if re.search(pattern,text):errors.append('Potential credential in '+rel)
 if errors:raise AssertionError('\n'.join(errors))
 print(json.dumps({'status':'PASS','python_files_parsed':python_count,'metric_rows_reconciled':len(metrics),'model_SHA256':SHA,'datasets_or_patient_records_included':False,'note':'Medical sample inclusion is bounded and licensed; see DATA_AND_ARTIFACT_LICENSES.md. This pattern scan is supplemented with detect-secrets.'},indent=2))
if __name__=='__main__':main()
