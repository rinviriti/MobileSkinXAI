"""Freeze the audited data protocol; no training, model imports or installations."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mobileskinxai-matplotlib')
import csv,json,hashlib,shutil,stat,uuid
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research/thesis_assets/phase4_duplicate_audit/v20261116T230847Z_34d5abf2'
AUDIT=ROOT/'research/thesis_assets/phase4_duplicate_audit_revision/v20261116T233232Z_86ade2d0'
CLASSES=['NV','MEL','BCC','SCC']
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def check_hash(t):
 p,h=t;p=Path(p);p=p if p.is_absolute() else ROOT/p
 assert sha(p)==h,str(p)
 return True
version='v'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
OUT=ROOT/'research/thesis_assets/final_dataset_release'/version
OUT.mkdir(parents=True,exist_ok=False)
def js(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')
def write(name,rs):
 rs=list(rs);assert rs,name
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
# Snapshot every prior audit artifact and original CSV before any release writes.
snapshot={rel(p):sha(p) for directory in [OLD,AUDIT] for p in directory.rglob('*') if p.is_file()}
snapshot.update({rel(p):sha(p) for p in (ROOT/'dataset').rglob('*.csv')})
js('input_preservation_snapshot.json',snapshot)
configs={m:json.loads((ROOT/f'final_training/configs/{m}.json').read_text()) for m in ['M1','M2','M3','M4','M5']}
for m in configs:
 p=OUT/f'previous_active_configs/{m}.json';p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f'final_training/configs/{m}.json',p)
shutil.copyfile(ROOT/'final_training/data_approval.json',OUT/'previous_data_approval.json')
raw=read(ROOT/'dataset/audit/raw_manifest.csv');rawmap={(r['dataset'],r['image']):r for r in raw}
base=read(ROOT/'dataset/ISIC2019/preprocessed/rgb224_letterbox_gray114_v1/full/manifest.csv')
revised=read(AUDIT/'manifests/ISIC2019_revised_research_manifest.csv')
pad=read(ROOT/'dataset/PADUFES20/preprocessed/rgb224_letterbox_gray114_v1/manifest.csv')
decisions={r['image']:r for r in read(AUDIT/'training_decision_table.csv')}
pairs=read(AUDIT/'all_candidate_pair_decisions.csv');unresolved=read(AUDIT/'high_risk_unresolved.csv');assert len(unresolved)==139
membership=read(OLD/'identity_group_membership.csv');identity={(r['dataset'],r['image']):r['identity_group'] for r in membership}
proposed={(r['dataset'],r['image']):r for r in read(AUDIT/'evaluation_eligibility_PROPOSED.csv')}
# Independently recompute the risk graph instead of trusting proposed CSV decisions.
parent={r['identity_group']:r['identity_group'] for r in membership}
def find(x):
 while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
 return x
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:parent[max(a,b)]=min(a,b)
def key(p,s):return p['dataset_'+s],p['image_'+s]
for p in unresolved:union(identity[key(p,'a')],identity[key(p,'b')])
patients=defaultdict(list)
for r in pad:patients[r['patient_id']].append(identity[('PADUFES20',r['image'])])
for gs in patients.values():
 for g in gs[1:]:union(gs[0],g)
pre_active={('ISIC2019',r['image']):r for r in revised};pre_active.update({('PADUFES20',r['image']):r for r in pad})
risk_splits=defaultdict(set)
for k,r in pre_active.items():risk_splits[find(identity[k])].add(r['split'])
elig={}
for k,r in pre_active.items():
 sp=r['split'];ss=risk_splits[find(identity[k])]
 quarantine=(sp=='test' and 'validation' in ss) or (sp=='external' and bool(ss & {'train','validation','test'}))
 elig[k]=not quarantine
 if sp!='train':assert elig[k]==(proposed[k]['proposed_score_eligible']=='True'),k
# Identify contradictory labels on exact captures and on declared lesion groups.
conflicts=set();conflict_evidence=[]
for basis in ['sha256','pixel_sha256','lesion']:
 d=defaultdict(list)
 for k,r in pre_active.items():
  rr=rawmap[k];token=(k[0],rr['patient_id'],rr['lesion_id']) if basis=='lesion' else rr[basis]
  if token:d[token].append(k)
 for token,ks in d.items():
  if len({pre_active[k]['label'] for k in ks})>1:
   # Different lesions belonging to a PAD patient are not contradictory labels.
   assert len({pre_active[k]['split'] for k in ks})==1,'Cross-partition label conflict needs separate review'
   conflicts.update(ks)
   for k in ks:conflict_evidence.append({'dataset':k[0],'image':k[1],'label':pre_active[k]['label'],'basis':basis,'identity_token':str(token),'action':'quarantine entire contradictory identity component; patient-complete for PAD; no relabeling'})
# Exclude known identity companions of contradictory captures; PAD identity includes patient closure.
conflictgroups={identity[k] for k in conflicts}
conflicts={k for k in pre_active if identity[k] in conflictgroups}
training_conflicts={k for k in conflicts if pre_active[k]['split']=='train'}
assert {k[1] for k in training_conflicts}=={'ISIC_0067502','ISIC_0067980','ISIC_0069013','ISIC_0071017'}
pad_conflicts=conflicts-training_conflicts
print('Contradictory-label exclusions:',len(training_conflicts),'training,',len(pad_conflicts),'PAD patient-complete',flush=True)
write('label_conflict_evidence.csv',conflict_evidence)
# Verify all original raw/prepared image checksums, not just retained images.
prepared=read(OLD/'prepared_image_preservation.csv');preparedmap={(r['dataset'],r['image']):r for r in prepared}
checks={}
for name,items in [('raw_images',[(r['source'],r['sha256']) for r in raw]),('prepared_images',[(r['path'],r['sha256']) for r in prepared])]:
 with ThreadPoolExecutor(max_workers=4) as pool:count=sum(pool.map(check_hash,items,buffersize=16))
 checks[name]={'checked':count,'hash_mismatches':0};print(name,count,'verified',flush=True)
allrows=[]
for dataset,rs,data_root in [('ISIC2019',base,'dataset/ISIC2019/preprocessed/rgb224_letterbox_gray114_v1/full'),('PADUFES20',pad,'dataset/PADUFES20/preprocessed/rgb224_letterbox_gray114_v1')]:
 for r in rs:
  k=(dataset,r['image']);rr=rawmap[k];sp=r['split'];g=identity[k]
  active=k in pre_active and elig.get(k,False) and k not in conflicts
  reason='retained_audited_partition'
  if k in conflicts:reason='confirmed_exact_capture_contradictory_labels; quarantine_complete_identity_component; patient_complete_for_PAD'
  elif sp=='train' and k not in pre_active:reason=decisions[r['image']]['justification']
  elif not active:reason=proposed[k]['reason']
  role={'train':'training','validation':'model_selection','test':'internal_test','external':'external_PAD'}[sp]
  allrows.append({**r,'dataset':dataset,'original_split':sp,'split':sp,'release_role':role if active else 'quarantined','release_eligible':active,'optimization_allowed':active and sp=='train','model_selection_allowed':active and sp=='validation','calibration_allowed':False,'exclusion_reason':'' if active else reason,'retention_reason':reason if active else '', 'identity_group':g,'risk_component':find(g),'patient_id':rr['patient_id'],'source':rr['source'],'source_sha256':rr['sha256'],'oriented_pixel_sha256':rr['pixel_sha256'],'processed_path':data_root+'/'+r['processed_relative_path'],'processed_sha256':preparedmap[k]['sha256'],'dataset_version':version})
active=[r for r in allrows if r['release_eligible']];amap={(r['dataset'],r['image']):r for r in active}
assert len(allrows)==20795 and len(active)==18492-len(pad_conflicts)
write('dataset_manifest.csv',allrows)
write('eligible_manifest.csv',active)
write('exclusions.csv',[r for r in allrows if not r['release_eligible']])
# This ISIC-only compatibility view is the sole split CSV wired into all five trainers.
write('manifests/ISIC2019_training_evaluation.csv',[r for r in active if r['dataset']=='ISIC2019'])
write('manifests/PADUFES20_external_evaluation.csv',[r for r in active if r['dataset']=='PADUFES20'])
# No confirmed identity, source-lesion, patient or exact-capture component may cross retained partitions.
intersections=[]
for basis in ['identity_group','source_sha256','oriented_pixel_sha256','known_lesion','PAD_patient']:
 owners=defaultdict(set)
 for r in active:
  if basis=='known_lesion':token=(r['dataset'],r['patient_id'] if r['dataset']=='PADUFES20' else '',r['lesion_id'])
  elif basis=='PAD_patient':
   if r['dataset']!='PADUFES20':continue
   token=r['patient_id']
  else:token=r[basis]
  owners[token].add(r['split'])
 bad={str(k):list(v) for k,v in owners.items() if len(v)>1};assert not bad,(basis,bad)
 intersections.append({'basis':basis,'groups_checked':len(owners),'cross_partition_groups':0})
write('identity_intersection_checks.csv',intersections)
resolution=[]
for p in unresolved:
 ka,kb=key(p,'a'),key(p,'b');both=ka in amap and kb in amap
 assert not both,p['pair_id']
 resolution.append({**p,'a_final_eligible':ka in amap,'b_final_eligible':kb in amap,'resolved_by_policy':True,'identity_classification':'unresolved; not reclassified as biological duplicate','action':'test lesion-component quarantine; validation preserved' if p['dataset_a']==p['dataset_b']=='ISIC2019' else 'PAD patient-complete quarantine','decision_basis':'Pre-model similarity/identity audit only; no model predictions, difficulty or performance used'})
write('unresolved_pair_disposition.csv',resolution)
remaining=[]
for p in pairs:
 a,b=key(p,'a'),key(p,'b')
 if a in amap and b in amap and amap[a]['split']!=amap[b]['split'] and p['revised_classification']!='visually_similar_but_distinct':remaining.append(p['pair_id'])
assert not remaining
checks['candidate_boundaries']={'candidate_pairs_checked':len(pairs),'unresolved_or_confirmed_pairs_crossing_final_partitions':remaining,'unresolved_pairs_quarantined':139}
# Metadata, dimensions and label checks on every eligible prepared file.
def validate_image(r):
 p=ROOT/r['processed_path']
 with Image.open(p) as im:
  im.load();assert im.mode=='RGB' and im.size==(224,224),str(p)
 assert r['label'] in CLASSES and int(r['label_index'])==CLASSES.index(r['label'])
 return True
with ThreadPoolExecutor(max_workers=4) as pool:count=sum(pool.map(validate_image,active,buffersize=16))
checks['eligible_images']={'decoded_RGB_224x224':count,'missing_or_corrupt':0}
# Exact-copy collapse uses file/pixel hashes only, not lesion labels or predictions.
# It is only a scoring view; all retained captures remain in the release manifest.
capture_parent={(r['dataset'],r['image']):(r['dataset'],r['image']) for r in active if r['split']!='train'}
def cf(k):
 while capture_parent[k]!=k:capture_parent[k]=capture_parent[capture_parent[k]];k=capture_parent[k]
 return k
for field in ['source_sha256','oriented_pixel_sha256']:
 seen={}
 for r in active:
  if r['split']=='train':continue
  k=(r['dataset'],r['image']);h=r[field]
  if h in seen:
   a,b=cf(k),cf(seen[h]);capture_parent[max(a,b)]=min(a,b)
  else:seen[h]=k
canonical=defaultdict(list)
for k in capture_parent:canonical[cf(k)].append(k)
canon={k:min(ks) for ks in canonical.values() for k in ks}
units=[]
for r in active:
 if r['split']=='train':continue
 k=(r['dataset'],r['image']);c=canon[k]
 unit=':'.join([r['dataset'],r['patient_id'],r['lesion_id']]) if r['dataset']=='PADUFES20' else 'ISIC2019:'+r['lesion_id']
 units.append({'dataset':r['dataset'],'image':r['image'],'split':r['split'],'label':r['label'],'lesion_scoring_unit':unit,'bootstrap_cluster':r['patient_id'] if r['dataset']=='PADUFES20' else unit,'exact_capture_representative_dataset':c[0],'exact_capture_representative_image':c[1],'use_in_lesion_probability_mean':k==c})
# A lesion must have a unique class before probability aggregation is well-defined.
unitlabels=defaultdict(set)
for r in units:unitlabels[r['lesion_scoring_unit']].add(r['label'])
assert all(len(v)==1 for v in unitlabels.values())
write('evaluation_scoring_units.csv',units)
counts=[]
for sp in ['train','validation','test','external']:
 for c in CLASSES:
  rs=[r for r in active if r['split']==sp and r['label']==c]
  assert rs,(sp,c)
  counts.append({'split':sp,'class':c,'images':len(rs)})
write('class_split_counts.csv',counts)
traincounts={r['class']:r['images'] for r in counts if r['split']=='train'};n=sum(traincounts.values())
weights={'class_order':CLASSES,'counts':traincounts,'formula':'N_train / (4 * n_train_class)','weights':{c:n/(4*v) for c,v in traincounts.items()},'source_manifest':'manifests/ISIC2019_training_evaluation.csv','dataset_version':version}
js('class_weights.json',weights)
scoring=[]
for sp in ['validation','test','external']:
 for c in CLASSES:
  rs=[r for r in units if r['split']==sp and r['label']==c]
  scoring.append({'split':sp,'class':c,'images':len(rs),'unique_exact_captures':sum(r['use_in_lesion_probability_mean'] for r in rs),'lesion_scoring_units':len({r['lesion_scoring_unit'] for r in rs}),'bootstrap_clusters':len({r['bootstrap_cluster'] for r in rs})})
write('evaluation_denominators.csv',scoring)
policy={'dataset_version':version,'status':'ACTIVATED_AND_LOCKED_PRE_MODEL','classes':CLASSES,'input':'224x224 RGB aspect-preserving letterbox gray114 (existing prepared images unchanged)','label_integrity':'Exclude complete identity components containing exact captures with contradictory labels; train4, PAD patient-complete'+str(len(pad_conflicts))+'; no relabeling', 'training_exclusions':'Existing revised exclusions plus all four members of exact-capture NV/MEL label-conflict component G012775; no relabeling','quarantine':'Risk graph joins unresolved pair known-identity components, with complete PAD patient closure. Preserve validation; quarantine test component connected to validation, and PAD patient component connected to any ISIC partition. Original split labels unchanged.','uses_model_results':False,'quarantined_usage':'Forbidden for training, calibration, early stopping, model selection and primary internal/external scoring','calibration':'No calibration permitted in this release; any future calibration protocol requires a separately frozen training-only subset','primary_test_evaluation':'For each declared lesion, arithmetic mean of softmax probabilities over exact-capture representatives chosen lexicographically by dataset/image ID before predictions; argmax in NV,MEL,BCC,SCC order. Report macro-F1, balanced accuracy, accuracy and class recalls at lesion level.','uncertainty':'Predeclared seed42 cluster bootstrap 2000 replicates, percentile95 intervals: ISIC lesion clusters, PAD patient clusters; no tuning using heldout results','secondary_evaluation':'Image-level metrics on eligible images with image/lesion denominators disclosed; no quarantined sensitivity set used for model selection','model_selection':'Eligible validation only; train-only statistics and augmentation; five fixed configurations share this exact data release','independence_limitations':'No guarantee of unknown biological lesion or patient independence. ISIC globally linkable patient IDs unavailable; distinct photograph screening is not biological proof; unresolved139 are quarantined, not declared duplicates; finite candidate-search recall.'}
js('DATASET_PROTOCOL.json',policy)
# Freeze data bindings for exactly the five existing configurations; model/optimizer settings unchanged.
manifest=OUT/'manifests/ISIC2019_training_evaluation.csv'
for m,cfg in configs.items():
 cfg['split_manifest']=rel(manifest);cfg['dataset_version']=version;cfg['dataset_release']=rel(OUT);cfg['external_evaluation_manifest']=rel(OUT/'manifests/PADUFES20_external_evaluation.csv');cfg['evaluation_scoring_units']=rel(OUT/'evaluation_scoring_units.csv');cfg['dataset_protocol']=rel(OUT/'DATASET_PROTOCOL.json')
 js(f'experiment_configs/{m}.json',cfg)
# Check preserved inputs and known model/code/weight snapshots before approval.
with ThreadPoolExecutor(max_workers=4) as pool:checked=sum(pool.map(check_hash,list(snapshot.items()),buffersize=16))
checks['prior_audits_original_CSVs']={'checked':checked,'hash_mismatches':0}
model_snapshot=json.loads((OLD/'model_preservation_snapshot.json').read_text())
for p,h in model_snapshot.items():assert sha(ROOT/p)==h,p
checks['models_code_pretrained_weights_and_prior_configs']={'checked':len(model_snapshot),'hash_mismatches':0,'note':'Active config data pointers updated only after these checks; backups retained'}
# Check every referenced review sheet exists and is decodable; classifications are prior recorded screening.
review=read(AUDIT/'visual_review.csv');assert len(review)==303
contact=list((AUDIT/'contact_sheets').rglob('*.jpg'))+list((AUDIT/'restoration_support/contact_sheets').rglob('*.jpg'))+[AUDIT/'high_risk_full_image_comparisons.jpg']
for p in contact:
 with Image.open(p) as im:im.verify()
checks['review_evidence']={'visual_decisions':len(review),'contact_sheets_read_and_verified':len(contact),'restoration_driving_pairs':len(read(AUDIT/'restoration_support/visual_clearance.csv')),'independent_clinical_review':False}
js('integrity_checks.json',checks)
# Actual-count figures only; no methodology diagrams.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
(OUT/'figures').mkdir(exist_ok=True)
colors=['#4477AA','#EE6677','#228833','#CCBB44']
x=np.arange(4);width=.20
fig,ax=plt.subplots(figsize=(9,4.6))
for j,sp in enumerate(['train','validation','test','external']):
 vals=[next(r['images'] for r in counts if r['split']==sp and r['class']==c) for c in CLASSES]
 bars=ax.bar(x+(j-1.5)*width,vals,width,label={'external':'PAD external'}.get(sp,sp.title()),color=colors[j]);ax.bar_label(bars,fontsize=7,padding=2)
ax.set_xticks(x,CLASSES);ax.set_ylabel('Eligible images');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,7800);fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'figures/class_distribution.{ext}',dpi=300)
plt.close(fig)
totals=[sum(r['images'] for r in counts if r['split']==sp) for sp in ['train','validation','test','external']]
write('figures/final_split_distribution.csv',[{'split':sp,'images':v} for sp,v in zip(['train','validation','test','external'],totals)])
shutil.copyfile(OUT/'class_split_counts.csv',OUT/'figures/class_distribution.csv')
fig,ax=plt.subplots(figsize=(7,4.5));bars=ax.bar(['ISIC train','ISIC validation','ISIC test','PAD external'],totals,color=colors);ax.bar_label(bars,padding=3);ax.set_ylabel('Eligible images');ax.set_ylim(0,max(totals)*1.15);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'figures/final_split_distribution.{ext}',dpi=300)
plt.close(fig)
summary={'dataset_version':version,'approved':True,'technical_integrity_checks_passed':True,'counts':counts,'totals':dict(zip(['train','validation','test','external'],totals)),'excluded_training':2054,'confirmed_cross_boundary_training_exclusions':2,'precautionary_training_exclusions':2048,'contradictory_label_training_exclusions':4,'quarantined_test':238,'quarantined_PAD_similarity':11,'quarantined_PAD_label_conflict':len(pad_conflicts),'quarantined_PAD_total':1333-totals[3],'unresolved_identity_pairs':139,'unresolved_pairs_crossing_eligible_partitions':0,'class_weights':weights,'independence_guaranteed':False}
js('release_summary.json',summary)
rows='\n'.join('| '+c+' | '+' | '.join(str(next(r['images'] for r in counts if r['split']==sp and r['class']==c)) for sp in ['train','validation','test','external'])+' |' for c in CLASSES)
report=f'''# MobileSkinXAI final dataset release

**Approved for the five fresh final training configurations on technical dataset integrity grounds.** Dataset version `{version}`. This is not a guarantee of biological patient independence. No model training, package installation or Flutter modification occurred.

| Class | ISIC training | ISIC validation | ISIC internal test | PAD external |
|---|---:|---:|---:|---:|
{rows}
| **Total** | **{totals[0]}** | **{totals[1]}** | **{totals[2]}** | **{totals[3]}** |

## Frozen selection and exclusions

The source ISIC split remains train13,686 / validation2,835 / test2,941; PAD remains external1,333. Original images, split CSVs and all prior audit artifacts are preserved. This release determines eligibility only; no image is reassigned or relabeled.

The prior revised training proposal restores144 images after component-complete screening and excludes2,050:2 confirmed train–heldout near-copies and2,048 precautionary cases. Final integrity review additionally found two pairs of **byte-identical photographs with conflicting NV/MEL labels**, entirely in training: ISIC_0067502/ISIC_0067980 and ISIC_0069013/ISIC_0071017. All four members of identity component G012775 are excluded without adjudicating the true label. Final training11,632: NV6,838, MEL2,611, BCC1,799, SCC384. `label_conflict_evidence.csv` and `exclusions.csv` give exact evidence and reasons. These are metadata/image-integrity decisions, not classification-difficulty filtering.

## Locked conservative held-out policy

The pre-model proposal was independently reproduced and applied. Unresolved links join known identity components; PAD components additionally close over whole patients. Validation is preserved. Test members of components connected to validation are quarantined; complete PAD patient components linked to ISIC are quarantined. The unresolved-similarity policy removes238 test images and11 PAD images from scoring and every other experimental use. All139 unresolved pair records are preserved as **unresolved**, not asserted duplicates; none has both endpoints in eligible partitions. All four classes survive in every final set. In addition, exact file/decoded-pixel matches with contradictory diagnoses affect six PAD patients. Patient-complete label-integrity quarantine excludes another {len(pad_conflicts)} PAD images, leaving {totals[3]} for external scoring. This is deterministic data-integrity filtering; no diagnosis is chosen and no original label is changed. `label_conflict_evidence.csv` records every conflicting capture.

No quarantined validation/test/PAD image may enter optimization, calibration, early stopping, model selection or primary scoring. No calibration is planned. Eligible validation is the only model-selection set. Heldout exclusions were fixed exclusively by existing similarity/identity evidence before final model predictions; no model results or difficulty estimates were read.

`DATASET_PROTOCOL.json` fixes primary lesion-level evaluation: lexicographically choose exact-capture representatives; average their softmax probabilities within each declared lesion, then argmax in NV,MEL,BCC,SCC order. PAD lesions are keyed by patient+lesion, not whole-patient identity components, because one patient can have different diagnoses at different lesions. Cluster bootstrap is by lesion for ISIC, patient for PAD. `evaluation_scoring_units.csv` fixes membership and representatives; `evaluation_denominators.csv` reports image/capture/lesion/cluster counts. Secondary image-level metrics must disclose their denominators. Test and PAD data never tune the model or protocol.

## Integrity and limits

`integrity_checks.json` verifies all27,629 raw image and20,795 prepared image hashes, all eligible224x224RGB files, immutable audit/CSV inputs and model artifacts before config data bindings change. File hashes, decoded-pixel hashes, known source lesion IDs, confirmed near-copy groups and PAD patients have zero intersections across eligible partitions. All4,826 candidates are checked against final eligibility; no confirmed or unresolved candidate boundary survives.

Biological lesion and patient independence cannot be guaranteed without globally linkable ISIC patient identifiers. Different photograph morphology is screening evidence, not biological proof; candidate search can miss cases. The139 ambiguous relationships remain uncertain and are handled by conservative quarantine. The164 previously screened distinct pairs remain as documented image-screen clearances, not independently certified clinical adjudications. Within-partition exact captures remain preserved; scoring aggregation avoids exact-copy overweighting. Some residual training duplicates remain within training; their repetition and finite audit coverage must be disclosed in the thesis.

Train-only weights are N_train/(4*n_class), in NV,MEL,BCC,SCC order: {list(weights['weights'].values())}. All five configurations share this exact release. Active configuration updates change only data bindings/metadata; model/optimizer settings and checkpoint paths are preserved, with previous configs archived.

## Artifacts and immutability

Release folder: `{rel(OUT)}`. Canonical `dataset_manifest.csv` includes eligible and excluded records, original splits, exclusion reasons and original/prepared SHA256 hashes. `eligible_manifest.csv` is its eligible-only view. ISIC trainer view: `manifests/ISIC2019_training_evaluation.csv`; external view: `manifests/PADUFES20_external_evaluation.csv`. `class_split_counts.csv`, `class_weights.json`, `unresolved_pair_disposition.csv` and `identity_intersection_checks.csv` are machine-readable evidence. Dataset-only figures are `figures/class_distribution.*` and `figures/final_split_distribution.*` (PNG300DPI, SVG, PDF, actual CSV data).

The release is append-only by convention, files are read-only, and `checksums.sha256` plus `RELEASE_LOCK.json` detect edits. The file owner can still change permissions; this is a checksummed frozen research release, not filesystem-level WORM storage. Any future change must create a new version; never edit this release in place. Original files and previous reports remain untouched.
'''
# Normalize compact numeric prose for readability without altering factual tables.
for a,b in [('train13','train 13'),('validation2','validation 2'),('test2','test 2'),('external1','external 1'),('restores144','restores 144'),('excludes2','excludes 2'),('proposal11','proposal 11'),('training11','training 11'),('component G','component G'),('removes238','removes 238'),('and11','and 11'),('All139','All 139'),('all27','all 27'),('and20','and 20'),('eligible224','eligible 224'),('All4,826','All 4,826'),('The139','The 139'),('The164','The 164'),('PNG300','PNG 300')]:report=report.replace(a,b)
(OUT/'DATASET_RELEASE_REPORT.md').write_text(report)
approval={'approved':True,'status':'approved_technical_integrity_with_documented_independence_limits','dataset_version':version,'manifest_path':rel(manifest),'manifest_sha256':sha(manifest),'release_manifest_path':rel(OUT/'dataset_manifest.csv'),'release_manifest_sha256':sha(OUT/'dataset_manifest.csv'),'review_document':rel(OUT/'DATASET_RELEASE_REPORT.md'),'protocol_path':rel(OUT/'DATASET_PROTOCOL.json'),'protocol_sha256':sha(OUT/'DATASET_PROTOCOL.json'),'unresolved_identity_pairs_quarantined':139,'unresolved_eligible_boundary_pairs':0,'patient_independence_guaranteed':False,'approved_scope':'Exactly five fresh models using frozen release; no training started'}
js('data_approval.json',approval)
shutil.copyfile(Path(__file__),OUT/'release_builder.py')
# No active approval until all release artifacts are complete and checksummed.
files=sorted(p for p in OUT.rglob('*') if p.is_file())
(OUT/'checksums.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in files))
js('RELEASE_LOCK.json',{'version':version,'checksums_sha256':sha(OUT/'checksums.sha256'),'files':len(files),'immutability':'read-only files + content hashes; owner-reversible permissions','created_UTC':datetime.now(timezone.utc).isoformat()})
for p in OUT.rglob('*'):
 if p.is_file():p.chmod(stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH)
for m,cfg in configs.items():(ROOT/f'final_training/configs/{m}.json').write_text(json.dumps(cfg,indent=2)+'\n')
# Validate each active config is byte-identical to frozen snapshot.
for m in configs:assert sha(ROOT/f'final_training/configs/{m}.json')==sha(OUT/f'experiment_configs/{m}.json')
(ROOT/'final_training/data_approval.json').write_text(json.dumps(approval,indent=2)+'\n')
print(json.dumps({'release':rel(OUT),'summary':summary},indent=2),flush=True)
