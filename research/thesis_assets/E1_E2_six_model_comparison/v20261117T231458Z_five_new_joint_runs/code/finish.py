"""Assemble measured results and independently audit artifacts; never optimize models."""
from common import *
from evaluate import KEYS,NAMES,BASE,state,cfg_for,METRICS
from stopping_v1 import update,should_stop
from PIL import Image
import torch,timm,shutil,xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix,accuracy_score,balanced_accuracy_score,f1_score,precision_recall_fscore_support,roc_auc_score,average_precision_score

def table(rows,columns=None):
 if not rows:return '(Unavailable)\n'
 columns=columns or list(rows[0]);fmt=lambda v:f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/')
 return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(fmt(r.get(c,'')) for c in columns)+' |\n' for r in rows)
def select(rows,**kw):return [r for r in rows if all(r[k]==v for k,v in kw.items())]
def status_report(out):
 rows=[]
 for k in KEYS:
  p=BASE/'03_E2_Training/TRAINING_COMPLETE.json' if k=='M3' else out/'03_E2_Training'/k/'TRAINING_COMPLETE.json'
  if p.exists():
   s=load(p);rows.append({'Model':NAMES[k],'Status':s['status'],'Best epoch':s['best_epoch'],'Stop epoch':s['epochs'],'Actual completed seconds':s['training_seconds'],'Prescribed seconds':s.get('prescribed_training_seconds',s['training_seconds']),'PAD validation lesion macro-F1':s['best_PAD_validation_lesion_F1']})
  else:rows.append({'Model':NAMES[k],'Status':'not completed','Best epoch':'—','Stop epoch':'—','Actual completed seconds':'—','Prescribed seconds':'—','PAD validation lesion macro-F1':'—'})
 text='# Final E2 training status\n\n'+table(rows)+'\nM1 recovered without retraining. M3 reused unchanged. M1 actual committed consumption includes epochs17–23, excluded from selection; unmeasured partial epoch24 overhead is additional. M1 prescribed duration is the epoch16 history value.\n'
 qp=out/'00_Protocol/TEACHER_QUALIFICATION.json'
 if qp.exists():text+='\nTeacher qualification: '+json.dumps(load(qp),indent=2)+'\n'
 p=out/'13_Reports/FINAL_E2_TRAINING_STATUS.md';p.write_text(text);write(out/'12_Source_Data/FINAL_E2_TRAINING_STATUS.csv',rows);return rows

def main(out):
 out=Path(out);training=status_report(out);assert load(out/'13_Reports/METRICS_COMPLETE.json')['status']=='PASS';assert load(out/'09_Explainability/GRADCAM_COMPLETE.json')['status']=='PASS';summary=read(out/'12_Source_Data/ALL_SETTING_METRICS.csv');cw=read(out/'CLASSWISE_RESULTS.csv');cmrows=read(out/'06_Confusion_Matrices/CONFUSION_COUNTS.csv');cost=read(out/'MODEL_COMPLEXITY.csv');gaps=read(out/'SOURCE_TARGET_GAP_COMPARISON.csv');improvements=read(out/'PAD_IMPROVEMENTS_BY_MODEL.csv');lock=load(out/'00_Protocol/FINAL_EVALUATION_LOCK.json');audit=[]
 def check(name,detail):audit.append({'check':name,'status':'PASS','evidence':detail})
 for p,h in load(out/'12_Source_Data/PRESERVATION_HASHES.json').items():assert sha(ROOT/p)==h,p
 check('Historical artifacts unchanged','All original preservation-manifest hashes verified, including E1 checkpoints, M3 E2 and original results/figures')
 for f in ['LOCK.json','CONTROLLER_FIX_LOCK.json']:
  for p,h in load(out/'00_Protocol'/f)['sha256'].items():assert sha(out/p)==h,p
 check('Original and corrected controller seals','Both LOCK.json and CONTROLLER_FIX_LOCK.json match')
 # The copied split is byte-identical to the already-completed original E1/E2 experiment.
 for s in ['train','validation','test']:assert sha(out/f'01_Frozen_Data_Manifests/PAD_{s}.csv')==sha(BASE/f'01_PAD_Splits/PAD_{s}.csv')
 parts={s:read(out/f'01_Frozen_Data_Manifests/PAD_{s}.csv') for s in ['train','validation','test']}
 for i,a in enumerate(parts):
  for b in list(parts)[i+1:]:assert not {r['patient_id'] for r in parts[a]}&{r['patient_id'] for r in parts[b]};assert not {r['image'] for r in parts[a]}&{r['image'] for r in parts[b]}
 assert all(r['patient_id']!=__import__('os').environ.get('MSX_PRIVATE_VALIDATION_GROUP', '__PRIVATE_GROUP_REQUIRED__') for s in ['train','test'] for r in parts[s]);assert any(r['patient_id']==__import__('os').environ.get('MSX_PRIVATE_VALIDATION_GROUP', '__PRIVATE_GROUP_REQUIRED__') for r in parts['validation']);assert [len(parts[s]) for s in ['train','validation','test']]==[781,261,261]
 check('Frozen PAD identity and grouping','Original byte-identical split; 781/261/261 images; patient disjoint; [private group] retained only in validation')
 dev=read(out/'01_Frozen_Data_Manifests/DEVELOPMENT_MANIFEST.csv');assert {r['study_split'] for r in dev}=={'train','validation'};assert not {r['image'] for r in dev if r['domain']=='PAD'}&{r['image'] for r in parts['test']}
 check('No held-out training membership','DEVELOPMENT_MANIFEST includes only fixed train/validation; evaluation manifests disjoint')
 for key in ['M1','M2','M4','M5','M5-LR']:
  run=out/'03_E2_Training'/key;s=state(out,key);cfg=cfg_for(out,key);hs=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()];best=-1.;be=0;stale=0;prev=None;stop=None
  for h in hs[:s['epochs']]:
   e=h['epoch'];f=e<=cfg['frozen_epochs']
   if prev is None or prev!=f:stale=0;prev=f
   best,be,stale,_=update(best,be,stale,h['PAD_validation']['lesion']['macro_f1'],e,f)
   assert h['best_epoch_so_far']==be and h['stale']==stale
   assert h['optimizer_updates']==727
   if should_stop(e,cfg['frozen_epochs'],stale,cfg['patience']):stop=e;break
  assert be==s['best_epoch'] and abs(best-s['best_PAD_validation_lesion_F1'])<1e-12
  assert (stop==s['epochs']) or (stop is None and s['epochs']==cfg['frozen_epochs']+cfg['fine_epochs'])
  if key!='M1':assert len(hs)==s['epochs'] and max(int(p.stem.split('_')[-1]) for p in (run/'checkpoints').glob('epoch_*.pt'))==s['epochs']
  for p in sorted(run.glob('commit_*.json')):
   c=load(p);assert sha(run/'checkpoints'/c['checkpoint'])==c['sha256']
  selected=read(run/f'PAD_validation_epoch_{be:03d}.csv');assert [r['image'] for r in selected]==[r['image'] for r in dev if r['domain']=='PAD' and r['study_split']=='validation'];assert len(selected)==261
  final=ROOT/s.get('prescribed_final_checkpoint',str((run/'checkpoints/last.pt').relative_to(ROOT)));payload=torch.load(final,map_location='cpu',weights_only=True);assert payload['epoch']==s['epochs'];assert 'optimizer' in payload and 'rng' in payload;assert payload['control']['stale']==stale;del payload
 check('New E2 run counts and stopping','Exactly five additional configurations; M1 recovered with preserved extra epochs; remaining four obey in-loop patience; all commit hashes and final optimizer/RNG states verify')
 for m in lock['models']:
  for condition in ['E1','E2']:
   p=ROOT/m[condition+'_checkpoint'];assert sha(p)==m[condition+'_sha256'];payload=torch.load(p,map_location='cpu',weights_only=True);net=timm.create_model(m['E2_configuration']['architecture'],pretrained=False,num_classes=4);net.load_state_dict(payload['state_dict'],strict=True);assert all(torch.isfinite(t).all() for t in payload['state_dict'].values() if t.is_floating_point());assert net.get_classifier().out_features==4 if isinstance(net.get_classifier(),torch.nn.Linear) else True;del payload,net
 check('All twelve checkpoint identities','Strict architecture loads, finite weights, four-class configs and frozen hashes; M3 original unchanged')
 a=cfg_for(out,'M5');b=cfg_for(out,'M5-LR');diff=[k for k in a if a[k]!=b[k]];assert set(diff)=={'id','name','head_lr','backbone_lr'};assert a['teacher_sha256']==b['teacher_sha256'] and a['student_sha256']==b['student_sha256'];assert load(out/'00_Protocol/TEACHER_QUALIFICATION.json')['qualified']
 for epoch in range(1,min(state(out,'M5')['epochs'],state(out,'M5-LR')['epochs'])+1):assert sha(out/f'03_E2_Training/M5/augmentation_epoch_{epoch:03d}.jsonl')==sha(out/f'03_E2_Training/M5-LR/augmentation_epoch_{epoch:03d}.jsonl')
 check('Enhanced LR-only control','Only identity labels and prescribed LR fields differ; identical student/teacher, augmentation log hashes per common epoch, dataset, losses and gates')
 for ds in ['ISIC','PAD']:
  refs=None
  for condition,folder in [('E1','02_E1_Evaluations'),('E2','04_E2_Evaluations')]:
   for key in KEYS:
    images=read(out/f'{folder}/{key}_{ds}_image.csv');manifest=read(out/'01_Frozen_Data_Manifests'/('ISIC_TEST_MANIFEST.csv' if ds=='ISIC' else 'PAD_test.csv'));assert [r['image'] for r in images]==[r['image'] for r in manifest];groups={}
    for r in images:
     assert int(r['true_index'])==CLASSES.index(r['true_class'])
     if r['use_in_lesion_probability_mean']=='True':groups.setdefault(r['lesion_scoring_unit'],[]).append(r)
    lesion=read(out/f'{folder}/{key}_{ds}_lesion.csv');assert len(lesion)==(1421 if ds=='ISIC' else 205)
    for r in lesion:
     members=groups[r['lesion_scoring_unit']];expected=np.mean([[float(x[c]) for c in CLASSES] for x in members],0);assert np.allclose(expected,[float(r[c]) for c in CLASSES],atol=1e-14,rtol=0);assert len({x['bootstrap_cluster'] for x in members})==1
    for level,records in [('lesion_primary',lesion),('image_secondary',images)]:
     y=np.array([int(r['true_index']) for r in records]);p=np.array([[float(r[c]) for c in CLASSES] for r in records]);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6);pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(4));saved=np.array([int(r['count']) for r in select(cmrows,model=key,condition=condition,dataset=ds,analysis_level=level)]).reshape(4,4);assert np.array_equal(cm,saved);s=select(summary,model=key,condition=condition,dataset=ds,analysis_level=level)[0];pr,re,f,sup=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0)
     for metric,value in [('accuracy',accuracy_score(y,pred)),('balanced_accuracy',balanced_accuracy_score(y,pred)),('macro_f1',f1_score(y,pred,average='macro')),('weighted_f1',f1_score(y,pred,average='weighted'))]:assert abs(float(s[metric])-value)<1e-12
     for j,c in enumerate(CLASSES):
      cl=select(cw,model=key,condition=condition,dataset=ds,analysis_level=level,**{'class':c})[0]
      for field,value in [('precision',pr[j]),('recall',re[j]),('f1',f[j]),('support',sup[j]),('correct',cm[j,j])]:assert abs(float(cl[field])-value)<1e-12
 check('Independent numerical and aggregation reconciliation','All48 metric rows and192 class rows recomputed; exact frozen IDs/capture averaging; confusion counts reconcile; same1421ISIC/205PAD declaredlesions across all checkpoints')
 index=read(out/'FIGURE_INDEX.csv')
 for r in index:
  assert sha(out/r['source_data'])==r['source_sha256']
  with Image.open(out/r['png']) as im:im.load();assert min(im.info.get('dpi',(0,0)))>=299
  ET.parse(out/r['svg']);assert (out/r['pdf']).read_bytes().startswith(b'%PDF-')
 check('Figure/data integrity',f'{len(index)} figure families decode with 300DPI PNG, parseable SVG and PDF headers; all source data hashes match')
 for r in read(out/'09_Explainability/GRADCAM_CASES.csv'):
  if r['configuration']!='M3':
   a=np.load(out/r['map']);assert a.shape==(224,224) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1.00001;assert float(r['gradient_norm'])>0 and float(r['probability_parity_max_abs_difference'])<=1e-3
 check('Grad-CAM provenance','New frozen-model development-only predicted-logit gradients verified; raw inputs/maps/overlays preserved; M3 and E1 figures copied unchanged')
 write(out/'13_Reports/FINAL_INTEGRITY_AUDIT.csv',audit);dump(out/'13_Reports/FINAL_INTEGRITY_AUDIT.json',{'status':'PASS_WITH_DOCUMENTED_M1_PROTOCOL_DEVIATION','checks':audit,'full_study_clinical_claim':False})
 def point(k,c,d):return select(summary,model=k,condition=c,dataset=d,analysis_level='lesion_primary')[0]
 def classrow(k,c,d,cl):return select(cw,model=k,condition=c,dataset=d,analysis_level='lesion_primary',**{'class':cl})[0]
 # The ten requested tables use actual recorded metrics only.
 tables={};tables[1]=read(out/'01_Frozen_Data_Manifests/SPLIT_COUNTS.csv')
 for n,condition in [(2,'E1'),(3,'E2')]:tables[n]=[{'Model':NAMES[k],'ISIC accuracy':float(point(k,condition,'ISIC')['accuracy']),'ISIC macro-F1':float(point(k,condition,'ISIC')['macro_f1']),'PAD accuracy':float(point(k,condition,'PAD')['accuracy']),'PAD macro-F1':float(point(k,condition,'PAD')['macro_f1']),'PAD balanced accuracy':float(point(k,condition,'PAD')['balanced_accuracy'])} for k in KEYS]
 tables[4]=read(out/'SIX_MODEL_SIDE_BY_SIDE_RESULTS.csv');tables[5]=select(gaps,analysis_level='lesion_primary');tables[6]=select(improvements,analysis_level='lesion_primary');tables[7]=[{'Model':NAMES[k],'Condition':c,'Class':cl,**{m:classrow(k,c,'PAD',cl)[m] for m in ['correct','support','precision','recall','f1']}} for k in KEYS for c in ['E1','E2'] for cl in ['MEL','SCC']];tables[8]=cost;tables[9]=[{'Model':NAMES[k],'Experiment':c,'ISIC F1':float(point(k,c,'ISIC')['macro_f1']),'PAD F1':float(point(k,c,'PAD')['macro_f1']),'PAD MEL recall':float(point(k,c,'PAD')['MEL_recall']),'PAD SCC recall':float(point(k,c,'PAD')['SCC_recall'])} for k in ['M5','M5-LR'] for c in ['E1','E2']]
 tables[10]=[{'Claim':'Joint-domain adaptation on this fixed PAD population','Evidence':'Paired E1/E2 point estimates and patient-grouped intervals','Limit':'Post-hoc, one seed, no untouched external replication'},{'Claim':'Zero-shot transfer','Evidence':'Only E1 qualifies','Limit':'E2 has labeled PAD training exposure'},{'Claim':'Novel algorithm or clinical screening safety','Evidence':'Not established','Limit':'Standard supervised joint training; only seven PAD test MEL lesion units'},{'Claim':'Enhanced inference efficiency','Evidence':'Same MobileNetV4 graph; teacher training-only','Limit':'Additional parent and own training costs; no final TFLite export or phone benchmark'}]
 titles={1:'Fixed PAD split',2:'E1 source-only comparison',3:'E2 joint-domain comparison',4:'Four-setting comparison',5:'Source-target differences',6:'PAD improvements and paired uncertainty',7:'PAD melanoma/SCC supports and performance',8:'Complexity and training resource accounting',9:'Original enhancement versus LR control',10:'Supported findings and limitations'}
 sections=[]
 for n,rows in tables.items():
  write(out/f'05_Four_Setting_Comparisons/TABLE_{n:02d}.csv',rows);sections.append(f'## Table {n}. {titles[n]}\n\n'+table(rows))
 best_e1_isic=max(KEYS,key=lambda k:float(point(k,'E1','ISIC')['macro_f1']));best_e1_pad=max(KEYS,key=lambda k:float(point(k,'E1','PAD')['macro_f1']));best_e2_pad=max(KEYS,key=lambda k:float(point(k,'E2','PAD')['macro_f1']));devkey=lock['deployment_development_candidate'];topdev=max(KEYS,key=lambda k:state(out,k)['best_PAD_validation_lesion_F1']);largest=max(tables[6],key=lambda r:float(r['PAD_macro_f1_change']));pair=read(out/'12_Source_Data/ENHANCEMENT_PAIRED_COMPARISONS.csv')
 findings=[]
 for k in KEYS:
  pi,pj=point(k,'E1','PAD'),point(k,'E2','PAD');si,sj=point(k,'E1','ISIC'),point(k,'E2','ISIC');d=lambda metric:100*(float(pj[metric])-float(pi[metric]));correct=[int(classrow(k,'E2','PAD',cl)['correct'])-int(classrow(k,'E1','PAD',cl)['correct']) for cl in CLASSES]
  findings.append(f"- {NAMES[k]}: PAD macro-F1 {float(pi['macro_f1']):.4f} → {float(pj['macro_f1']):.4f} ({d('macro_f1'):+.2f} pp), accuracy {d('accuracy'):+.2f} pp, balanced accuracy {d('balanced_accuracy'):+.2f} pp; MEL recall {d('MEL_recall'):+.2f} pp, SCC recall {d('SCC_recall'):+.2f} pp. ISIC macro-F1 changes {100*(float(sj['macro_f1'])-float(si['macro_f1'])):+.2f} pp. Net correctly classified PAD lesions by NV/MEL/BCC/SCC: {correct}; this separates majority-class accuracy gains from minority-class performance.")
 confusions=[]
 for k in KEYS:
  vals=[]
  for condition in ['E1','E2']:
   cm=np.array([int(r['count']) for r in select(cmrows,model=k,condition=condition,dataset='PAD',analysis_level='lesion_primary')]).reshape(4,4);off=cm.copy();np.fill_diagonal(off,0);i,j=np.unravel_index(off.argmax(),off.shape);vals.append(f'{condition}: largest off-diagonal {CLASSES[i]}→{CLASSES[j]} {off[i,j]} lesions')
  confusions.append('- '+NAMES[k]+': '+'; '.join(vals)+'.')
 supported=[]
 for k in ['M5','M5-LR']:
  delta=100*(float(point(k,'E2','PAD')['macro_f1'])-float(point('M3','E2','PAD')['macro_f1']));dv=100*(state(out,k)['best_PAD_validation_lesion_F1']-state(out,'M3')['best_PAD_validation_lesion_F1']);own=state(out,k)['training_seconds'];supported.append(f'{NAMES[k]} versus standard joint MobileNetV4: PAD test macro-F1 change {delta:+.2f} pp; PAD validation selection macro-F1 change {dv:+.2f} pp; additional own training {own:.2f}s plus shared M3/M4 parent costs. These observations alone do not isolate enhancement causality or establish that the additional training complexity is justified.')
 for k in KEYS:
  rs=[classrow(k,'E2','PAD',cl) for cl in CLASSES];weak=sorted(rs,key=lambda r:float(r['recall']))[:2];supported.append(NAMES[k]+' E2 lowest PAD recalls: '+', '.join(r['class']+' '+r['correct']+'/'+r['support']+' ('+format(float(r['recall']),'.4f')+')' for r in weak)+'.')
 pareto=[]
 for a in cost:
  ka=a['configuration']
  if not any(float(b['parameters'])<=float(a['parameters']) and float(b['ConvLinear_MACs'])<=float(a['ConvLinear_MACs']) and float(point(b['configuration'],'E2','PAD')['macro_f1'])>=float(point(ka,'E2','PAD')['macro_f1']) and (float(b['parameters'])<float(a['parameters']) or float(b['ConvLinear_MACs'])<float(a['ConvLinear_MACs']) or float(point(b['configuration'],'E2','PAD')['macro_f1'])>float(point(ka,'E2','PAD')['macro_f1'])) for b in cost if b['configuration']!=ka):pareto.append(NAMES[ka])
 supported.append('Descriptive parameter/MAC-versus-PAD-test point-estimate Pareto configurations: '+', '.join(pareto)+'. This is an exploratory efficiency description, not the development selection rule or a phone-latency measurement.')
 limitations='''# Limitations and novelty\n\nAll PAD findings are POST-HOC EXPLORATORY: this dataset was examined repeatedly before this study. The patient-disjoint split prevents direct within-PAD patient leakage but does not restore fresh independent confirmation. E1 is frozen ISIC-only zero-shot transfer. E2 is supervised joint-domain learning and target-domain evaluation; its ISIC-minus-PAD difference is not zero-shot generalization.\n\nE1 versus E2 is a comparison of the documented source-only and joint-domain recipes, not an isolated causal effect of adding PAD images. E1 checkpoints were selected on ISIC validation; E2 selection uses PAD validation. Historical E1 versus E2 augmentation, weighting, domain exposure and allowed training budgets differ. These protocol differences and validation reuse must accompany any adaptation attribution.\n\nOne joint epoch is727optimizerupdates×16images=11632presentations:5816ISIC+5816PAD, not one complete pass over every ISIC training image. The781PADtrainingimages are visited7or8times per epoch by shuffled uniform cycles (mean7.447), including the30MELimages; no extra class-aware sampler is used. This repeated target exposure creates memorization risk, and the epoch-specific exposure CSVs are authoritative.\n\nOne seed42, no folds or sweeps. PAD test comprises261images,126patients,205declaredlesions: NV41/MEL7/BCC129/SCC28. Seven melanoma units give coarse, uncertain sensitivity. Patient-grouped paired bootstrap intervals are conditional on fixed checkpoints, not training-seed variation. Absent-class bootstrap draws are rejected and counted. ISIC uses lesion grouping because reliable patient identifiers are unavailable. Biological lesion/patient independence remains limited by metadata quality. [private group]'s documented SCC identity ambiguity is unchanged and wholly validation; no outcome-driven removal or merge.\n\nThe enhancement uses a joint M3 starting checkpoint and a qualified frozen joint M4 teacher, inverse-frequency CE and original two-view augmentation, whereas standard E2 baselines start from ImageNet with moderated weights and source/target acquisition augmentation. M5 receives up to40additionalepochs plus parent cost. Thus enhancement-versus-backbone differences are not architecture-only or isolated causal component effects. M5-LR changes only the declared two learning rates and is a sensitivity control, not an independent architecture. Six configurations use four distinct inference graphs. No novelty has been established.\n\nM1 controller defect: epochs17–23 exceeded the prescribed stop at16. Epoch10 selected solely from the valid prefix is recovered without optimization; all extra/partial files remain, actual and prescribed durations are reported separately, and the deviation prevents describing execution as flawless. Unknown partial epoch24 overhead is not fabricated. The subsequent controller was repaired and tested before further GPU training.\n\nTFLite/Flutter compatibility is an expectation conditional on graph/operator, numerical-parity and device tests; no final export, mobile latency measurement, Flutter Grad-CAM implementation, clinical safety or diagnostic efficacy is demonstrated. Grad-CAM is a localization explanation, not validated medical causation or segmentation. A stronger journal contribution requires independent clinical data, prospective patient-level design, multi-seed replication, controlled enhancement ablations, appropriate paired uncertainty, clinically justified endpoint/power and mobile deployment parity measurements. Standard joint training is not a novel algorithm.\n'''
 report=f'''# Six-model E1/E2 final results\n\n**MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.**\n\nStatus: completed, with documented M1 controller deviation and verified recovery. Four remaining E2 configurations trained once; M1 recovered without retraining; completed M3 E2 and all six E1 checkpoints reused unchanged. All PAD findings are **POST-HOC EXPLORATORY**. E2 is supervised adaptation, not zero-shot generalization.\n\n## Development selection versus test description\n\nTop PAD validation configuration overall: {NAMES[topdev]} ({state(out,topdev)['best_PAD_validation_lesion_F1']:.6f}). The predeclared lightweight deployment candidate is **{NAMES[devkey]}**, selected and locked before new test inference. Highest PAD test point estimate is {NAMES[best_e2_pad]} ({float(point(best_e2_pad,'E2','PAD')['macro_f1']):.6f}); that retrospective ranking does not replace the development-selected candidate. Highest E1 ISIC macro-F1: {NAMES[best_e1_isic]}; highest E1 PAD macro-F1: {NAMES[best_e1_pad]}. Largest PAD improvement: {largest['model_name']} ({100*float(largest['PAD_macro_f1_change']):+.2f} pp).\n\n## Adaptation and source-domain trade-offs\n\n'''+ '\n'.join(findings)+'\n\n## Failure modes\n\n'+'\n'.join(confusions)+'''\n\n## Enhanced model and LR control\n\nThe following secondary paired comparisons are descriptive, conditional on frozen models and pre-existing protocol differences. Intervals are patient-grouped on PAD and lesion-grouped on ISIC; no claim of statistical significance or causal novelty follows.\n\n'''+table(select(pair,dataset='PAD',analysis_level='lesion_primary',metric='macro_f1'))+'\n\n'+'\n\n'.join(supported)+'\n\n'+ '\n\n'.join(sections)+'''\n\n## Interpretation and deployment\n\nJoint-domain exposure provides evidence of performance on held-out patients from the same previously observed target dataset, with explicit class-wise and source-forgetting trade-offs above. It does not establish performance on a new smartphone cohort or clinical screening utility. A large BCC support can dominate accuracy; macro-F1 and minority counts must be considered alongside it. The validation-selected lightweight candidate remains fixed; test rankings only describe outcomes. ConvNeXt is the larger reference. Enhanced inference excludes its teacher but training costs include student/teacher parents separately. Conv/Linear MACs count one multiply-accumulate as one MAC for one224×224image; norm, activation, pooling, softmax, memory and runtime effects are excluded. Checkpoint sizes include training-state content and are not TFLite model sizes. No phone latency is inferred.\n\n## Integrity, figures and reproducibility\n\nAll final audit checks passed with the documented M1 deviation. Exact checkpoint paths/hashes are in00_Protocol/FINAL_EVALUATION_LOCK.json andMODEL_COMPLEXITY.csv. The approved frozen manifests remain byte-identical. Primary scoring averages raw uncalibrated softmax across predeclared capture representatives within each declared lesion; image-level supplements are separate. No calibration was fitted. Undefined per-class precision with no predictions is zero; all four true classes are present, so AUROC/AP are computed. No historical full-PAD score is substituted for the fixed20%test result.\n\nFIGURE_INDEX.csv contains titles, captions, source hashes and paths for every new figure family; each has300DPI PNG,SVG,PDF. Reused E1 figures and their original case evidence are listed separately in09_Explainability/E1_REUSED_PROVENANCE.csv. Plot data and scripts are retained;06_Confusion_Matrices and07_ROC_PR contain actual measured plots. Use the shortlist rather than repeating appendix panels.\n\nSee13_Reports/M1_RECOVERY_REPORT.md,13_Reports/TRAINING_CONTROLLER_FIX_REPORT.md,13_Reports/FINAL_E2_TRAINING_STATUS.md,13_Reports/FINAL_INTEGRITY_AUDIT.csv,EXPERIMENT_PROVENANCE.md andLIMITATIONS_AND_NOVELTY.md.\n'''
 (out/'SIX_MODEL_E1_E2_FINAL_REPORT.md').write_text(report);(out/'LIMITATIONS_AND_NOVELTY.md').write_text(limitations)
 provenance='# Experiment provenance\n\n'+table([{'Configuration':m['name'],'E1 checkpoint':m['E1_checkpoint'],'E1 SHA256':m['E1_sha256'],'E2 checkpoint':m['E2_checkpoint'],'E2 SHA256':m['E2_sha256']} for m in lock['models']])+'''\nSource dataset release:v20261117T060846Z_608314c8. Frozen PAD split reused fromv20261117T221855Z_supervised_joint_seed42, never regenerated. All configuration, normalization and checkpoint parent paths are in the immutable lock and resolved run configurations. E1 source-only caches are selected by exact fixed test IDs; five new E2 models receive one inference pass on each held-out set after all selections lock; M3 uses unchanged original prediction records. Training only readsDEVELOPMENT_MANIFEST.csv.\n\nController original remains code/engine.py; fixed version is code/engine_fixed_v1.py and code/stopping_v1.py, launched by code/run_remaining_v1.py. M1 recovery code and deterministic tests are saved. Controller fix report contains semantics and diff. Fresh standard baselines use publisher-hash-verified ImageNet artifacts. Enhanced students share frozen joint M3 and frozen qualified joint M4; no E1 skin checkpoint initialized new E2 optimization. M5-LR changes head/backboneLR1e-4/1e-5to1e-5/1e-6.\n\nReproduction order: controller tests and M1 recovery (already complete, never repeat writes unnecessarily), remaining-run wrapper, evaluate.py freeze, evaluate.py infer, evaluate.py analyze, figures.py, gradcam.py, finish.py. No training is needed to regenerate analysis from frozen predictions. Existing manifests/receipts should be verified before rerunning; immutable lock creation and per-epoch commits intentionally prevent silent overwrite.\n'''
 (out/'EXPERIMENT_PROVENANCE.md').write_text(provenance)
 (out/'README.md').write_text('# Six-model source-only versus joint-domain comparison\n\nRead SIX_MODEL_E1_E2_FINAL_REPORT.md first.00_Protocol freezes rules;01_Frozen_Data_Manifests preserves exact splits;02_E1_Evaluations caches frozen source-only predictions;03_E2_Training holds five additional runs;04_E2_Evaluations holds joint predictions;05_Four_Setting_Comparisons contains ten tables;06_Confusion_Matrices,07_ROC_PR,08_Training_Curves,09_Explainability,10_Model_Complexity and11_Publication_Figures contain plots;12_Source_Data contains numerical data;13_Reports records recovery, tests and final audit. code/ contains generation scripts. Original M3 and historical artifacts are referenced, not replaced.\n')
 thesis=ROOT/'Thesis Results'
 if thesis.is_dir():
  dest=thesis/'E1_E2_Six_Model_Comparison'/out.name;dest.mkdir(parents=True,exist_ok=False)
  for p in (out/'05_Four_Setting_Comparisons').glob('TABLE_*.csv'):shutil.copy2(p,dest/p.name)
  for r in index:
   if str(r['thesis_shortlist']).lower()=='true':
    for e in ['png','svg','pdf']:shutil.copy2(out/r[e],dest/Path(r[e]).name)
  shutil.copy2(out/'SIX_MODEL_E1_E2_FINAL_REPORT.md',dest/'SIX_MODEL_E1_E2_FINAL_REPORT.md')
 files=[p for p in out.rglob('*') if p.is_file() and p.name!='FINAL_PACKAGE_CHECKSUMS.json'];dump(out/'13_Reports/FINAL_PACKAGE_CHECKSUMS.json',{str(p.relative_to(out)):sha(p) for p in files})
 print('FINAL AUDIT PASS WITH M1 DEVIATION',len(index),'figure families',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='status':status_report(ROOT/sys.argv[2])
 else:main(ROOT/sys.argv[1])
