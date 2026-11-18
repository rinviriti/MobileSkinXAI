from common import *
from PIL import Image
import torch,timm,shutil,collections,xml.etree.ElementTree as ET

def table(rows,fields):return '| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+'\n'.join('| '+' | '.join(str(r.get(k,'')) for k in fields)+' |' for r in rows)+'\n'
def prepare(out):
 out=Path(out);status=load(out/'03_E2_Training/TRAINING_COMPLETE.json');e1=load(out/'00_Protocol/E1_PROVENANCE.json');cfg=load(out/'00_Protocol/E2_CONFIG.json');torch.set_num_threads(4);net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=4).eval();mac=[0];hooks=[]
 def hook(m,inp,v):
  if isinstance(m,torch.nn.Conv2d):mac[0]+=v.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
  else:mac[0]+=v.numel()*m.in_features
 for m in net.modules():
  if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):hooks.append(m.register_forward_hook(hook))
 with torch.no_grad():assert net(torch.zeros(1,3,224,224)).shape==(1,4)
 params=sum(p.numel() for p in net.parameters());assert params==2498148 and mac[0]==184730688
 rows=[]
 for key,cp,seconds,best,epochs in [('E1',e1['checkpoint'],367.30055779199756,18,24),('E2',status['best_checkpoint'],status['training_seconds'],status['best_epoch'],status['epochs'])]:rows.append({'model':key,'architecture':cfg['architecture'],'parameters':params,'ConvLinear_MACs':mac[0],'training_seconds':seconds,'best_epoch':best,'epochs':epochs,'checkpoint':cp,'checkpoint_sha256':sha(ROOT/cp),'checkpoint_bytes':(ROOT/cp).stat().st_size,'TFLite_status':'same plain inference architecture; export not performed in this study'})
 # Source E1 timing/epoch metadata from immutable training package, never estimated.
 source=next(r for r in read(ROOT/'research/thesis_assets/final_training_results/v20261117T091430Z_704ed3cc/11_Source_Data/model_metrics.csv') if r['model']=='M3');assert float(source['training_wall_seconds'])==rows[0]['training_seconds'];assert int(source['best_epoch'])==rows[0]['best_epoch'];write(out/'MODEL_EFFICIENCY.csv',rows)
 # Verify/log actual balanced-domain and rare-class repeat exposure.
 dev=[r for r in read(out/'11_Source_Data/DEVELOPMENT_MANIFEST.csv') if r['study_split']=='train'];meta={r['image']:r for r in dev};cumulative=collections.Counter();exposure=[]
 for ep in range(1,status['epochs']+1):
  rows=read(out/f'03_E2_Training/exposure_epoch_{ep:03d}.csv')
  for r in rows:cumulative[r['image']]+=int(r['presentations'])
  assert sum(int(r['presentations']) for r in rows if r['domain']=='ISIC')==5816 and sum(int(r['presentations']) for r in rows if r['domain']=='PAD')==5816
  for d in ['ISIC','PAD']:
   for c in CLASSES:
    rs=[r for r in rows if r['domain']==d and r['class']==c];values=[int(r['presentations']) for r in rs];exposure.append({'epoch':ep,'domain':d,'class':c,'unique_available':len(rs),'presentations':sum(values),'min_per_image':min(values),'max_per_image':max(values)})
  # Every individual batch has8+8, not merely balanced epoch totals.
  batch=[]
  with (out/f'03_E2_Training/augmentation_epoch_{ep:03d}.jsonl').open() as f:
   for line in f:
    batch.append(json.loads(line)['domain'])
    if len(batch)==16:assert batch.count('ISIC')==batch.count('PAD')==8;batch=[]
  assert not batch
 write(out/'11_Source_Data/EXPOSURE_SUMMARY.csv',exposure);write(out/'11_Source_Data/CUMULATIVE_IMAGE_EXPOSURE.csv',[{'image':name,'domain':meta[name]['domain'],'class':meta[name]['label'],'presentations':count} for name,count in sorted(cumulative.items())])
 # Reference existing verified ORIGINAL M3 validation Grad-CAM; no new E1 heatmaps fabricated.
 xai=ROOT/'research/thesis_assets/final_training_results/v20261117T091430Z_704ed3cc/10_Explainability';records=[]
 for ext in ['png','svg','pdf']:
  source=xai/('10_M3_GradCAM_CUDA_BF16.'+ext);assert source.exists();records.append({'model':'E1 original M3','source':str(source.relative_to(ROOT)),'sha256':sha(source),'scope':'existing original ISIC validation Grad-CAM; not new PAD test examples'})
 write(out/'09_Explainability/E1_EXISTING_XAI_REFERENCE.csv',records)
 print('EFFICIENCY AND EXPOSURE VERIFIED',flush=True)
def finish(out):
 out=Path(out);status=load(out/'03_E2_Training/TRAINING_COMPLETE.json');cfg=load(out/'00_Protocol/E2_CONFIG.json');e1=load(out/'00_Protocol/E1_PROVENANCE.json');assert load(out/'09_Explainability/GRADCAM_COMPLETE.json')['status']=='PASS'
 for name,h in load(out/'00_Protocol/TRAINING_LOCK.json')['sha256'].items():assert sha(out/name)==h,name
 for name,h in load(out/'11_Source_Data/PRESERVATION_HASHES.json').items():assert sha(ROOT/name)==h,name
 for commit in (out/'03_E2_Training').glob('commit_*.json'):
  r=load(commit);assert sha(out/'03_E2_Training/checkpoints'/r['checkpoint'])==r['sha256']
 assert sha(ROOT/status['best_checkpoint'])==status['best_checkpoint_sha256'];assert sha(ROOT/e1['checkpoint'])==e1['checkpoint_sha256'];split=read(out/'01_PAD_Splits/PAD_SPLIT.csv');pidgroups=collections.defaultdict(set)
 for r in split:pidgroups[r['patient_id']].add(r['study_split'])
 assert all(len(v)==1 for v in pidgroups.values());# Independent postflight reconciliation from existing saved probabilities; no inference.
 from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score,balanced_accuracy_score
 metricsrows=read(out/'05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv');classrows_saved=read(out/'CLASSWISE_RESULTS.csv');data=read(out/'11_Source_Data/DEVELOPMENT_MANIFEST.csv');padtest=read(out/'01_PAD_Splits/PAD_test.csv');isictest=read(out/'11_Source_Data/ISIC_TEST_MANIFEST.csv')
 training=[r for r in data if r['study_split']=='train'];heldout=[r for r in data if r['study_split']=='validation']+padtest+isictest
 for field in ['identity_group','risk_component','source_sha256','oriented_pixel_sha256']:assert not {r[field] for r in training}&{r[field] for r in heldout}
 # Verify every PADtestpatient is absent from E2 train/development selection.
 testpatients={r['patient_id'] for r in padtest};assert not testpatients&{r['patient_id'] for r in data if r['domain']=='PAD'}
 for r in data+padtest+isictest:assert sha(ROOT/r['processed_path'])==r['processed_sha256'] and sha(Path(r['source']))==r['source_sha256']
 for key,folder in [('E1','02_E1_Frozen_Evaluation'),('E2','04_E2_Evaluation')]:
  for domain,cohort in [('ISIC',isictest),('PAD',padtest)]:
   image=read(out/f'{folder}/{key}_{domain}_image.csv');assert [r['image'] for r in image]==[r['image'] for r in cohort];assert len({r['image'] for r in image})==len(cohort)
   receipt=load((out/f'{folder}/{key}_{domain}_image.csv').with_suffix('.receipt.json'));assert sha(out/f'{folder}/{key}_{domain}_image.csv')==receipt['sha256']
   group=collections.defaultdict(list)
   for r in image:
    if r['use_in_lesion_probability_mean']=='True':group[r['lesion_scoring_unit']].append(r)
   lesions=read(out/f'{folder}/{key}_{domain}_lesion.csv');assert {r['lesion_scoring_unit'] for r in lesions}==set(group)
   for r in lesions:assert np.array_equal([float(r[c]) for c in CLASSES],np.mean([[float(z[c]) for c in CLASSES] for z in group[r['lesion_scoring_unit']]],axis=0))
   for level,records in [('image_secondary',image),('lesion_primary',lesions)]:
    y=np.array([int(r['true_index']) for r in records]);prb=np.array([[float(r[c]) for c in CLASSES] for r in records]);assert np.isfinite(prb).all() and np.allclose(prb.sum(1),1,atol=1e-6);pred=prb.argmax(1);cm=confusion_matrix(y,pred,labels=range(4));pr,re,f,support=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0);saved=next(r for r in metricsrows if r['model']==key and r['dataset']==domain and r['analysis_level']==level)
    for metric,value in [('accuracy',accuracy_score(y,pred)),('macro_f1',f.mean()),('weighted_f1',np.average(f,weights=support)),('balanced_accuracy',balanced_accuracy_score(y,pred))]:assert abs(float(saved[metric])-value)<1e-12
    for j,c in enumerate(CLASSES):
     saved=next(r for r in classrows_saved if r['model']==key and r['dataset']==domain and r['analysis_level']==level and r['class']==c);assert int(saved['correct'])==cm[j,j] and int(saved['support'])==support[j]
     for metric,value in [('precision',pr[j]),('recall',re[j]),('f1',f[j])]:assert abs(float(saved[metric])-value)<1e-12
 summary=read(out/'05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv');side=read(out/'E1_E2_SIDE_BY_SIDE_RESULTS.csv');cw=read(out/'CLASSWISE_RESULTS.csv');pairs=read(out/'05_Cross_Dataset_Comparison/PAIRED_DIFFERENCES.csv');gaps=read(out/'GENERALIZATION_COMPARISON.csv');eff=read(out/'MODEL_EFFICIENCY.csv');idx=read(out/'FIGURE_INDEX.csv');counts=read(out/'01_PAD_Splits/SPLIT_COUNTS.csv')
 # Add actual E2 development Grad-CAM exports to publication index without regenerating result figures.
 if not any(r['figure_id']=='E2_development_GradCAM' for r in idx):
  r={'figure_id':'E2_development_GradCAM','title':'Joint ISIC+PAD MobileNetV4-Conv-Small development Grad-CAM','caption':'Actual PAD development-validation correct/error cases, median-confidence rule; predicted-logit Grad-CAM at blocks.4. Missing/blank strata documented. Not clinical localization or Flutter inference.','source_data':'09_Explainability/GRADCAM_CASES.csv','source_sha256':sha(out/'09_Explainability/GRADCAM_CASES.csv'),'script':'code/gradcam.py','PNG_DPI':300,**{ext:'09_Explainability/E2_validation_GradCAM.'+ext for ext in ['png','svg','pdf']}};idx.append(r)
  for ext in ['png','svg','pdf']:shutil.copyfile(out/r[ext],out/'10_Publication_Figures'/Path(r[ext]).name)
  write(out/'FIGURE_INDEX.csv',idx)
 for r in idx:
  assert sha(out/r['source_data'])==r['source_sha256']
  for ext in ['png','svg','pdf']:
   p=out/r[ext]
   if ext=='png':
    with Image.open(p) as im:assert min(im.info['dpi'])>=299;im.verify()
   elif ext=='svg':ET.parse(p)
   else:assert p.read_bytes().startswith(b'%PDF-')
 def point(k,d,m):return float(next(r for r in summary if r['model']==k and r['dataset']==d and r['analysis_level']=='lesion_primary')[m])
 padgain=next(r for r in pairs if r['dataset']=='PAD' and r['analysis_level']=='lesion_primary' and r['metric']=='macro_f1');table2=[{k:(f'{float(v):.4f}' if k!='Metric' else v) for k,v in r.items()} for r in side];padclasses=[r for r in cw if r['dataset']=='PAD' and r['analysis_level']=='lesion_primary'];classrows=[{'Model':r['model'],'Class':r['class'],'Correct/total':r['correct']+'/'+r['support'],'Precision':f"{float(r['precision']):.4f}",'Recall':f"{float(r['recall']):.4f}",'F1':f"{float(r['f1']):.4f}"} for r in padclasses];gaprows=[{'Model':r['model'],'ISIC−PAD macro-F1':f"{float(r['ISIC_minus_PAD_macro_F1']):+.4f}",'95% grouped CI':f"[{float(r['CI_low']):+.4f},{float(r['CI_high']):+.4f}]",'Interpretation':r['interpretation']} for r in gaps if r['analysis_level']=='lesion_primary']
 limitations='''# Limitations

All PAD findings are POST-HOC EXPLORATORY because prior whole-PAD results influenced research direction. New patient-disjoint held-out partitions do not erase previous exposure. E1 is zero-shot ISIC-to-PAD transfer; E2 is ordinary supervised joint-domain training with labeled target-domain exposure, not domain-generalization novelty.

PAD test has only7melanoma declared lesions from7patients and28SCClesions; one melanoma error changes recall by14.29percentage points. Development validation likewise has7melanoma lesions. Patient-grouped uncertainty is wide; absent-class bootstrap candidates are rejected and counted, so reported intervals condition on allclassespresent. One seed, no cross-validation, no significance or clinical screening claim.

Image-stratified60/20/20 produced781/261/261images,550/127/126patients,616/205/205declared lesions. Patient fractions differ because patientimageburden varies and image/class balance was prioritized. Metadata grouping does not prove biological independence. [private group]duplicate SCC photograph spans2declared lesion IDs, is wholly validation and is retained as frozen metadata; patient clustering is preserved, no hidden merge. ISIC global patient linkage unavailable; known identity/quarantine audits cannot detect every unknown relationship.

50:50domain sampling repeatedly exposes each of781PADtrainimages roughly7–8times per epoch. No melanoma-specific oversampling; exact logs retained. Multiple views/patients with many photos have higher exposure, and memorization remains possible. Augmentation cannot replace independent lesions or patients. Class weights use training-domain mixture only.

E1 is a frozen historical baseline, not retrained matched control: original inverse class weighting/augmentations/ISICimage-validation selection andup30epochs differ from E2moderated weights/50:50sampling/PADlesion-validation selection andup35epochs. Differences cannot be attributed solely to addingPAD data. Actual timing budgets vary with pipeline, hardware state and data loading. E1 uses provenance-verified cached historical probabilities; E2 uses new batch16inference on sameimages. Both use the exact same deterministic prepared224RGB inference pipeline; historical batchcontext differs and BF16 numerical rounding may vary slightly.

E1 gap is zero-shot performance drop; E2 arithmetic difference is after supervisedPADexposure, NOT zero-shot generalization. Evaluate absolute source/target scores and every class. BCCdominance can inflate accuracy. No test outcome selected checkpoints, thresholds, seeds or extra runs. No standard joint-training novelty claim.

Same plain MobileNetV4 graph supports lightweight deployment expectations; no trained TFLite conversion, phone latency, Flutterintegration or on-deviceGradCAM implemented. Explanations are qualitative gradients, not clinical validation. A genuine unused external smartphonecohort, prospective/clinical validation, matched replicated controls, stronger novelty differentiation and actual device/export measurements remain needed for stronger journal/deployment claims.
'''
 (out/'LIMITATIONS.md').write_text(limitations)
 report='''# E1 versus E2 supervised joint-domain study

MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.

**POST-HOC EXPLORATORY PAD findings.** E1=frozen original final MobileNetV4 M3, no retraining. E2=one new officialImageNet-initialized MobileNetV4 seed42 jointly trained on ISIC+PAD. This is supervised target-domain exposure, not zero-shot domain generalization or a new algorithm.

## Table1: frozen PAD split

'''+table(counts,list(counts[0]))+'\n## Table2: primary declared-lesion comparison\n\n'+table(table2,list(table2[0]))+'\n## Table3: performance differences and PAD improvement\n\n'+table(gaprows,list(gaprows[0]))+f"\nE2 PAD macro-F1 minus E1: **{float(padgain['E2_minus_E1']):+.4f}** ({100*float(padgain['E2_minus_E1']):+.2f}pp), patient-grouped percentile95%CI[{float(padgain['CI_low']):+.4f},{float(padgain['CI_high']):+.4f}]. These are descriptive paired intervals on this previously examined cohort, not independent confirmation.\n"+'\n## Table4: PAD class precision/recall/F1\n\n'+table(classrows,list(classrows[0]))+'\n## Table5: same inference architecture, measured training budgets\n\n'+table(eff,list(eff[0]))+f'''
## Actual training and development selection

E2completed{status['epochs']}epochs, best{status['best_epoch']}, trainingwall{status['training_seconds']/60:.2f}minutes, peakallocated{status['peak_allocated_GiB']:.3f}GiB. Selected only byPADdevelopmentdeclared-lesion macro-F1({status['best_PAD_validation_lesion_F1']:.4f}); image metrics andISICvalidation diagnostic inhistory. No testmetric entered training. Best andlast plusall epoch optimizer/RNG/checkpoint hashes are preserved. Benchmark model was disposable and not used for actualE2initialization.

## Thesis-ready interpretation

ISIC-only transfer to the new PADtest has macro-F1{point('E1','PAD','macro_f1'):.4f}, accuracy{point('E1','PAD','accuracy'):.4f}, balancedaccuracy{point('E1','PAD','balanced_accuracy'):.4f}. Jointtraining has correspondingvalues{point('E2','PAD','macro_f1'):.4f}, {point('E2','PAD','accuracy'):.4f}, {point('E2','PAD','balanced_accuracy'):.4f}. PADmacro-F1change{100*(point('E2','PAD','macro_f1')-point('E1','PAD','macro_f1')):+.2f}pp; balancedaccuracychange{100*(point('E2','PAD','balanced_accuracy')-point('E1','PAD','balanced_accuracy')):+.2f}pp.

MELrecall changes{point('E1','PAD','MEL_recall'):.4f}→{point('E2','PAD','MEL_recall'):.4f}; SCCrecall{point('E1','PAD','SCC_recall'):.4f}→{point('E2','PAD','SCC_recall'):.4f}. SourceISICmacro-F1 changes{point('E1','ISIC','macro_f1'):.4f}→{point('E2','ISIC','macro_f1'):.4f} ({100*(point('E2','ISIC','macro_f1')-point('E1','ISIC','macro_f1')):+.2f}pp). These measurements quantify both target benefit/deterioration and source retention rather than optimizing a target headline. Inspect allTable4classes: minorityrecalls, precision andwrong-classconfusions determine whether any gain is majorityBCCdominated.

The PADtest confusion counts show40additional correct lesions:32BCC,4MEL,4SCC and0NV. Thus80%of the accuracy-count gain comes from the majorityBCCclass, but macro-F1improvement is not merely majority-class inflation: MELclassF1rises0.0476→0.8333, NV0.7200→0.8675, BCC0.7203→0.8603, SCC0.1250→0.2791. SCCstillmisses22/28lesions and SCC→BCCerrorsincrease16→22; jointtraining reduces SCC→NV/MEL errors while leaving a serious BCCconfusion failure. E2MELprecision1.0 reflects only5predicted melanoma lesions in this smallpost-hoc subset, not established clinical predictive value.

Source retention is materially worse: ISIC macro-F1drops7.32pp, MELrecall62.11%→48.42%, SCC31.58%→15.79%. Paired source lesion-bootstrap intervals in05_Cross_Dataset_Comparison/PAIRED_DIFFERENCES.csv are negative for these metrics. TargetPADmacro-F1gain30.68pp has a paired patient interval[19.26,39.78]pp; SCCrecallgain14.29pp has interval[-3.23,31.82]pp and is uncertain. No unsupported significance or clinical claim is made. E2's negative ISIC-minus-PADdifference means higher observed targetmacro-F1after targetsupervision, not superior zero-shot generalization.

Original release optimization_allowed/model_selection_allowed fields remain historical source-only-study metadata and are unchanged. The new user-authorized supervised-adaptation protocol and study_split field define E2PADdevelopment eligibility; no old Phase6external role was silently relabeled.

The defensible contribution is a transparent frozen-source-versus-supervisedjointdomain comparison, patient-disjoint PAD split, exact exposure/provenance accounting, lesion-level outcomes and honest source/target trade-offs. Ordinary jointtraining is not an algorithmic novelty. A practical mobile benefit can only be suggested by measured target improvement at unchanged inference complexity; clinical adequacy, unseen-cohort transfer, realphone performance and prospective screening utility remain unproven. SeeLIMITATIONS.md for the full Q1journal evidence gaps. No furthertraining is automatically triggered.

## Explanations and publication files

Existing original M3 ISIC-validation GradCAM references and hashes are recorded without regenerating outputs. NewE2GradCAM uses onlyPADdevelopmentvalidation images, deterministicmedianconfidence correct/error examples, actual predicted-logit gradient atblocks.4, original BF16batchcontext parity. Missingstrata/blankmaps documented; no fabricatedheatmaps, segmentation orFlutterclaim. Alltestdecisions stayedlocked. FIGURE_INDEX.csv andFIGURE_CAPTIONS.md listactualPNG300DPI/SVG/PDF andsources, including lossdefinition/image-vs-lesion labels.

Image-level supplementary results, classspecificity/AUC/AP, NLL/Brier/ECE andallraw/normalizedconfusioncounts are in05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv, CLASSWISE_RESULTS.csv and06_Confusion_Matrices/CONFUSION_COUNTS.csv. Pairedbootstrapvalidreplicates2000; actual rejected absent-classcounts:{json.dumps(load(out/'05_Cross_Dataset_Comparison/METRICS_COMPLETE.json')['rejected_absent_class_candidates'])}. Knownidentitygroups/lesions/patients cannot cross newPADpartitions; originalISICsplit andallpriorartifacts remain unchanged.
'''
 (out/'E1_E2_EXPERIMENT_REPORT.md').write_text(report);(out/'EXPERIMENT_PROVENANCE.md').write_text('# Provenance\n\n'+json.dumps({'E1':e1,'E2':status,'split_lock':load(out/'00_Protocol/SPLIT_LOCK.json'),'ImageNet_initialization':load(out/'00_Protocol/E2_CONFIG.json')['pretrained_artifact'],'no_E3_folds_multiseed_or_export':True},indent=2)+'\n');(out/'FIGURE_CAPTIONS.md').write_text('\n\n'.join(f"## {r['figure_id']}\n\n{r['caption']}" for r in idx)+'\n')
 rootresults=ROOT/'Thesis Results'
 if rootresults.is_dir():
  target=rootresults/'E1_E2_supervised_joint_v20261117T221855Z';target.mkdir(exist_ok=True)
  for source in list((out/'10_Publication_Figures').glob('*'))+[out/'E1_E2_SIDE_BY_SIDE_RESULTS.csv',out/'CLASSWISE_RESULTS.csv',out/'GENERALIZATION_COMPARISON.csv',out/'MODEL_EFFICIENCY.csv']:
   dest=target/source.name
   if dest.exists():assert sha(dest)==sha(source)
   else:shutil.copyfile(source,dest)
 dump(out/'12_Reports/POSTFLIGHT.json',{'status':'PASS','E1_frozen_original_M3':True,'new_full_training_runs':1,'seed':42,'PAD_patients_disjoint':True,'all_previous_artifacts_hash_verified':True,'all_epoch_hashes_verified':True,'figures':len(idx),'test_selection_or_calibration':False,'PAD_status':'POST-HOC EXPLORATORY','Flutter_or_TFLite_export':False,'independent_probability_metric_reconciliation':True,'all_train_development_test_identity_boundaries_verified':True,'all_current_source_prepared_hashes_verified':True})
 (out/'PACKAGE_CHECKSUMS.sha256').write_text('\n'.join(f'{sha(p)}  {p.relative_to(out)}' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='PACKAGE_CHECKSUMS.sha256')+'\n');print('E1/E2 FINAL PACKAGE VERIFIED',out,flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','finish']);p.add_argument('output');a=p.parse_args();(prepare if a.mode=='prepare' else finish)(ROOT/a.output)
