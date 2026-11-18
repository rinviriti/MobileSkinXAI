"""Finalize an evidence-linked proposal. Never trains or modifies source datasets."""
import csv,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'research/thesis_assets/phase4_duplicate_audit/v20261116T230847Z_34d5abf2'
N=ROOT/'research/thesis_assets/phase4_duplicate_audit_revision/v20261116T233232Z_86ade2d0'
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 rows=list(rows);p=N/name;p.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def js(name,value): (N/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def key(p,side):return (p['dataset_'+side],p['image_'+side])
pairs=read(S/'pair_audit.csv');adv={p['pair_id']:p for p in read(N/'advanced_metrics.csv')};vis={p['pair_id']:p for p in read(N/'visual_review.csv')}
base=read(ROOT/'dataset/ISIC2019/preprocessed/rgb224_letterbox_gray114_v1/full/manifest.csv');pad=read(ROOT/'dataset/PADUFES20/preprocessed/rgb224_letterbox_gray114_v1/manifest.csv')
old={p['image']:p for p in read(S/'training_exclusions.csv')}
members=read(S/'identity_group_membership.csv');groups={(r['dataset'],r['image']):r['identity_group'] for r in members}
bygroup=defaultdict(list)
for r in members:bygroup[r['identity_group']].append(r)
boundary=defaultdict(list)
for p in pairs:
 if p['both_originally_included']=='True' and p['split_a']!=p['split_b']:
  for side in 'ab':
   if p['split_'+side]=='train':boundary[groups[key(p,side)]].append(p)
clear=lambda p: p['final_decision']=='distinct_image_pair' or adv.get(p['pair_id'],{}).get('strict_negative_screen_clearance')=='True'
confirmed_groups={g for g,rs in bygroup.items() if any(r['original_split']=='train' for r in rs) and any(r['original_split'] in ('validation','test','external') for r in rs)}
oldgroups={groups[('ISIC2019',i)] for i in old}
restorable={g for g,ps in boundary.items() if all(clear(p) for p in ps)} & oldgroups - confirmed_groups
# Actual full-frame/central-crop contact sheets inspected before recording these decisions.
driving=read(N/'restoration_support/driving_pairs.csv')
assert set(r['pair_id'] for r in driving)=={p['pair_id'] for g in restorable for p in boundary[g]}
write('restoration_support/visual_clearance.csv',({**r,'classification':'visually_similar_but_distinct','evidence':'Full frame and central crops inspected at listed panel: different focal architecture, border/texture and surrounding landmarks; no matching photograph. Joint strict-negative rule also passed. Not biological/patient-identity certification.','reviewer':'assistant_visual_screening; no independent clinical adjudication'} for r in driving))
restored={r['image'] for r in base if r['split']=='train' and r['image'] in old and groups[('ISIC2019',r['image'])] in restorable}
excluded=set(old)-restored
retained=[r for r in base if r['split']!='train' or r['image'] not in excluded]
assert len(restored)==144 and len(excluded)==2050
write('manifests/ISIC2019_revised_research_manifest.csv',retained)
confirmed_images={r['image'] for r in base if r['split']=='train' and groups[('ISIC2019',r['image'])] in confirmed_groups}
assert confirmed_images=={'ISIC_0012887_downsampled','ISIC_0013163_downsampled'}
decisions=[]
for r in base:
 if r['split']!='train':continue
 i=r['image'];g=groups[('ISIC2019',i)];o=old.get(i,{})
 category=('confirmed_train_heldout_photograph_conflict' if i in confirmed_images else 'precautionary_direct_candidate' if o.get('exclusion_reason','').startswith('implicated_') else 'precautionary_known_identity_companion' if o else 'not_previously_excluded')
 decisions.append({**r,'identity_group':g,'prior_exclusion_category':category,'prior_exclusion_reason':o.get('exclusion_reason',''),'prior_evidence_pair_ids':o.get('evidence_pair_ids',''),'all_component_boundary_pair_ids':';'.join(sorted({p['pair_id'] for p in boundary[g]})),'revised_decision':'exclude' if i in excluded else 'restore' if i in restored else 'retain','justification':'Confirmed near-identical train/heldout photograph' if i in confirmed_images else 'Unresolved boundary identity risk; precaution only, not a confirmed duplicate' if i in excluded else 'Every candidate boundary edge of known identity component cleared by strict joint negative-image rule and actual contact-sheet review; biological identity not certified' if i in restored else 'No unresolved candidate boundary involving this retained known identity component in audited search scope'})
write('training_decision_table.csv',decisions)
write('exclusion_reason_audit.csv',(r for r in decisions if r['image'] in old))
write('restored_training_images.csv',(r for r in decisions if r['image'] in restored))
write('remaining_training_exclusions.csv',(r for r in decisions if r['image'] in excluded))
finalpairs=[]
for p in pairs:
 pid=p['pair_id'];v=vis.get(pid);a=adv.get(pid,{})
 classification=(v['classification'] if v else 'exact_duplicate' if p['final_decision']=='confirmed_exact_duplicate_image' else 'likely_same_lesion' if p['final_decision']=='confirmed_near_duplicate' else 'visually_similar_but_distinct' if clear(p) else 'unresolved')
 finalpairs.append({**p,**a,'revised_classification':classification,'revised_evidence':v['evidence'] if v else p['review_evidence'],'evidence_scope':'Confirmed near-identical photograph' if p['final_decision']=='confirmed_near_duplicate' else 'Exact identity hash' if classification=='exact_duplicate' else 'Image-level screening; biological/patient identity not independently certified','revised_a_retained':key(p,'a') not in {('ISIC2019',i) for i in excluded} and p['split_a']!='excluded','revised_b_retained':key(p,'b') not in {('ISIC2019',i) for i in excluded} and p['split_b']!='excluded'})
write('all_candidate_pair_decisions.csv',finalpairs)
held=[p for p in finalpairs if p['pair_id'] in vis];assert len(held)==303
vt=[p for p in held if p['dataset_a']==p['dataset_b']=='ISIC2019'];pd=[p for p in held if p not in vt]
write('validation_test_review.csv',vt)
pmeta={r['img_id']:r for r in read(ROOT/'dataset/PADUFES20/raw/metadata.csv')}
for p in pd:
 side='a' if p['dataset_a']=='PADUFES20' else 'b';m=pmeta[p['image_'+side]]
 p.update({'relationship':p['split_'+('b' if side=='a' else 'a')]+'-PAD_external','pad_patient_id_verified':m['patient_id']==p['patient_id_'+side],'pad_lesion_id_verified':m['lesion_id']==p['lesion_id_'+side],'pad_metadata':json.dumps(m,sort_keys=True),'cross_dataset_patient_linkage':'Unavailable: no global patient identifier shared with ISIC; different namespaces and acquisition modality do not prove identity independence'})
write('pad_relationship_review.csv',pd)
unresolved=sorted([p for p in held if p['revised_classification']=='unresolved'],key=lambda p:float(p['review_priority_score_not_identity_probability']),reverse=True)
assert len(unresolved)==139
write('high_risk_unresolved.csv',unresolved)
# Confirmed identity component crossing checks use every group, including raw excluded intermediaries.
active={( 'ISIC2019',r['image']):r['split'] for r in retained};active.update({('PADUFES20',r['image']):'external' for r in pad})
cross=[]
for g,rs in bygroup.items():
 splits={active[(r['dataset'],r['image'])] for r in rs if (r['dataset'],r['image']) in active}
 if len(splits)>1:cross.append({'group':g,'splits':sorted(splits)})
assert not cross
js('confirmed_identity_cross_partition_check.json',{'crossing_groups':cross,'basis':'Known source lesion IDs (PAD patient+lesion), original exact hashes, previously confirmed near-identical photographs, including excluded-raw intermediaries','does_not_assert':'Unresolved biological identity or global patient independence'})
# Proposed evaluation risk graph: joins unresolved evaluation edges and patient-complete PAD components.
parent={g:g for g in bygroup}
def find(g):
 while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
 return g
def union(a,b):parent[find(a)]=find(b)
for p in unresolved:union(groups[key(p,'a')],groups[key(p,'b')])
patients=defaultdict(list)
for r in pad:patients[r['patient_id']].append(groups[('PADUFES20',r['image'])])
for gs in patients.values():
 for g in gs[1:]:union(gs[0],g)
risk_splits=defaultdict(set)
for k,sp in active.items():risk_splits[find(groups[k])].add(sp)
elig=[]
for k,sp in active.items():
 if sp=='train':continue
 rg=find(groups[k]);ss=risk_splits[rg]
 reason=('quarantine_test_component_linked_to_validation' if sp=='test' and 'validation' in ss else 'quarantine_PAD_patient_component_linked_to_ISIC' if sp=='external' and bool(ss & {'train','validation','test'}) else 'eligible_under_proposed_risk_quarantine')
 elig.append({'dataset':k[0],'image':k[1],'original_split':sp,'class':next(r['class'] for r in bygroup[groups[k]] if r['dataset']==k[0] and r['image']==k[1]),'identity_group':groups[k],'risk_component':rg,'proposed_score_eligible':not reason.startswith('quarantine'),'reason':reason,'status':'PROPOSAL_NOT_ACTIVATED; original assignment preserved'})
write('evaluation_eligibility_PROPOSED.csv',elig)
classes=['NV','MEL','BCC','SCC'];counts=[]
for sp in ['train','validation','test']:
 for c in classes:counts.append({'split':sp,'class':c,'count':sum(r['split']==sp and r['label']==c for r in retained)})
write('class_counts.csv',counts)
traincounts={r['class']:r['count'] for r in counts if r['split']=='train'};total=sum(traincounts.values())
js('class_weights.json',{'class_order':classes,'formula':'retained_train_total / (4 * retained_train_class_count)','counts':traincounts,'weights':{c:total/(4*n) for c,n in traincounts.items()},'status':'PROPOSED; data approval false'})
# Provenance receipts preserved verbatim as report evidence, without source edits.
receipts={str(p.relative_to(ROOT)):json.loads(p.read_text()) for p in (ROOT/'dataset/PADUFES20').rglob('*.source.json')}
js('pad_provenance_receipts.json',receipts)
checks={}
for name,snap in [('prior_audit_and_original_metadata',json.loads((N/'input_snapshot.json').read_text())),('model_configs_and_weights',json.loads((S/'model_preservation_snapshot.json').read_text()))]:
 errors=[p for p,h in snap.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h];assert not errors,errors
 checks[name]={'files_checked':len(snap),'hash_mismatches':errors}
assert [r for r in retained if r['split']!='train']==[r for r in base if r['split']!='train']
checks['heldout_rows_unchanged']=True
js('preservation_checks.json',checks)
summary={'approval':False,'prior_exclusions':dict(Counter(r['prior_exclusion_category'] for r in decisions if r['image'] in old)),'restored_training_count':len(restored),'restored_training_per_class':dict(Counter(r['label'] for r in decisions if r['image'] in restored)),'remaining_training_exclusions':len(excluded),'confirmed_training_exclusions':len(confirmed_images),'precautionary_training_exclusions':len(excluded)-len(confirmed_images),'proposed_split_counts':{sp:sum(r['split']==sp for r in retained) for sp in ['train','validation','test']},'proposed_train_per_class':traincounts,'validation_test_classifications':{c:sum(p['revised_classification']==c for p in vt) for c in ['exact_duplicate','likely_same_lesion','visually_similar_but_distinct','unresolved']},'PAD_classifications':{c:sum(p['revised_classification']==c for p in pd) for c in ['exact_duplicate','likely_same_lesion','visually_similar_but_distinct','unresolved']},'PAD_relationships':dict(Counter(p['relationship'] for p in pd)),'unresolved_heldout_pairs':len(unresolved),'evaluation_proposal_eligible_counts':dict(Counter(r['original_split'] for r in elig if r['proposed_score_eligible'])),'evaluation_proposal_quarantine_counts':dict(Counter(r['original_split'] for r in elig if not r['proposed_score_eligible'])),'known_confirmed_cross_partition_groups':0,'limits':'Image-level assistant screening is not independent clinical or patient-level identity verification; candidate search is finite; unchanged heldout partitions still contain unresolved risks. No training approval.'}
js('summary.json',summary)
protocol='''# Proposed held-out evaluation protocol — NOT ACTIVATED

Preserve original split labels and files. The revised manifest changes training eligibility only. Freeze the eligibility CSV, rule set and hashes before any final training or model selection.

Known-identity groups join declared ISIC lesion IDs, PAD patient+lesion IDs, exact file/decoded hashes and confirmed near-identical photographs (including raw excluded intermediaries). None cross the retained partitions. Continue excluding the two confirmed training conflicts and every remaining uncertain training component.

For unresolved evaluation links, join known-identity components into a separate risk graph. Preserve validation for model selection. Quarantine test scoring members of any component linked to validation; do not move them into another split. Quarantine complete PAD patient groups linked by unresolved evidence to any ISIC partition. The accompanying eligibility CSV computes this proposal without applying it. Report excluded scoring groups and class denominators explicitly. This conservative quarantine addresses known candidate uncertainty, not all possible unknown leakage.

Predeclare lesion-level probability aggregation and one contribution per lesion; collapse exact repeated captures before aggregation using a deterministic ID-based representative. Use patient-level bootstrap for PAD, lesion-level bootstrap for ISIC (patient IDs unavailable). Image-level metrics may be secondary. Never use test/PAD labels, performance or images for optimization or protocol tuning. Keep PAD as external smartphone evaluation, not validation.

Before approval, either adjudicate unresolved identity links with stronger provenance/landmarks or explicitly review and lock this conservative scoring quarantine. Current unchanged heldout assignments are not certified clean. No automatic image similarity threshold proves different biological lesions, and same-patient independence cannot be verified across ISIC sources without patient identifiers.
'''
(N/'EVALUATION_PROTOCOL_PROPOSED.md').write_text(protocol)
report=f'''# Revised MobileSkinXAI duplicate audit

**Not approved for final training.** This version is a proposal; original source files, split assignments and previous audit outputs are preserved. No model was trained.

The previous 2,194 exclusions comprise **2 confirmed train–heldout near-identical photographs**, **840 direct precautionary exclusions**, and **1,352 precautionary known-identity companions**. They are not 2,194 confirmed duplicates. Confirmed training images remain excluded: ISIC_0012887_downsampled (validation partner ISIC_0012989_downsampled; P000560) and ISIC_0013163_downsampled (test partner ISIC_0013109_downsampled; P000595).

All 4,826 original candidates have a revised decision record; 2,504 originally included boundary candidates received additional central SSIM over eight orientations, texture and masked SIFT geometry. The strict negative rule was recorded before scoring in policy.json. It clears 64 candidate image edges; only 23 excluded training identity components have every boundary edge cleared. Actual contact-sheet inspection supports restoring **144 training images: NV21, MEL113, BCC10, SCC0**. Larger restored groups include known-lesion companion photographs, not 144 independently adjudicated biological identities. Remaining exclusions: **2,050 (2 confirmed, 2,048 precautionary)**.

Proposed train **11,636** (NV6840, MEL2613, BCC1799, SCC384), validation **2,835**, test **2,941**. Validation/test records are unchanged byte-for-byte at row level. Train-only inverse-frequency weights are in class_weights.json.

All 303 previously unresolved heldout pairs were visually screened with full-image/central-crop sheets. Validation–test: **105 visually distinct at image screening, 122 unresolved**. PAD-related: **59 visually distinct, 17 unresolved**; 34 validation–PAD and 42 test–PAD relationships involve 24 PAD images/patients. No exact or defensibly likely-same-lesion cases were newly confirmed among these 303. Namespaced patient/lesion IDs and official source receipts were checked, but ISIC has no globally linkable patient ID. Different modalities and missing demographics limit cross-source inference.

Prioritized unresolved evidence is in high_risk_unresolved.csv and contact_sheets/. Leading validation–test cases include P003185 and P004136; P004033 remains unresolved despite 12 apparent SIFT inliers because coverage is only 1.65% and aligned central SSIM is near zero. Enlarged comparisons are in high_risk_full_image_comparisons.jpg. Visual screening is assistant image review, not independent clinical adjudication; low SSIM, unlike labels, or different declared lesion IDs alone cannot establish biological independence.

No known/confirmed identity component crosses retained partitions, but **139 unresolved heldout pairs remain (122 validation–test,17 PAD)**. Consequently approval remains false. EVALUATION_PROTOCOL_PROPOSED.md and evaluation_eligibility_PROPOSED.csv propose explicit lesion/patient-complete scoring quarantine without changing assignments. Proposed eligible scoring counts: {summary['evaluation_proposal_eligible_counts']}; proposed quarantine counts: {summary['evaluation_proposal_quarantine_counts']}. These are not activated or substituted for original split counts.

Review exclusion_reason_audit.csv, training_decision_table.csv, restoration_support/visual_clearance.csv, validation_test_review.csv and pad_relationship_review.csv for exhaustive evidence. The manifest is manifests/ISIC2019_revised_research_manifest.csv. Automated candidate search and image review cannot guarantee elimination of unknown leakage; remaining uncertainty and evaluation policy must be resolved/locked before final training approval.
'''
(N/'REVISED_AUDIT_REPORT.md').write_text(report)
approval={'approved':False,'reason':'139 unresolved heldout candidate pairs; revised manifest and conservative scoring-quarantine protocol remain proposed, not locked','manifest':str((N/'manifests/ISIC2019_revised_research_manifest.csv').relative_to(ROOT)),'report':str((N/'REVISED_AUDIT_REPORT.md').relative_to(ROOT)),'summary':str((N/'summary.json').relative_to(ROOT))}
approval.update({'status':'blocked','manifest_path':approval['manifest'],'manifest_sha256':sha(N/'manifests/ISIC2019_revised_research_manifest.csv'),'review_document':approval['report'],'remaining_identity_review_pairs':139,'unresolved_blockers':['122 validation-test identity cases unresolved','17 PAD-related identity cases unresolved','Evaluation quarantine proposal requires review and locking before approval'],'known_patient_independence':'unverified; globally linkable ISIC patient identifiers unavailable'})
js('data_approval.json',approval)
activeapproval=ROOT/'final_training/data_approval.json';backup=N/'previous_active_data_approval.json'
if not backup.exists():backup.write_bytes(activeapproval.read_bytes())
activeapproval.write_text(json.dumps(approval,indent=2)+'\n')
print(json.dumps(summary,indent=2))
if __name__=='__main__':pass
