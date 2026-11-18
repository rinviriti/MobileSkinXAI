"""Descriptive predeclared component contrasts; never loads images or model checkpoints."""
from common import *
import shutil
out=ROOT/sys.argv[1];sealcheck(out);assert load(out/'07_Cross_Dataset_Exploratory/POSTHOC_COMPLETE.json')['status']=='PASS'
val={r['model']:r for r in read(out/'ABLATION_COMPARISON.csv')};post={(r['model'],r['split']):r for r in read(out/'07_Cross_Dataset_Exploratory/EXPLORATORY_RESULTS.csv') if r['level']=='lesion_primary'};order=['M3','M5']+KEYS
contrasts=[]
for a,b,label in [('M5','B1','Full reliability mechanism (hard+continuous; includes auxiliary attenuation)'),('B1','B2','Teacher KD conditional on shared ungated student JS'),('B3','M3','Separate joint source-augmentation/supervised-only robustness intervention')]:
 r={'contrast':NAMES[a]+' minus '+NAMES[b],'interpretation':label,'validation_image_macro_f1_difference':float(val[a]['macro_f1'])-float(val[b]['macro_f1'])}
 for split in ['test','external']:
  rep=np.load(out/f'07_Cross_Dataset_Exploratory/{split}_lesion_primary_bootstrap.npy');diff=rep[:,order.index(a),1]-rep[:,order.index(b),1];lo,hi=np.percentile(diff,[2.5,97.5]);r[split+'_lesion_macro_f1_difference']=float(post[(a,split)]['macro_f1'])-float(post[(b,split)]['macro_f1']);r[split+'_CI_low']=float(lo);r[split+'_CI_high']=float(hi)
 contrasts.append(r)
write(out/'11_Source_Data/PREDECLARED_COMPONENT_CONTRASTS.csv',contrasts)
text='''\n## Component and robustness interpretation

Reliability gating: original full MobileSkinXAI-Enhanced and its no-reliability counterpart have nearly identical validation macro-F1 (0.6912 versus0.6905), with the same selected epoch1 and realized7epochs. Original gating reduces NV→MEL errors221versus240, while the ungated arm has higher MEL/SCC recall. Both hard gating and continuous reliability are removed together, and auxiliary attenuation changes; these data do not establish that reliability improves macro-F1. In exploratory PAD reuse the ungated point macro-F1 is higher (0.4133 versus0.3977), with the paired interval against full enhancement including zero. No gating benefit or harm is established generally.

Teacher guidance: B1 versus B2 is the controlled KD contrast under ungated student consistency. KD raises validation macro-F1 from0.6763to0.6905 and reduces NV→MEL errors321to240, but decreases melanoma recall0.7782to0.7355 and SCC recall0.4211to0.3895. It is a precision/recall tradeoff, not uniform clinical improvement. Exploratory external point differences are also recorded in PREDECLARED_COMPONENT_CONTRASTS.csv, with paired grouped intervals. One seed and validation selection prevent a robust general causal conclusion.

Source-only robustness: B3 has validation macro-F1 0.6845 versus baseline0.6832, with SCC recall0.4105versus0.3474 and almost unchanged melanoma recall. Its selected checkpoint is additional epoch8 after14epochs, so extra optimization is an alternative explanation. NV→MEL and SCC→BCC errors are158/42versus baseline139/35. PAD lesion macro-F1 is0.4105 versus baseline0.3833; SCC recall23/140=0.1643 versus5/140=0.0357. The higher point SCC recall still misses117/140SCC lesions. B3 jointly changes objective/augmentation and is not an isolated augmentation experiment.

The source augmentation policy used no PAD statistics or target fitting, but study direction followed previously exposed PAD results. All new held-out comparisons are POST-HOC EXPLORATORY. The generic enhancement has conditional support for KD-related tradeoffs, not evidence that the full reliability mechanism is necessary, clinically safe or novel. No original model role or selected checkpoint is replaced. A genuinely independent smartphone cohort, appropriate matched controls and multiple seeds remain necessary for a stronger paper; no additional runs are authorized by this report.
'''
# Improve human-readable spacing in the deliberately compact numerical comparisons.
for a,b in [('versus0','versus 0'),('epoch1','epoch 1'),('realized7epochs','realized 7 epochs'),('errors221versus240','errors 221 versus 240'),('from0.6763to0.6905','from 0.6763 to 0.6905'),('errors321to240','errors 321 to 240'),('recall0.7782to0.7355','recall 0.7782 to 0.7355'),('recall0.4211to0.3895','recall 0.4211 to 0.3895'),('baseline0.6832','baseline 0.6832'),('recall0.4105versus0.3474','recall 0.4105 versus 0.3474'),('epoch8 after14epochs','epoch 8 after 14 epochs'),('are158/42versus baseline139/35','are 158/42 versus baseline 139/35'),('is0.4105','is 0.4105'),('baseline0.3833','baseline 0.3833'),('recall23/140','recall 23/140'),('versus5/140','versus 5/140'),('misses117/140SCC','misses 117/140 SCC')]:text=text.replace(a,b)
with (out/'ABLATION_STUDY_REPORT.md').open('a') as f:f.write(text)
with (out/'CROSS_DATASET_ANALYSIS.md').open('a') as f:f.write(text)
(out/'12_Reports/COMPONENT_INTERPRETATION.md').write_text(text)
shutil.copyfile(Path(__file__),out/'code/interpret.py')
lines=[]
for p in sorted(out.rglob('*')):
 if p.is_file() and p.name!='PACKAGE_CHECKSUMS.sha256':lines.append(f'{sha(p)}  {p.relative_to(out)}')
(out/'PACKAGE_CHECKSUMS.sha256').write_text('\n'.join(lines)+'\n')
print('Component contrasts and interpretation saved; no model changes')
