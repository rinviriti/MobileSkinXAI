from common import *
from evaluate_amended import NAMES,state

def main(out):
 out=Path(out);rows=[]
 for k in ['M5','M5-LR']:
  s=state(out,k);h=[json.loads(l) for l in (out/f'03_E2_Training/{k}/history.jsonl').read_text().splitlines()][s['best_epoch']-1];l=h['loss_components'];ce=l['ce'];kd=.5*l['kd'];js=.1*l['consistency'];total=ce+kd+js;assert abs(total-h['training_total_loss'])<1e-5
  rows.append({'model':k,'model_name':NAMES[k],'best_epoch':s['best_epoch'],'CE':ce,'weighted_KD':kd,'weighted_JS':js,'component_sum':total,'weighted_KD_scalar_share':kd/total,'not_a_gradient_contribution_measure':True})
 write(out/'12_Source_Data/SELECTED_EPOCH_LOSS_SCALES.csv',rows)
 pairs=read(out/'12_Source_Data/ENHANCEMENT_PAIRED_COMPARISONS.csv');control=[]
 for r in pairs:
  if r['comparison']=='M5-LR minus M5':control.append({**r,'interpretation':'LR-only intended setting difference; same initialization/teacher/loss/views/sampling/seed; realized early-stopping budgets differ'})
 write(out/'05_Four_Setting_Comparisons/TABLE_H_CONTROL_CLARIFICATION.csv',control)
 lines=[]
 for r in rows:lines.append(f"- {r['model_name']}, selected additional epoch{r['best_epoch']}: CE={r['CE']:.6f}, weightedKD={r['weighted_KD']:.6f}, weightedJS={r['weighted_JS']:.6f}; KD is{100*r['weighted_KD_scalar_share']:.2f}%of the recorded scalar training objective. This does not measure gradient shares.")
 text='''# Teacher-analysis addendum and controlled-comparison clarification\n\nThe original superiority criterion remains FAILED and was explicitly waived only for post-hoc exploratory use of the frozen weaker teacher. No numerical results are changed by this addendum.\n\n'''+ '\n'.join(lines)+'''\n\nGate acceptance of approximately81–96%by class is high despite the teacher's0.629633validationmacro-F1. Gates test confidence, entropy, cross-view agreement and correctness on already observed TRAIN examples, so high acceptance is not validated out-of-sample reliability. These pooled counts combine ISIC/PAD training presentations; domain-specific gates were not separately logged and are not fabricated. The original T=4, coefficients and full-batch weighting denominator are unchanged. The large KD scalar is a measured property of the fixed objective, not by itself an implementation bug or proof of gradient dominance.\n\nThe current PAD test point estimates are joint M3=0.710042, amended M5=0.672227 and LRcontrol=0.696475. Thus this comparison does not demonstrate a target-domain advantage from the complete enhanced recipes over ordinary joint MobileNetV4. Conversely it cannot prove KD caused the deficits: no matched E2 no-teacher ablation exists, and weighting, training views and budget differ from M3. LR control has a higher point estimate than M5, but the paired patient interval spans zero. No significance, clinical or novelty claim follows.\n\n## Annotation clarification\n\nThe raw predeclared ENHANCEMENT_PAIRED_COMPARISONS.csv used one generic confounding annotation for both enhanced-versus-standard and LR-versus-enhanced rows. For the latter, initialization and losses are in fact matched. The only intended configuration differences are the prescribed learning rates (plus identity/display labels); realized stopping epochs differ. TABLE_H_CONTROL_CLARIFICATION.csv retains exactly the same numerical differences and intervals with this precise annotation. Original CSV and TABLE_H remain preserved. The final report uses the corrected description; no probabilities, scores, comparisons or checkpoints are changed.\n'''
 # Use measured values rather than the rounded explanatory constants above.
 metrics=read(out/'12_Source_Data/ALL_SETTING_METRICS.csv');pts={r['model']:float(r['macro_f1']) for r in metrics if r['condition']=='E2' and r['dataset']=='PAD' and r['analysis_level']=='lesion_primary'}
 text=text.replace('joint M3=0.710042, amended M5=0.672227 and LRcontrol=0.696475',f"joint M3={pts['M3']:.6f}, amended M5={pts['M5']:.6f} and LRcontrol={pts['M5-LR']:.6f}")
 with (out/'ENHANCED_MODEL_TEACHER_ANALYSIS_ADDENDUM.md').open('x') as f:f.write(text)
 print('Teacher scalar-loss evidence and control annotation clarified; all original numerical results preserved')
if __name__=='__main__':main(ROOT/sys.argv[1])
