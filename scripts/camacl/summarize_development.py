"""After actual training only: source validation/stress summaries. Cannot read held-out files."""
from common import *
from train import metric
from sklearn.metrics import roc_auc_score,average_precision_score

def main(out):
 out=Path(out)
 for i in range(5):
  path=out/f'runs/C{i}/status.json'
  if not path.exists() or load(path)['status']!='complete':raise RuntimeError('All five actual runs must complete before any results summary/figures; none generated')
 summary=[];eligible={r['image']:r for r in read(out/'source_data/approved_train_validation.csv') if r['split']=='validation'}
 for i in range(5):
  status=load(out/f'runs/C{i}/status.json');rows=read(out/f'runs/C{i}/validation_epoch_{status["best_epoch"]:03d}.csv');assert len(rows)==len(eligible) and {r['image'] for r in rows}==set(eligible)
  y=np.array([int(r['label_index']) for r in rows]);p=np.array([[float(r[c]) for c in CLASSES] for r in rows]);assert all(CLASSES[int(r['label_index'])]==eligible[r['image']]['label'] for r in rows)
  m=metric(y,p);cm=np.array(m['confusion_matrix']);top=p.max(1);correct=p.argmax(1)==y;ece=0
  for j in range(10):
   use=(top>=j/10)&(top<(j+1)/10 if j<9 else top<=1)
   if use.any():ece+=float(use.mean()*abs(correct[use].mean()-top[use].mean()))
  s={'model':f'C{i}','accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'balanced_accuracy':m['balanced_accuracy'],'NLL':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),'Brier_sum':float(((p-np.eye(4)[y])**2).sum(1).mean()),'ECE10':ece,'macro_AUROC':float(np.mean([roc_auc_score(y==j,p[:,j]) for j in range(4)])),'macro_AP':float(np.mean([average_precision_score(y==j,p[:,j]) for j in range(4)]))}
  for j,c in enumerate(CLASSES):
   tp=cm[j,j];fp=cm[:,j].sum()-tp;tn=cm.sum()-cm[j].sum()-fp;s[c+'_recall']=m['classwise'][c]['recall'];s[c+'_specificity']=float(tn/(tn+fp))
  stress=load(out/f'runs/C{i}/source_stress_results.json');s['mean_stress_macro_f1']=float(np.mean([v['macro_f1'] for v in stress.values()]));s['stress_degradation']=s['macro_f1']-s['mean_stress_macro_f1'];summary.append(s)
 base=summary[0];enh=summary[4];criteria={'interpretation':'exploratory source-development criteria, not clinical or external transfer proof','clean_macro_f1_noninferiority_margin_point_estimate':.01,'MEL_SCC_clean_recall_margin_point_estimate':.03,'mean_stress_macro_f1_improves':enh['mean_stress_macro_f1']>base['mean_stress_macro_f1'],'clean_F1_condition':enh['macro_f1']>=base['macro_f1']-.01,'MEL_condition':enh['MEL_recall']>=base['MEL_recall']-.03,'SCC_condition':enh['SCC_recall']>=base['SCC_recall']-.03,'checkpoint_selection_uses_stress':False}
 write(out/'reports/ACTUAL_DEVELOPMENT_COMPARISON.csv',summary);dump(out/'reports/ACTUAL_SOURCE_CRITERIA.json',criteria)
if __name__=='__main__':main(ROOT/sys.argv[1])
