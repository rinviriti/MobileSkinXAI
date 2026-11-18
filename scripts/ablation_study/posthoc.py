from common import *
from analyze import assess,Figures,wrap
import torch,gc,matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import confusion_matrix
from collections import defaultdict
ALL=['M3','M5']+KEYS
class Data(Dataset):
 def __init__(self,rows,cfg):self.rows=rows;self.tf=T.Compose([T.ToTensor(),T.Normalize(**cfg['normalization'])])
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(ROOT/r['processed_path']) as im:assert im.mode=='RGB' and im.size==(224,224);return self.tf(im.copy())
def stats(cm):
 tp=np.diagonal(cm,axis1=-2,axis2=-1);support=cm.sum(-1);pred=cm.sum(-2);rec=np.divide(tp,support,out=np.zeros_like(tp,dtype=float),where=support>0);pre=np.divide(tp,pred,out=np.zeros_like(tp,dtype=float),where=pred>0);f=np.divide(2*pre*rec,pre+rec,out=np.zeros_like(pre),where=pre+rec>0)
 return np.stack([tp.sum(-1)/cm.sum((-2,-1)),f.mean(-1),rec.mean(-1),rec[...,1],rec[...,3]],-1)
METRICS=['accuracy','macro_f1','balanced_accuracy','MEL_recall','SCC_recall']
def lesion_rows(rows):
 groups=defaultdict(list)
 for r in rows:
  if r['use_in_lesion_probability_mean']=='True':groups[r['lesion_scoring_unit']].append(r)
 result=[]
 for group,rs in sorted(groups.items()):
  assert len({r['true_class'] for r in rs})==1 and len({r['bootstrap_cluster'] for r in rs})==1;p=np.mean([[float(r[c]) for c in CLASSES] for r in rs],0);result.append({'lesion_scoring_unit':group,'bootstrap_cluster':rs[0]['bootstrap_cluster'],'true_class':rs[0]['true_class'],'true_index':int(rs[0]['true_index']),**{c:float(p[j]) for j,c in enumerate(CLASSES)},'representative_images':';'.join(r['image'] for r in rs)})
 return result

def main(out):
 out=Path(out);sealcheck(out);lockfile=out/'00_Protocol/FINAL_MODEL_SELECTION_LOCK.json';assert sha(lockfile)==load(out/'00_Protocol/FINAL_LOCK_RECEIPT.json')['sha256'];lock=load(lockfile);assert len(lock['models'])==3;assert torch.cuda.is_available(),'CUDA required; no post-hoc predictions started';assert torch.cuda.is_bf16_supported();torch.set_num_threads(4);core.seed_all(42)
 checkfiles(PHASE6,'PACKAGE_CHECKSUMS.sha256')
 folder=out/'07_Cross_Dataset_Exploratory';predfolder=folder/'prediction_records';predfolder.mkdir(exist_ok=True)
 dump(out/'11_Source_Data/CUDA_environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'VRAM_bytes':torch.cuda.get_device_properties(0).total_memory,'BF16':torch.cuda.is_bf16_supported(),'precision':'CUDA BF16 autocast, FP32 softmax'})
 # First held-out model access in this study, after all3 choices are frozen.
 release_lock=load(RELEASE/'RELEASE_LOCK.json');assert sha(RELEASE/'checksums.sha256')==release_lock['checksums_sha256'];checkfiles(RELEASE,'checksums.sha256')
 with (RELEASE/'dataset_manifest.csv').open() as f:manifest=[r for r in csv.DictReader(f) if r['release_eligible']=='True' and r['split'] in ['test','external']]
 units={(r['dataset'],r['image']):r for r in read(RELEASE/'evaluation_scoring_units.csv')}
 for r in manifest:r.update(units[(r['dataset'],r['image'])]);assert sha(ROOT/r['processed_path'])==r['processed_sha256']
 assert CounterLocal(manifest,'test')==2703 and CounterLocal(manifest,'external')==1303
 for k in KEYS:
  cfg=load(out/f'00_Protocol/{k}.json');m=lock['models'][k];assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256'];net=core.model(cfg);payload=torch.load(ROOT/m['checkpoint'],map_location='cpu',weights_only=True);net.load_state_dict(payload['state_dict'],strict=True);del payload;net=net.cuda().eval().requires_grad_(False)
  for split in ['test','external']:
   path=predfolder/f'{k}_{split}_images.csv';assert not path.exists(),'Once-only inference; refuse overwrite';rows=[r for r in manifest if r['split']==split];loader=DataLoader(Data(rows,cfg),batch_size=16,shuffle=False,num_workers=4,pin_memory=True);result=[];pos=0;start=time.perf_counter()
   with torch.inference_mode():
    for x in loader:
     with core.autocast():logits=net(x.cuda(non_blocking=True))
     p=logits.float().softmax(1).cpu().numpy();assert np.isfinite(p).all()
     for prob in p:
      r=rows[pos];result.append({'model':k,'model_name':NAMES[k],'dataset':r['dataset'],'split':split,'image':r['image'],'true_class':r['label'],'true_index':int(r['label_index']),'predicted_class':CLASSES[int(prob.argmax())],**{c:float(prob[j]) for j,c in enumerate(CLASSES)},'lesion_id':r['lesion_id'],'patient_id':r['patient_id'],'lesion_scoring_unit':r['lesion_scoring_unit'],'bootstrap_cluster':r['bootstrap_cluster'],'use_in_lesion_probability_mean':r['use_in_lesion_probability_mean'],'checkpoint_sha256':m['checkpoint_sha256'],'analysis_status':'POST-HOC EXPLORATORY'});pos+=1
   assert pos==len(rows);write(path,result);print('POSTHOC',k,split,pos,round(time.perf_counter()-start,2),flush=True)
  del net;gc.collect();torch.cuda.empty_cache()
 summary=[];boot={};classwise=[]
 for split in ['test','external']:
  for level in ['lesion_primary','image_secondary']:
   arrays=[];clusters=None
   for k in ALL:
    rows=read(predfolder/f'{k}_{split}_images.csv') if k in KEYS else read(PHASE6/f'independent_prediction_records/{k}_{split}_image_probabilities.csv')
    if level=='lesion_primary':rows=lesion_rows(rows);write(predfolder/f'{k}_{split}_lesions.csv',rows)
    y=np.array([int(r['true_index']) for r in rows]);p=np.array([[float(r[c]) for c in CLASSES] for r in rows]);pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(4));values=stats(cm);summary.append({'model':k,'model_name':NAMES[k],'split':split,'level':level,'n':len(rows),'status':'POST-HOC EXPLORATORY' if k in KEYS else 'Frozen original Phase6 reference; exploratory comparison',**dict(zip(METRICS,map(float,values)))})
    from sklearn.metrics import precision_recall_fscore_support
    pr,re,f,s=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0)
    classwise.extend({'model':k,'split':split,'level':level,'class':c,'precision':float(pr[j]),'recall':float(re[j]),'f1':float(f[j]),'support':int(s[j])} for j,c in enumerate(CLASSES))
    groupids=sorted({r['bootstrap_cluster'] for r in rows});assert clusters is None or groupids==clusters;clusters=groupids;gmap={v:i for i,v in enumerate(groupids)};arr=np.zeros((len(clusters),4,4),int)
    for r,a,b in zip(rows,y,pred):arr[gmap[r['bootstrap_cluster']],a,b]+=1
    arrays.append(arr)
   arrays=np.stack(arrays);rng=np.random.default_rng(42 if split=='test' else 43);reps=[]
   for i in range(2000):
    weights=np.bincount(rng.integers(len(clusters),size=len(clusters)),minlength=len(clusters));cm=np.einsum('g,kgij->kij',weights,arrays);assert np.all(cm.sum(-1)>0);reps.append(stats(cm))
   reps=np.stack(reps);boot[(split,level)]=reps
   for ki,k in enumerate(ALL):
    r=next(r for r in summary if r['model']==k and r['split']==split and r['level']==level)
    for mi,metric in enumerate(METRICS):lo,hi=np.percentile(reps[:,ki,mi],[2.5,97.5]);r[metric+'_CI_low']=float(lo);r[metric+'_CI_high']=float(hi)
   np.save(folder/f'{split}_{level}_bootstrap.npy',reps)
 write(folder/'EXPLORATORY_RESULTS.csv',summary);write(folder/'EXPLORATORY_CLASSWISE.csv',classwise)
 gaps=[];deltas=[]
 for level in ['lesion_primary','image_secondary']:
  for ki,k in enumerate(ALL):
   a=next(r for r in summary if r['model']==k and r['split']=='test' and r['level']==level);b=next(r for r in summary if r['model']==k and r['split']=='external' and r['level']==level)
   for mi,metric in enumerate(METRICS):
    lo,hi=np.percentile(boot[('test',level)][:,ki,mi]-boot[('external',level)][:,ki,mi],[2.5,97.5]);gaps.append({'model':k,'model_name':NAMES[k],'level':level,'metric':metric,'ISIC_minus_PAD':a[metric]-b[metric],'CI_low':float(lo),'CI_high':float(hi)})
  for split in ['test','external']:
   for k in KEYS:
    for ref in ['M3','M5']:
     a=next(r for r in summary if r['model']==k and r['split']==split and r['level']==level);b=next(r for r in summary if r['model']==ref and r['split']==split and r['level']==level);lo,hi=np.percentile(boot[(split,level)][:,ALL.index(k),1]-boot[(split,level)][:,ALL.index(ref),1],[2.5,97.5]);deltas.append({'model':k,'reference':ref,'split':split,'level':level,'macro_f1_difference':a['macro_f1']-b['macro_f1'],'CI_low':float(lo),'CI_high':float(hi),'interpretation':'exploratory descriptive only; no significance claim'})
 write(folder/'EXPLORATORY_DOMAIN_GAPS.csv',gaps);write(folder/'EXPLORATORY_PAIRED_F1_DIFFERENCES.csv',deltas)
 figs=Figures(out)
 for metric in METRICS:
  fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
  for j,split in enumerate(['test','external']):
   rs=[next(r for r in summary if r['model']==k and r['split']==split and r['level']=='lesion_primary') for k in ALL];v=[r[metric] for r in rs];bars=ax.barh(np.arange(5)+(j-.5)*.32,v,height=.3,color=[COLORS[k] for k in ALL],hatch='' if j==0 else '///',alpha=1 if j==0 else .65,label='ISIC' if j==0 else 'PAD');ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
  ax.set(yticks=range(5),yticklabels=[wrap(NAMES[k]) for k in ALL],xlim=(0,1.12),xlabel=metric.replace('_',' '),title='POST-HOC EXPLORATORY · lesion-level '+metric.replace('_',' '));ax.invert_yaxis();ax.legend();figs.save('07_Cross_Dataset_Exploratory','exploratory_'+metric,fig,'POST-HOC EXPLORATORY '+metric,'Prior PAD exposure informed research direction. All new checkpoint choices frozen before reuse. No external superiority confirmation; primary lesion means and grouped bootstrap intervals in source CSV.',['07_Cross_Dataset_Exploratory/EXPLORATORY_RESULTS.csv'],'test;external POST-HOC EXPLORATORY')
 fig,ax=plt.subplots(figsize=(11,6),layout='constrained');rs=[r for r in gaps if r['level']=='lesion_primary' and r['metric']=='macro_f1'];v=np.array([r['ISIC_minus_PAD'] for r in rs]);lo=np.array([r['CI_low'] for r in rs]);hi=np.array([r['CI_high'] for r in rs]);ax.barh(range(5),v,color=[COLORS[k] for k in ALL]);ax.errorbar(v,range(5),xerr=[v-lo,hi-v],fmt='none',color='black',capsize=3);ax.set(yticks=range(5),yticklabels=[wrap(NAMES[k]) for k in ALL],xlabel='ISIC minus PAD macro-F1',title='POST-HOC EXPLORATORY · generalization gaps and95%CI');ax.invert_yaxis();figs.save('07_Cross_Dataset_Exploratory','exploratory_macro_f1_gaps',fig,'POST-HOC EXPLORATORY generalization gaps','Independent dataset cluster draws, paired models; percentile95%CI,2000replicates; single-seed uncertainty only conditional on fixed models.',['07_Cross_Dataset_Exploratory/EXPLORATORY_DOMAIN_GAPS.csv'],'test;external POST-HOC EXPLORATORY')
 dump(folder/'POSTHOC_COMPLETE.json',{'status':'PASS','new_models_evaluated_once':KEYS,'new_image_predictions':3*4006,'no_model_or_hyperparameter_changes_after_lock':True,'selection_lock_sha256':sha(lockfile),'previous_exposure_disclosed':True})
def CounterLocal(rows,split):return sum(r['split']==split for r in rows)
if __name__=='__main__':main(ROOT/sys.argv[1])
