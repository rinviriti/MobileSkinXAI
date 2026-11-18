from common import *
from inference import Images,netload
from figures import Writer,plt,DS,wrap
import torch,torch.nn.functional as F,gc
from PIL import Image

def main(out):
 out=Path(out);lock=lockcheck(out);assert torch.cuda.is_available(),'CUDA inaccessible; no Grad-CAM generated';torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 w=Writer(out);w.records=read(out/'FIGURE_INDEX.csv');records=[];missing=[];maps=out/'11_Source_Data/gradcam_maps';maps.mkdir(exist_ok=True)
 meta=read(out/'11_Source_Data/locked_evaluation_manifest.csv')
 for k,m in lock['models'].items():
  assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256'];net=netload(m);prefix='stages' if hasattr(net,'stages') else 'blocks';target_name=f'{prefix}.{len(getattr(net,prefix))-1}';target=net.get_submodule(target_name)
  for split in DS:
   preds=read(out/f'independent_prediction_records/{k}_{split}_image_probabilities.csv');rows=[r for r in meta if r['split']==split];assert [r['image'] for r in rows]==[r['image'] for r in preds];ds=Images(rows,m['config']);batch=m['config']['batch_size'];fig,axs=plt.subplots(4,4,figsize=(14,13),layout='constrained');local=[]
   for ci,cl in enumerate(CLASSES):
    for correct in [True,False]:
     col=0 if correct else 2;pool=[r for r in preds if r['true_class']==cl and (r['predicted_class']==cl)==correct]
     if not pool:
      missing.append({'model':k,'split':split,'true_class':cl,'correct':correct,'reason':'No evaluated cases in stratum'})
      for ax in axs[ci,col:col+2]:ax.axis('off');ax.text(.1,.5,'No evaluated case',transform=ax.transAxes)
      continue
     median=float(np.median([max(float(r[c]) for c in CLASSES) for r in pool]));r=min(pool,key=lambda r:(abs(max(float(r[c]) for c in CLASSES)-median),r['image']));i=int(r['evaluation_row']);start=i//batch*batch;indices=list(range(start,min(start+batch,len(rows))));offset=i-start;x=torch.stack([ds[j] for j in indices]).contiguous().cuda().requires_grad_(True);capture={}
     def hook(module,inputs,output):capture['a']=output
     handle=target.register_forward_hook(hook)
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x)
     p=logits[offset].float().softmax(0);pred=int(p.argmax());diff=float(np.abs(p.detach().cpu().numpy()-np.array([float(r[c]) for c in CLASSES])).max());assert pred==int(r['predicted_index']) and diff<=.02,('Grad-CAM parity failed',k,r['image'],diff)
     a=capture['a'];assert a.ndim==4 and min(a.shape[-2:])>1;g=torch.autograd.grad(logits[offset,pred].float(),a)[0][offset].float();feature=a[offset].float();assert torch.isfinite(g).all();cam=torch.relu((g.mean((1,2),keepdim=True)*feature).sum(0)).detach().cpu().numpy();cam-=cam.min();zero=float(cam.max())<=1e-12
     if not zero:cam/=cam.max()
     up=F.interpolate(torch.tensor(cam).reshape(1,1,*cam.shape),size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy();name=f'{k}_{split}_{r["image"]}';np.savetxt(maps/(name+'.csv'),cam,delimiter=',');np.save(maps/(name+'.npy'),up)
     with Image.open(ROOT/r['processed_path']) as im:raw=np.asarray(im.convert('RGB')).copy()
     overlay=np.clip(.6*raw/255+.4*plt.colormaps['magma'](up)[...,:3],0,1)
     axs[ci,col].imshow(raw);axs[ci,col+1].imshow(overlay)
     for ax in axs[ci,col:col+2]:ax.axis('off')
     axs[ci,col].set_title(f'{cl} · {"correct" if correct else "incorrect"}\n{r["image"]}',fontsize=8);axs[ci,col+1].set_title(f'Predicted {CLASSES[pred]} ({float(p[pred]):.3f})\nGrad-CAM: predicted-class logit',fontsize=9)
     rec={'model':k,'model_name':NAMES[k],'split':split,'image':r['image'],'true_class':cl,'predicted_class':CLASSES[pred],'correct':correct,'confidence':float(p[pred]),'selection':'nearest median top1 confidence within true-class/correctness stratum','target_layer':target_name,'gradient_norm':float(g.norm()),'zero_map':zero,'max_probability_parity_difference':diff,'processed_path':r['processed_path'],'processed_sha256':r['processed_sha256'],'checkpoint_sha256':m['checkpoint_sha256'],'batch_image_ids':';'.join(rows[j]['image'] for j in indices),'batch_offset':offset,'spatial_map':str((maps/(name+'.csv')).relative_to(out)),'upsampled_map':str((maps/(name+'.npy')).relative_to(out))};records.append(rec);local.append(rec);handle.remove();del x,logits,a,g,feature,p;gc.collect()
   write(out/'11_Source_Data/GradCAM_selected_cases.csv',records)
   title=f'{NAMES[k]}\n{DS[split]} · selected evaluated cases';fig.suptitle(title)
   w.save('06_Explainability',f'{k}_{split}_GradCAM',fig,title,'Standard gradient-weighted class activation mapping of predicted-class logit at final spatial stage; genuine correct/error cases selected by frozen median-confidence rule. Maps are qualitative, per-image normalized and not lesion masks or clinical explanations. CUDA BF16 batch context verified against frozen predictions.',['11_Source_Data/GradCAM_selected_cases.csv',f'independent_prediction_records/{k}_{split}_image_probabilities.csv'],split,'image_secondary')
   print('GRADCAM',k,split,len(local),flush=True)
  del net;gc.collect();torch.cuda.empty_cache()
 write(out/'11_Source_Data/GradCAM_unavailable_strata.csv',missing or [{'reason':'All requested strata available'}]);dump(out/'GRADCAM_COMPLETE.json',{'status':'PASS','cases':len(records),'missing_strata':missing,'max_probability_difference':max(r['max_probability_parity_difference'] for r in records),'no_optimizer_or_training':True})
if __name__=='__main__':main(ROOT/sys.argv[1])
