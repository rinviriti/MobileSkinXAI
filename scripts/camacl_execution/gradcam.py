"""Actual validation Grad-CAM only, deterministic cases; no optimizer or weight changes."""
from common import *
from model import Student
from pipeline import SourceDataset,preprocess
from analyze import KEYS,NAMES
from figures import Writer,plt,wrap
import torch,torch.nn.functional as F,gc
from PIL import Image

def main(out):
 out=Path(out);assert load(out/'ALL_FIVE_COMPLETE.json')['status']=='PASS';assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA required for saved BF16 context parity';torch.set_num_threads(4);seed_all(42);w=Writer(out);predictions=read(out/'02_Validation_Results/selected_validation_probabilities.csv');records=[];missing=[];maps=out/'09_Explainability/maps';maps.mkdir(exist_ok=True);assets=out/'09_Explainability/validation_assets';assets.mkdir(exist_ok=True)
 for key in KEYS:
  cfg=load(out/f'configs/{key}.json');status=load(out/f'runs/{key}/status.json');cp=out/f'runs/{key}/checkpoints/best.pt';assert sha(cp)==status['best_checkpoint_sha256'];net=Student(cfg['fusion'],False,cfg['seed']);state=torch.load(cp,map_location='cpu',weights_only=True);net.load_state_dict(state['student'],strict=True);del state;net=net.cuda().eval().requires_grad_(False);ds=SourceDataset(out,cfg,'validation');ids=[r['image'] for r in ds.rows];pos={name:i for i,name in enumerate(ids)};poolmodel=[r for r in predictions if r['model']==key];assert [r['image'] for r in poolmodel]==ids
  stages=['backbone.blocks.1','backbone.blocks.2','backbone.blocks.4'] if cfg['fusion'] else ['backbone.blocks.4'];cols=1+len(stages);fig,axs=plt.subplots(8,cols,figsize=(4*cols,23),layout='constrained');local=[]
  for ci,cl in enumerate(CLASSES):
   for correct in [True,False]:
    rowindex=ci*2+(0 if correct else 1);pool=[r for r in poolmodel if r['true_class']==cl and (r['predicted_class']==cl)==correct]
    if not pool:
     missing.append({'model':key,'class':cl,'correct':correct,'reason':'No actual evaluated case in stratum'})
     for ax in axs[rowindex]:ax.axis('off');ax.text(.05,.5,'No evaluated case in stratum',transform=ax.transAxes)
     continue
    median=float(np.median([float(r['confidence']) for r in pool]));r=min(pool,key=lambda z:(abs(float(z['confidence'])-median),z['image']));location=pos[r['image']];start=location//16*16;indices=list(range(start,min(start+16,len(ds))));offset=location-start;x=torch.stack([ds[j][0] for j in indices]).contiguous().cuda().requires_grad_(True);capture={};handles=[]
    for stage in stages:
     def hook(module,inp,value,stage=stage):capture[stage]=value
     handles.append(net.get_submodule(stage).register_forward_hook(hook))
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x)
    prob=logits[offset].float().softmax(0);pred=int(prob.argmax());delta=float(abs(prob.detach().cpu().numpy()-np.array([float(r[c]) for c in CLASSES])).max());assert pred==CLASSES.index(r['predicted_class']) and delta<=1e-3,('Explanation parity failed; no alternative case selected',key,r['image'],delta)
    gradients=torch.autograd.grad(logits[offset,pred].float(),[capture[s] for s in stages]);im,_,_=preprocess(ds.rows[location]['source'],cfg['preprocessing']['treatment']);raw=np.array(im);asset=assets/(r['image']+'.png')
    if not asset.exists():im.save(asset)
    axs[rowindex,0].imshow(raw);axs[rowindex,0].axis('off');axs[rowindex,0].set_title(f'True {cl} · {"correct" if correct else "incorrect"}\nPred {CLASSES[pred]} ({float(prob[pred].detach()):.3f})\n{r["image"]}',fontsize=8)
    for si,(stage,gradient) in enumerate(zip(stages,gradients)):
     g=gradient[offset].float();a=capture[stage][offset].float();assert torch.isfinite(g).all();cam=F.relu((g.mean((1,2),keepdim=True)*a).sum(0)).detach().cpu().numpy();cam-=cam.min();maximum=float(cam.max());zero=maximum<=1e-12;cam=cam/maximum if not zero else np.zeros_like(cam);up=F.interpolate(torch.from_numpy(cam).view(1,1,*cam.shape),size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy();name=f'{key}_{r["image"]}_{stage.replace(".","_")}';np.savetxt(maps/(name+'.csv'),cam,delimiter=',');np.save(maps/(name+'.npy'),up);overlay=np.clip(.6*raw/255+.4*plt.colormaps['magma'](up)[...,:3],0,1);axs[rowindex,si+1].imshow(overlay);axs[rowindex,si+1].axis('off');axs[rowindex,si+1].set_title(stage+'\nGrad-CAM: predicted logit',fontsize=8)
     records.append({'model':key,'model_name':NAMES[key],'image':r['image'],'true_class':cl,'predicted_class':CLASSES[pred],'correct':correct,'confidence':float(prob[pred].detach()),'target_stage':stage,'spatial_shape':str(cam.shape),'gradient_norm':float(g.norm()),'zero_map':zero,'probability_max_abs_difference':delta,'checkpoint_sha256':status['best_checkpoint_sha256'],'source_sha256':ds.rows[location]['source_sha256'],'asset_sha256':sha(asset),'batch_context_ids':';'.join(ids[j] for j in indices),'batch_offset':offset,'raw_map':str((maps/(name+'.csv')).relative_to(out)),'upsampled_map':str((maps/(name+'.npy')).relative_to(out)),'selection':'nearest median confidence per true-class/correctness, imageID tie-break','interpretation':'separate per-layer maps; last stage excludes shallow direct residual branch; earlier stages also receive downstream gradients'})
    local.append(r['image'])
    for h in handles:h.remove()
    del x,logits,prob,gradients,capture;gc.collect()
  write(out/'09_Explainability/GradCAM_case_records.csv',records);fig.suptitle(wrap(NAMES[key])+'\nActual source validation cases · predicted-class Grad-CAM')
  w.save('09_Explainability',key+'_validation_GradCAM',fig,NAMES[key]+' actual validation Grad-CAM','Deterministic median-confidence correct/error cases per true class. Standard gradient-weighted activation maps, independently normalized. For fusion models, separate stage maps are not a summed explanation; deep-stage map omits direct shallow residual contributions. No clinical/segmentation/fidelity claim.',['09_Explainability/GradCAM_case_records.csv','02_Validation_Results/selected_validation_probabilities.csv'],script='gradcam.py')
  del net;gc.collect();torch.cuda.empty_cache();print('XAI',key,len(local),'cases',flush=True)
 write(out/'09_Explainability/UNAVAILABLE_STRATA.csv',missing or [{'reason':'All requested case strata available'}]);dump(out/'12_Reports/GRADCAM_COMPLETE.json',{'status':'PASS','stage_maps':len(records),'case_count':len({(r['model'],r['image']) for r in records}),'unavailable_strata':missing,'max_probability_difference':max(r['probability_max_abs_difference'] for r in records) if records else None,'optimizer_steps':0,'validation_only':True})
if __name__=='__main__':main(ROOT/sys.argv[1])
