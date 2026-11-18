"""Frozen-checkpoint development-only Grad-CAM; no optimization or held-out images."""
from common import *
from evaluate import KEYS,NAMES,BASE,cfg_for,state
from base_engine import Images
from PIL import Image
import torch,timm,torch.nn.functional as F,gc,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(out):
 out=Path(out);assert load(out/'13_Reports/FIGURES_COMPLETE.json')['status']=='PASS';assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();torch.set_num_threads(4);rows=[r for r in read(out/'01_Frozen_Data_Manifests/DEVELOPMENT_MANIFEST.csv') if r['domain']=='PAD' and r['study_split']=='validation'];features=load(out/'00_Protocol/SPATIAL_FEATURES.json');index=read(out/'FIGURE_INDEX.csv');known={r['figure_id'] for r in index};allrecords=[];missing=[]
 def register(key,source,caption,paths):
  fid=key+'_validation_GradCAM'
  if fid not in known:index.append({'figure_id':fid,'title':'E2 '+NAMES[key]+' development-validation Grad-CAM','caption':caption,'source_data':source,'source_sha256':sha(out/source),'script':'code/gradcam.py','PNG_DPI':300,'thesis_shortlist':False,**paths});known.add(fid);write(out/'FIGURE_INDEX.csv',index)
 for key in KEYS:
  folder=out/'09_Explainability'/key;folder.mkdir(exist_ok=True);receipt=folder/'COMPLETE.json'
  if receipt.exists():
   r=load(receipt)
   for p,h in r['sha256'].items():assert sha(out/p)==h
   allrecords.extend(read(folder/'CASES.csv'));continue
  status=state(out,key);cfg=cfg_for(out,key);assert sha(ROOT/status['best_checkpoint'])==status['best_checkpoint_sha256']
  paths={e:f'09_Explainability/{key}/{key}_validation_GradCAM.{e}' for e in ['png','svg','pdf']}
  if key=='M3':
   source=BASE/'09_Explainability';cases=read(source/'GRADCAM_CASES.csv');assert all(r['checkpoint_sha256']==status['best_checkpoint_sha256'] for r in cases)
   for e,p in paths.items():
    target=out/p
    if not target.exists():shutil.copy2(source/f'E2_validation_GradCAM.{e}',target)
    assert sha(target)==sha(source/f'E2_validation_GradCAM.{e}')
   for r in cases:r.update(configuration='M3',source_experiment=str(BASE.relative_to(ROOT)),provenance_note='Reused unchanged original M3 E2 Grad-CAM; asset paths relative to original experiment')
   write(folder/'CASES.csv',cases);allrecords.extend(cases)
  else:
   saved=read(out/f'03_E2_Training/{key}/PAD_validation_epoch_{status["best_epoch"]:03d}.csv');assert [r['image'] for r in rows]==[r['image'] for r in saved];ds=Images(rows,cfg);net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=4);payload=torch.load(ROOT/status['best_checkpoint'],map_location='cpu',weights_only=True);net.load_state_dict(payload['state_dict'],strict=True);del payload;net=net.cuda().eval().requires_grad_(False);layer=features[key]['layer'];target=net.get_submodule(layer);fig,axs=plt.subplots(8,2,figsize=(9,23),layout='constrained');cases=[];assets=folder/'assets';assets.mkdir(exist_ok=True)
   for j,c in enumerate(CLASSES):
    for correct in [True,False]:
     ri=j*2+(0 if correct else 1);pool=[(i,r) for i,r in enumerate(saved) if r['true_class']==c and (r['predicted_class']==c)==correct]
     if not pool:
      missing.append({'model':key,'class':c,'correct':correct,'reason':'No actual validation case in requested stratum'})
      for ax in axs[ri]:ax.axis('off');ax.text(.05,.5,'No case in stratum',transform=ax.transAxes)
      continue
     median=float(np.median([max(float(r[k]) for k in CLASSES) for _,r in pool]));i,r=min(pool,key=lambda z:(abs(max(float(z[1][k]) for k in CLASSES)-median),z[1]['image']));start=i//16*16;indices=range(start,min(start+16,len(ds)));offset=i-start;x=torch.stack([ds[n][0] for n in indices]).cuda().requires_grad_(True);capture={};handle=target.register_forward_hook(lambda m,inp,v:capture.update(value=v))
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x)
     p=logits[offset].float().softmax(0);pred=int(p.argmax());delta=float(np.max(np.abs(p.detach().cpu().numpy()-np.array([float(r[k]) for k in CLASSES]))));assert delta<=1e-3 and pred==int(r['predicted_index']);g=torch.autograd.grad(logits[offset,pred].float(),capture['value'])[0][offset].float();a=capture['value'][offset].float();assert torch.isfinite(g).all() and g.norm()>0;cam=F.relu((g.mean((1,2),keepdim=True)*a).sum(0)).detach().cpu().numpy();cam-=cam.min();mx=float(cam.max());blank=mx<=1e-12;cam=cam/mx if not blank else np.zeros_like(cam);up=F.interpolate(torch.from_numpy(cam).view(1,1,*cam.shape),size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy();assert sha(ROOT/rows[i]['processed_path'])==rows[i]['processed_sha256'];image=Image.open(ROOT/rows[i]['processed_path']).convert('RGB');stem=key+'_'+r['image'];image.save(assets/(stem+'_input.png'));np.save(assets/(stem+'_map.npy'),up);np.savetxt(assets/(stem+'_spatial.csv'),cam,delimiter=',');rgba=plt.colormaps['magma'](up);rgba[...,3]=.55*up;overlayimage=Image.fromarray((rgba*255).astype(np.uint8));overlayimage.save(assets/(stem+'_overlay.png'));overlay=Image.alpha_composite(image.convert('RGBA'),overlayimage);overlay.save(assets/(stem+'_composite.png'));axs[ri,0].imshow(image);axs[ri,1].imshow(overlay)
     for ax in axs[ri]:ax.axis('off')
     axs[ri,0].set_title(f'{r["image"]}\nTrue {c} · predicted {CLASSES[pred]} ({float(p[pred].detach()):.3f})',fontsize=9);axs[ri,1].set_title('Predicted-logit Grad-CAM · '+layer+('\nBlank: no positive localization evidence' if blank else ''),fontsize=9)
     cases.append({'configuration':key,'model':NAMES[key],'image':r['image'],'true_class':c,'predicted_class':CLASSES[pred],'probability':float(p[pred].detach()),'correct':correct,'feature_layer':layer,'gradient_norm':float(g.norm()),'blank_map':blank,'spatial_std':float(cam.std()),'probability_parity_max_abs_difference':delta,'split':'PAD development validation','checkpoint_sha256':status['best_checkpoint_sha256'],'source_prepared_sha256':rows[i]['processed_sha256'],'map':str((assets/(stem+'_map.npy')).relative_to(out)),'selection':'nearest median top1 confidence within trueclass/correctness;ID tie break'});handle.remove();del x,logits,capture,p;gc.collect()
   fig.suptitle('E2 '+NAMES[key]+'\nGenuine PAD development-validation Grad-CAM; no Flutter claim')
   for e,path in paths.items():
    if not (out/path).exists():fig.savefig(out/path,dpi=300,bbox_inches='tight')
   plt.close(fig);write(folder/'CASES.csv',cases);allrecords.extend(cases);del net;gc.collect();torch.cuda.empty_cache()
  source=f'09_Explainability/{key}/CASES.csv';register(key,source,'Frozen model predicted-logit Grad-CAM on development-validation images only. Median-confidence correct/error examples by true class; blank maps explicitly labeled. M3 copied from its verified original package. No on-device Flutter claim.',paths);dump(receipt,{'status':'PASS','sha256':{p:sha(out/p) for p in list(paths.values())+[source]},'checkpoint_sha256':status['best_checkpoint_sha256']});print('GRADCAM',key,len(cases),flush=True)
 # E1 examples are reused verbatim from the original six-model validation package.
 original=ROOT/'research/thesis_assets/final_training_results/v20261117T091430Z_704ed3cc';refs=[]
 for key in KEYS:
  source=original/f'11_Source_Data/{key}_CUDA_BF16_GradCAM_examples.csv';assert source.exists();records=read(source);models=load(out/'00_Protocol/E1_MODELS.json');assert all(r.get('checkpoint_sha256',models[key]['checkpoint_sha256'])==models[key]['checkpoint_sha256'] for r in records)
  for ext in ['png','svg','pdf']:
   src=original/f'10_Explainability/10_{key}_GradCAM_CUDA_BF16.{ext}';dest=out/f'09_Explainability/E1_{key}_reused.{ext}'
   if not dest.exists():shutil.copy2(src,dest)
   assert sha(src)==sha(dest);refs.append({'model':key,'condition':'E1','source_figure':str(src.relative_to(ROOT)),'copied_figure':str(dest.relative_to(out)),'figure_sha256':sha(src),'case_provenance':str(source.relative_to(ROOT)),'case_provenance_sha256':sha(source),'checkpoint_sha256':models[key]['checkpoint_sha256']})
 write(out/'09_Explainability/E1_REUSED_PROVENANCE.csv',refs);write(out/'09_Explainability/GRADCAM_CASES.csv',allrecords);write(out/'09_Explainability/UNAVAILABLE_STRATA.csv',missing or [{'reason':'All requested new-model strata available'}]);dump(out/'09_Explainability/GRADCAM_COMPLETE.json',{'status':'PASS','E2_models':6,'new_E2_models':5,'reused_M3':True,'E1_figures_reused':6,'cases':len(allrecords),'blank_maps':sum(str(r['blank_map']).lower()=='true' for r in allrecords),'missing':missing,'optimizer_steps':0,'development_only':True})
if __name__=='__main__':main(ROOT/sys.argv[1])
