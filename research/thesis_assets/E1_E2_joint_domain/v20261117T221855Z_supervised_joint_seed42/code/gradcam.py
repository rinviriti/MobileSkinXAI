from common import *
from engine import Images
from PIL import Image
import torch,timm,torch.nn.functional as F,gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(out):
 out=Path(out);status=load(out/'03_E2_Training/TRAINING_COMPLETE.json');assert sha(ROOT/status['best_checkpoint'])==status['best_checkpoint_sha256'];cfg=load(out/'00_Protocol/E2_CONFIG.json');assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();torch.set_num_threads(4);rows=[r for r in read(out/'11_Source_Data/DEVELOPMENT_MANIFEST.csv') if r['domain']=='PAD' and r['study_split']=='validation'];saved=read(out/f'03_E2_Training/PAD_validation_epoch_{status["best_epoch"]:03d}.csv');assert [r['image'] for r in rows]==[r['image'] for r in saved];ds=Images(rows,cfg);net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=4);state=torch.load(ROOT/status['best_checkpoint'],map_location='cpu',weights_only=True);net.load_state_dict(state['state_dict'],strict=True);del state;net=net.cuda().eval().requires_grad_(False);target=net.get_submodule('blocks.4');fig,axs=plt.subplots(8,2,figsize=(9,23),layout='constrained');records=[];missing=[];assets=out/'09_Explainability/assets';assets.mkdir(exist_ok=True)
 for j,c in enumerate(CLASSES):
  for correct in [True,False]:
   ri=j*2+(0 if correct else 1);pool=[(i,r) for i,r in enumerate(saved) if r['true_class']==c and (r['predicted_class']==c)==correct]
   if not pool:
    missing.append({'class':c,'correct':correct,'reason':'No actual validation case in requested stratum'})
    for ax in axs[ri]:ax.axis('off');ax.text(.05,.5,'No case in stratum',transform=ax.transAxes)
    continue
   median=float(np.median([max(float(r[k]) for k in CLASSES) for _,r in pool]));i,r=min(pool,key=lambda z:(abs(max(float(z[1][k]) for k in CLASSES)-median),z[1]['image']));start=i//16*16;indices=list(range(start,min(start+16,len(ds))));offset=i-start;x=torch.stack([ds[n][0] for n in indices]).cuda().requires_grad_(True);capture={};handle=target.register_forward_hook(lambda m,inp,v:capture.update(value=v))
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x)
   p=logits[offset].float().softmax(0);pred=int(p.argmax());delta=float(np.max(np.abs(p.detach().cpu().numpy()-np.array([float(r[k]) for k in CLASSES]))));assert delta<=1e-3 and pred==int(r['predicted_index']);g=torch.autograd.grad(logits[offset,pred].float(),capture['value'])[0][offset].float();a=capture['value'][offset].float();assert torch.isfinite(g).all() and g.norm()>0;cam=F.relu((g.mean((1,2),keepdim=True)*a).sum(0)).detach().cpu().numpy();cam-=cam.min();mx=float(cam.max());blank=mx<=1e-12;cam=cam/mx if not blank else np.zeros_like(cam);up=F.interpolate(torch.from_numpy(cam).view(1,1,*cam.shape),size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy();image=Image.open(ROOT/rows[i]['processed_path']).convert('RGB');raw=np.array(image);stem='E2_'+r['image'];image.save(assets/(stem+'_input.png'));np.save(assets/(stem+'_map.npy'),up);np.savetxt(assets/(stem+'_spatial.csv'),cam,delimiter=',');rgba=plt.colormaps['magma'](up);rgba[...,3]=.55*up;Image.fromarray((rgba*255).astype(np.uint8),'RGBA').save(assets/(stem+'_overlay.png'));overlay=Image.alpha_composite(image.convert('RGBA'),Image.fromarray((rgba*255).astype(np.uint8),'RGBA'));overlay.save(assets/(stem+'_composite.png'))
   axs[ri,0].imshow(raw);axs[ri,1].imshow(overlay)
   for ax in axs[ri]:ax.axis('off')
   axs[ri,0].set_title(f'{r["image"]}\nTrue {c} · predicted {CLASSES[pred]} ({float(p[pred].detach()):.3f})',fontsize=9);axs[ri,1].set_title('Predicted-logit Grad-CAM · blocks.4'+('\nBlank: no positive localization evidence' if blank else ''),fontsize=9)
   records.append({'model':'Joint ISIC+PAD MobileNetV4-Conv-Small','image':r['image'],'true_class':c,'predicted_class':CLASSES[pred],'probability':float(p[pred].detach()),'correct':correct,'feature_layer':'blocks.4','gradient_norm':float(g.norm()),'blank_map':blank,'spatial_std':float(cam.std()),'probability_parity_max_abs_difference':delta,'split':'PAD development validation','checkpoint_sha256':status['best_checkpoint_sha256'],'source_prepared_sha256':rows[i]['processed_sha256'],'map':'09_Explainability/assets/'+stem+'_map.npy','selection':'nearest median top1 confidence within trueclass/correctness;ID tie break'});handle.remove();del x,logits,capture,p;gc.collect()
 fig.suptitle('Joint ISIC+PAD MobileNetV4-Conv-Small\nGenuine PAD development-validation Grad-CAM; no Flutter claim')
 for ext in ['png','svg','pdf']:fig.savefig(out/'09_Explainability'/('E2_validation_GradCAM.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig);write(out/'09_Explainability/GRADCAM_CASES.csv',records);write(out/'09_Explainability/UNAVAILABLE_STRATA.csv',missing or [{'reason':'All requested strata available'}]);dump(out/'09_Explainability/GRADCAM_COMPLETE.json',{'status':'PASS','cases':len(records),'missing':missing,'blank_maps':sum(r['blank_map'] for r in records),'model_optimizer_steps':0,'development_only':True});print('DEVELOPMENT GRADCAM COMPLETE',len(records),flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
