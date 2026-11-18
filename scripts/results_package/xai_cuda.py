"""Canonical Grad-CAM in the exact CUDA BF16 validation batch context; never trains."""
import argparse,gc
from common import *
from models_xai import load_net
import torch
import torch.nn.functional as F

def generate(c):
 assert torch.cuda.is_available(),'CUDA inaccessible in sandbox; no canonical XAI started'
 assert torch.cuda.is_bf16_supported();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.manual_seed(42)
 data=c['data'];val=c['validation'];probs=read_csv(data/'selected_validation_probabilities.csv');assets=data/'xai_input_assets';rawfolder=data/'gradcam_cuda_maps';rawfolder.mkdir(exist_ok=True)
 # Reuse the exact validation dataset implementation and its ordering/normalization.
 sys.path.insert(0,str(PREP/'code'));from core import Images
 records=[];access=[];missing=[]
 for k in KEYS:
  net=load_net(c,k).cuda().eval().requires_grad_(False);cfg=c['models'][k]['config'];ds=Images(cfg,'validation');batchsize=cfg['batch_size'];ids=[r['image'] for r in ds.rows];pos={x:i for i,x in enumerate(ids)}
  copiedpred=data/f'recorded_runs/{k}/validation_predictions_epoch_{c["models"][k]["best_epoch"]:03d}.csv';ordered=read_csv(copiedpred);assert [r['image'] for r in ordered]==ids
  prefix='stages' if hasattr(net,'stages') else 'blocks';target_path=f'{prefix}.{len(getattr(net,prefix))-1}';target=net.get_submodule(target_path);poolmodel=[r for r in probs if r['model']==k];panels={};local=[]
  for cl in CLASSES:
   for correct in [True,False]:
    pool=[r for r in poolmodel if r['true_class']==cl and ((r['predicted_class']==cl)==correct)]
    if not pool:missing.append({'model':NAMES[k],'class':cl,'correct':correct,'reason':'No saved prediction in group'});continue
    median=float(np.median([float(r[r['predicted_class']]) for r in pool]));r=sorted(pool,key=lambda x:(abs(float(x[x['predicted_class']])-median),x['image']))[0];imageid=r['image'];location=pos[imageid];start=location//batchsize*batchsize;indices=list(range(start,min(start+batchsize,len(ds))));offset=location-start;batchids=[ids[i] for i in indices];items=[]
    for i in indices:
     meta=val[ids[i]];assert meta['release_eligible']=='True' and meta['split']=='validation';assert sha(ROOT/meta['processed_path'])==meta['processed_sha256'];asset=assets/(ids[i]+'.jpg')
     if not asset.exists():shutil.copyfile(ROOT/meta['processed_path'],asset)
     items.append(ds[i][0]);access.append({'model':NAMES[k],'target_image':imageid,'batch_image':ids[i],'split':'validation','input_sha256':sha(asset),'access':'original validation-batch context for selected Grad-CAM target'})
    x=torch.stack(items).cuda().requires_grad_(True);capture={}
    def hook(module,inputs,output):capture['activation']=output
    handle=target.register_forward_hook(hook)
    try:
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x)
     p=logits[offset].float().softmax(-1);pred=int(p.argmax());saved=np.array([float(r[s]) for s in CLASSES]);diff=float(np.abs(p.detach().cpu().numpy()-saved).max());assert pred==CLASSES.index(r['predicted_class']),'Recorded BF16 predicted class mismatch';assert diff<=.02,f'Original-batch CUDA BF16 parity failed: {diff}'
     activation=capture['activation'];assert activation.ndim==4 and min(activation.shape[-2:])>1;grad=torch.autograd.grad(logits[offset,pred].float(),activation)[0][offset].float();feature=activation[offset].float();assert torch.isfinite(grad).all();alpha=grad.mean((1,2),keepdim=True);cam=torch.relu((alpha*feature).sum(0)).detach().cpu().numpy();normal=cam-cam.min();maximum=float(normal.max());zero=maximum<=1e-12;normal=normal/maximum if not zero else np.zeros_like(normal);up=F.interpolate(torch.from_numpy(normal).view(1,1,*normal.shape),size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy()
     with Image.open(assets/(imageid+'.jpg')) as im:raw=np.asarray(im.convert('RGB')).copy()
     overlay=np.clip(.6*(raw/255)+.4*plt.colormaps['magma'](up)[...,:3],0,1);mapfile=rawfolder/f'{k}_{cl}_{"correct" if correct else "incorrect"}_{imageid}.csv';write_csv(mapfile,[{'row':i,**{f'column{j}':float(normal[i,j]) for j in range(normal.shape[1])}} for i in range(normal.shape[0])]);npy=rawfolder/(mapfile.stem+'_224.npy')
     if not npy.exists():np.save(npy,up.astype(np.float32))
     rec={'model':k,'model_name':NAMES[k],'image':imageid,'true_class':cl,'predicted_class':CLASSES[pred],'correct':correct,'selection':'nearest median saved prediction confidence in true-class/correctness group, lexicographic tie','target':'predicted raw logit','target_layer':target_path,'batchsize_original':batchsize,'actual_batchsize':len(indices),'target_batch_offset':offset,'original_batch_image_ids':json.dumps(batchids),'feature_shape':json.dumps(list(activation.shape)),'gradient_L2':float(grad.square().sum().sqrt()),'saved_probability':float(saved[pred]),'CUDA_BF16_probability':float(p[pred].detach()),'max_abs_probability_difference':diff,'parity_tolerance':.02,'zero_positive_map':zero,'map_csv':str(mapfile.relative_to(c['out'])),'map_224_npy':str(npy.relative_to(c['out'])),'image_asset':str((assets/(imageid+'.jpg')).relative_to(c['out'])),'image_sha256':val[imageid]['processed_sha256'],'checkpoint_sha256':c['models'][k]['checkpoint_sha256']};records.append(rec);local.append(rec);panels[(cl,correct)]=(raw,overlay,rec)
    except Exception as exc:missing.append({'model':NAMES[k],'class':cl,'correct':correct,'image':imageid,'reason':repr(exc)})
    finally:handle.remove()
  sp=write_csv(data/f'{k}_CUDA_BF16_GradCAM_examples.csv',local) if local else write_json(data/f'{k}_CUDA_GradCAM_missing.json',missing)
  fig,axes=plt.subplots(4,4,figsize=(13,11),layout='constrained')
  for i,cl in enumerate(CLASSES):
   for j,correct in enumerate([True,False]):
    axes[i,2*j].axis('off');axes[i,2*j+1].axis('off');item=panels.get((cl,correct))
    if item is None:axes[i,2*j+1].text(.5,.5,'Unavailable: see source log',ha='center',va='center');continue
    raw,overlay,r=item;axes[i,2*j].imshow(raw);axes[i,2*j+1].imshow(overlay);axes[i,2*j].set_title(cl+': '+('correct' if correct else 'incorrect')+'\n'+r['image'],fontsize=8);axes[i,2*j+1].set_title('Grad-CAM for '+r['predicted_class']+f"\np={r['CUDA_BF16_probability']:.3f}"+(' (zero map)' if r['zero_positive_map'] else ''),fontsize=8)
  fig.suptitle(NAMES[k]+'\nStandard Grad-CAM: original CUDA BF16 validation context',fontsize=13)
  save_figure(c,'10_Explainability',k+'_GradCAM_CUDA_BF16',fig,NAMES[k]+' — canonical validation Grad-CAM','Actual trained-checkpoint standard Grad-CAM on predicted raw logits: alpha_k=mean spatial gradient, map=ReLU(sum alpha_k feature_k), per-image minmax normalization, bilinear upsampling. Correct and incorrect examples in every true class are selected near median saved confidence, not by map appearance. Original validation batch order/size, unchanged preprocessing and CUDA BF16 mode are reused, with saved-probability parity checked (same top1; max absolute difference<=0.02). Original/overlay panels are paired. Colors show relative within-image activation, not segmentation, clinical localization or calibrated importance. This is Grad-CAM, not Grad-CAM++, Score-CAM or a Flutter feature.',[sp]+[c['out']/r['map_csv'] for r in local]+[c['out']/r['image_asset'] for r in local],models=[k],method='standard Grad-CAM; original CUDA BF16 batch context; validation only; zero optimizer steps',script='xai_cuda.py')
  del net;gc.collect();torch.cuda.empty_cache();print('CANONICAL_GRADCAM '+NAMES[k],flush=True)
 write_csv(data/'canonical_GradCAM_examples.csv',records);write_csv(data/'canonical_XAI_image_access_log.csv',access);write_json(data/'CUDA_BF16_XAI_availability.json',{'requested_examples':48,'generated_examples':len(records),'unavailable':missing,'zero_positive_maps':sum(r['zero_positive_map'] for r in records),'max_saved_probability_difference':max((r['max_abs_probability_difference'] for r in records),default=None),'device':torch.cuda.get_device_name(0),'precision':'original CUDA BF16 mode; gradient aggregation in float32','original_validation_batches_reused':True,'parameter_updates':0,'test_PAD_images_opened':False})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();generate(load_context(a.output))
if __name__=='__main__':main()
