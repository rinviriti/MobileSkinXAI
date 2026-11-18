from common import *
import torch,timm,gc,platform
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset,DataLoader
class Images(Dataset):
 def __init__(self,rows,cfg):self.rows=rows;self.transform=T.Compose([T.ToTensor(),T.Normalize(**cfg['normalization'])])
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(ROOT/r['processed_path']) as im:assert im.mode=='RGB' and im.size==(224,224);x=self.transform(im.copy())
  return x

def netload(m):
 net=timm.create_model(m['config']['architecture'],pretrained=False,num_classes=4)
 payload=torch.load(ROOT/m['checkpoint'],map_location='cpu',weights_only=True)
 net.load_state_dict(payload['state_dict'],strict=True);return net.cuda().eval().requires_grad_(False)

def main(out):
 out=Path(out);lock=lockcheck(out);assert torch.cuda.is_available(),'CUDA inaccessible; no predictions started'
 assert torch.cuda.is_bf16_supported();torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 manifest=read(out/'11_Source_Data/locked_evaluation_manifest.csv');records=[]
 env={'python':platform.python_version(),'torch':torch.__version__,'timm':timm.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'VRAM_bytes':torch.cuda.get_device_properties(0).total_memory,'free_total_VRAM':torch.cuda.mem_get_info(),'precision':'CUDA BF16 autocast; FP32 softmax','no_optimizer':True}
 dump(out/'11_Source_Data/inference_environment.json',env)
 for k,m in lock['models'].items():
  assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256'];net=netload(m)
  for split in ['test','external']:
   path=out/f'independent_prediction_records/{k}_{split}_image_probabilities.csv';assert not path.exists(),'Do not overwrite independent predictions'
   rows=[r for r in manifest if r['split']==split];ds=Images(rows,m['config']);loader=DataLoader(ds,batch_size=m['config']['batch_size'],shuffle=False,num_workers=4,pin_memory=True)
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();preds=[];offset=0
   with torch.inference_mode():
    for x in loader:
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x.cuda(non_blocking=True))
     p=logits.float().softmax(1).cpu().numpy();assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
     for j,prob in enumerate(p):
      r=rows[offset+j];pred=int(prob.argmax());preds.append({'model':k,'model_name':m['name'],'dataset':r['dataset'],'split':split,'image':r['image'],'true_class':r['label'],'true_index':int(r['label_index']),'predicted_class':CLASSES[pred],'predicted_index':pred,**{c:float(prob[i]) for i,c in enumerate(CLASSES)},'lesion_id':r['lesion_id'],'patient_id':r['patient_id'],'lesion_scoring_unit':r['lesion_scoring_unit'],'bootstrap_cluster':r['bootstrap_cluster'],'use_in_lesion_probability_mean':r['use_in_lesion_probability_mean'],'processed_path':r['processed_path'],'processed_sha256':r['processed_sha256'],'checkpoint_sha256':m['checkpoint_sha256'],'evaluation_row':offset+j,'probability_type':'uncalibrated raw softmax'})
     offset+=len(p)
   torch.cuda.synchronize();elapsed=time.perf_counter()-start;assert len(preds)==len(rows);write(path,preds)
   records.append({'model':k,'split':split,'images':len(rows),'seconds_including_loading':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'prediction_sha256':sha(path),'checkpoint_sha256':m['checkpoint_sha256'],'batch_size':m['config']['batch_size']})
   dump(out/'11_Source_Data/inference_runs.json',records);print(k,split,len(rows),round(elapsed,2),'seconds',flush=True)
  del net;gc.collect();torch.cuda.empty_cache();lockcheck(out)
 dump(out/'INFERENCE_COMPLETE.json',{'status':'PASS','datasets':2,'checkpoints':6,'image_predictions':sum(r['images'] for r in records),'no_training':True,'selection_lock_sha256':sha(out/'MODEL_SELECTION_LOCK.json')})
if __name__=='__main__':main(ROOT/sys.argv[1])
