"""Implementation contracts only, no optimizer steps or model performance inference."""
from common import *
from model import Student
from pipeline import *
from train import optimizer_for
from losses import Projector
import torch

def main(out):
 out=Path(out);torch.set_num_threads(4);plain=Student(False,True);fused=Student(True,True)
 assert all(torch.equal(v,fused.backbone.state_dict()[k]) for k,v in plain.backbone.state_dict().items())
 contracts=[]
 for i in range(5):
  cfg=load(out/f'configs/C{i}.json');net=Student(cfg['fusion'],True);proj=Projector()
  for frozen in [True,False]:
   opt=optimizer_for(net,proj,cfg,frozen);roles={g['role']:g['lr'] for g in opt.param_groups};ps=[id(p) for g in opt.param_groups for p in g['params']];assert len(ps)==len(set(ps));assert all(p.requires_grad for g in opt.param_groups for p in g['params'])
   if frozen:assert all(not p.requires_grad for n,p in net.named_parameters() if n.startswith('backbone.') and not n.startswith('backbone.classifier.'))
   else:assert all(p.requires_grad for p in net.parameters())
   if cfg['contrastive']:assert 'training_only_projector' in roles
   else:assert 'training_only_projector' not in roles
   contracts.append({'configuration':cfg['id'],'stage':'head' if frozen else 'fine','learning_rates':roles,'optimizer_steps':0})
 # EXIF orientation tests use synthetic colored corner fixtures, not fabricated lesion examples.
 fixture=np.zeros((24,40,3),dtype=np.uint8);fixture[:12,:20]=[255,0,0];fixture[12:,20:]=[0,255,0];tmp=out/'source_data/EXIF_unit_fixtures';tmp.mkdir(exist_ok=True)
 for orientation in range(1,9):
  p=tmp/f'orientation{orientation}.png';im=Image.fromarray(fixture);exif=im.getexif();exif[274]=orientation;im.save(p,exif=exif)
  result,mask,size=preprocess(p)
  assert size==((24,40) if orientation in [5,6,7,8] else (40,24));assert result.size==(224,224) and result.mode=='RGB'
  assert np.all(np.asarray(result)[~mask]==114)
 # No-op RGB roundtrip, deterministic optional treatments; exclusivity rejects unregistered combinations.
 p=tmp/'orientation1.png';a=preprocess(p,'RGB')[0];b=preprocess(p,'RGB')[0];assert np.array_equal(a,b)
 try:preprocess(p,'CC+CLAHE')
 except AssertionError:pass
 else:raise AssertionError('Combined treatments must not be silently accepted')
 dump(out/'reports/TRAINING_CONTRACT_CHECKS.json',{'status':'PASS','shared_pretrained_backbone_and_classifier_identical':True,'optimizer_stages':contracts,'optimizer_steps':0,'EXIF_cases':8,'letterbox_padding_fixed_114':True,'treatments_exclusive':True,'training_authorization':load(out/'TRAINING_AUTHORIZATION.json')['authorized']})
 print('TRAINING CONTRACTS PASS',flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
