"""Single-backbone inference graph. No training-only loss/projector in this module."""
import torch,timm,math
from torch import nn
from safetensors.torch import load_file
from common import ARCH,PRETRAIN,load,sha
class Student(nn.Module):
 def __init__(self,fusion=False,pretrained=False,seed=42):
  super().__init__();torch.manual_seed(seed)
  # All common backbone/classifier initializations consume exactly the same RNG.
  self.backbone=timm.create_model(ARCH,pretrained=False,num_classes=1000)
  if pretrained:
   record=load(PRETRAIN/'provenance.json');assert record['repo_id']=='timm/'+ARCH
   assert sha(PRETRAIN/'model.safetensors')==record['sha256']==record['publisher_sha256']
   self.backbone.load_state_dict(load_file(str(PRETRAIN/'model.safetensors')),strict=True)
  self.backbone.reset_classifier(4);self.fusion=fusion
  if fusion:
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed+101)
    self.proj8=nn.Sequential(nn.Linear(64,32),nn.ReLU6(),nn.Linear(32,1280))
    self.proj16=nn.Sequential(nn.Linear(96,32),nn.ReLU6(),nn.Linear(32,1280))
   self.scale_logits=nn.Parameter(torch.zeros(2));self.rho=nn.Parameter(torch.tensor(math.log(.1/.9)))
 def encode(self,x):
  x=self.backbone.bn1(self.backbone.conv_stem(x));f8=f16=None
  for i,block in enumerate(self.backbone.blocks):
   x=block(x)
   if i==1:f8=x
   elif i==2:f16=x
  d=self.backbone.forward_head(x,pre_logits=True)
  if self.fusion:
   alpha=self.scale_logits.softmax(0);u8=self.proj8(f8.mean((2,3)));u16=self.proj16(f16.mean((2,3)))
   d=d+self.rho.sigmoid()*(alpha[0]*u8+alpha[1]*u16)
  return d
 def forward(self,x):return self.backbone.classifier(self.encode(x))
 def training_outputs(self,x):
  d=self.encode(x);return self.backbone.classifier(d),d
