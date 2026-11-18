import torch
from torch import nn
from torch.nn import functional as F
class Projector(nn.Module):
 def __init__(self,seed=42):
  super().__init__()
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed+202);self.layers=nn.Sequential(nn.Linear(1280,128),nn.ReLU6(),nn.Linear(128,64))
 def forward(self,x):return F.normalize(self.layers(x.float()),dim=-1,eps=1e-8)
def js(p,q):
 p=p.float().clamp_min(1e-8);q=q.float().clamp_min(1e-8);m=.5*(p+q);return .5*((p*(p.log()-m.log())).sum(-1)+(q*(q.log()-m.log())).sum(-1))
def supcon(z,labels,temperature=.1):
 z=F.normalize(z.float(),dim=-1,eps=1e-8);labels=labels.long();classes=labels.unique()
 if len(classes)<2:return z.sum()*0,{'skipped_single_class':True,'minimum_positives':0}
 logits=z@z.T/temperature;diag=torch.eye(len(z),dtype=torch.bool,device=z.device);positive=(labels[:,None]==labels[None,:])&~diag
 counts=positive.sum(1);assert counts.min()>=1,'Paired view must always remain a valid positive'
 logden=torch.logsumexp(logits.masked_fill(diag,float('-inf')),dim=1);logprob=logits-logden[:,None];per=-(logprob*positive.float()).sum(1)/counts
 result=torch.stack([per[labels==c].mean() for c in classes]).mean()
 return result,{'skipped_single_class':False,'minimum_positives':int(counts.min()),'present_classes':int(len(classes))}
def objective(sw,ss,fw,fs,y,weights,projector,cfg,fine_epoch):
 sw,ss=sw.float(),ss.float();ce=.5*(F.cross_entropy(sw,y,weight=weights)+F.cross_entropy(ss,y,weight=weights));cons=js(sw.softmax(-1),ss.softmax(-1)).mean();contrast=sw.sum()*0;stats={};ramp=min(max(fine_epoch,0)/5,1.)
 if cfg['contrastive'] and fine_epoch>0:
  contrast,stats=supcon(torch.cat([projector(fw),projector(fs)]),y.repeat(2),cfg['loss']['temperature'])
 total=ce+ramp*(cfg['loss']['lambda_js']*cons+cfg['loss']['lambda_supcon']*contrast)
 return total,{'ce':ce.detach(),'js':cons.detach(),'supcon':contrast.detach(),'ramp':ramp,**stats}
