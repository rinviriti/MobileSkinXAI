"""RGB reference and source-only, stateless logged acquisition transforms. No lesion masks."""
from common import *
import io,cv2,torch
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
from torch.utils.data import Dataset
MEAN=np.array([.485,.456,.406],np.float32);STD=np.array([.229,.224,.225],np.float32)
STRESS={'brightness_low':{'brightness':.85},'brightness_high':{'brightness':1.15},'warm':{'gains':[1.04,1.,.96]},'cool':{'gains':[.96,1.,1.04]},'jpeg60':{'jpeg':60},'blur075':{'blur':.75},'frame085':{'scale':.85,'placement':[.5,.5]},'shading':{'shading':.12}}
def letterbox(im,size=224):
 w,h=im.size;scale=min(size/w,size/h);nw=max(1,int(w*scale+.5));nh=max(1,int(h*scale+.5));res=im.resize((nw,nh),Image.Resampling.LANCZOS);out=Image.new('RGB',(size,size),(114,)*3);left=(size-nw)//2;top=(size-nh)//2;out.paste(res,(left,top));mask=np.zeros((size,size),bool);mask[top:top+nh,left:left+nw]=True
 return out,mask

def preprocess(path,treatment='RGB'):
 with Image.open(path) as im:im=ImageOps.exif_transpose(im).convert('RGB');original_size=im.size;im,mask=letterbox(im)
 arr=np.asarray(im).copy()
 if treatment=='CC':
  x=arr.astype(np.float32)/255;lum=x.mean(2);valid=mask&(lum>.05)&(lum<.95)
  if int(valid.sum())>100:
   means=x[valid].mean(0,dtype=np.float64);gains=np.clip(means.mean()/np.maximum(means,1e-8),.95,1.05);effective=1+.25*(gains-1);adjust=np.floor(np.clip(arr.astype(np.float64)*effective,0,255)+.5).astype('uint8');arr[mask]=adjust[mask]
 elif treatment=='CLAHE':
  # Exclude letterbox from histograms by operating only on the valid rectangular crop.
  ys,xs=np.where(mask);a,b=ys.min(),ys.max()+1;c,d=xs.min(),xs.max()+1;crop=arr[a:b,c:d];ycc=cv2.cvtColor(crop,cv2.COLOR_RGB2YCrCb);Y=ycc[:,:,0];enh=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(Y);ycc[:,:,0]=np.floor(.8*Y.astype(float)+.2*enh.astype(float)+.5).clip(0,255).astype('uint8');arr[a:b,c:d]=cv2.cvtColor(ycc,cv2.COLOR_YCrCb2RGB)
 else:assert treatment=='RGB','CC and CLAHE are exclusive controlled alternatives'
 return Image.fromarray(arr),mask,original_size

def stable_rng(seed,epoch,image,stream):
 value=hashlib.sha256(f'{seed}|{epoch}|{image}|{stream}'.encode()).digest();return np.random.default_rng(int.from_bytes(value[:8],'little'))

def geometry(im,rng):
 hf=bool(rng.random()<.5);vf=bool(rng.random()<.5);angle=float(rng.uniform(-10,10))
 if hf:im=ImageOps.mirror(im)
 if vf:im=ImageOps.flip(im)
 im=im.rotate(angle,Image.Resampling.BILINEAR,expand=True,fillcolor=(114,)*3);im,_=letterbox(im)
 return im,{'horizontal_flip':hf,'vertical_flip':vf,'rotation_degrees':angle,'rotation_expand':True,'cropping':False}

def acquisition(im,rng):
 p={}
 if rng.random()<.8:p['brightness']=float(rng.uniform(.9,1.1));p['contrast']=float(rng.uniform(.9,1.1))
 if rng.random()<.5:
  gains=rng.uniform(.97,1.03,3);gains/=np.exp(np.log(gains).mean());p['gains']=gains.tolist()
 if rng.random()<.4:p['jpeg']=int(rng.integers(75,96))
 if rng.random()<.2:p['blur']=float(rng.uniform(.1,.6))
 if rng.random()<.3:p['scale']=float(rng.uniform(.9,1.0));p['placement']=rng.random(2).tolist()
 return apply_acquisition(im,p),p

def apply_acquisition(im,p):
 if 'brightness' in p:im=ImageEnhance.Brightness(im).enhance(p['brightness'])
 if 'contrast' in p:im=ImageEnhance.Contrast(im).enhance(p['contrast'])
 if 'gains' in p:im=Image.fromarray(np.floor(np.clip(np.asarray(im,dtype=float)*np.array(p['gains']),0,255)+.5).astype('uint8'))
 if 'jpeg' in p:
  f=io.BytesIO();im.save(f,format='JPEG',quality=p['jpeg'],subsampling=0);f.seek(0)
  with Image.open(f) as tmp:im=tmp.convert('RGB').copy()
 if 'blur' in p:im=im.filter(ImageFilter.GaussianBlur(p['blur']))
 if 'scale' in p:
  n=int(224*p['scale']+.5);n=min(224,max(1,n));res=im.resize((n,n),Image.Resampling.LANCZOS);x=int((224-n)*p['placement'][0]+.5);y=int((224-n)*p['placement'][1]+.5);im=Image.new('RGB',(224,224),(114,)*3);im.paste(res,(x,y))
 if 'shading' in p:
  field=1+p['shading']*np.linspace(-1,1,224)[None,:,None];im=Image.fromarray(np.floor(np.clip(np.asarray(im)*field,0,255)+.5).astype('uint8'))
 return im

def tensor(im):
 a=np.asarray(im,dtype=np.float32)/255.;return torch.from_numpy(((a-MEAN)/STD).transpose(2,0,1).copy())
class SourceDataset(Dataset):
 def __init__(self,out,cfg,split,epoch=0,limit=None,stress=None):
  assert split in ['train','validation'],'Held-out loaders are deliberately unavailable'
  self.rows=[r for r in read(Path(out)/'source_data/approved_train_validation.csv') if r['split']==split];self.cfg=cfg;self.split=split;self.epoch=epoch;self.stress=stress
  if limit:self.rows=self.rows[:limit]
  assert all(r['dataset']=='ISIC2019' and r['release_eligible']=='True' for r in self.rows)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];im,mask,size=preprocess(r['source'],self.cfg['preprocessing']['treatment']);meta={'image':r['image'],'epoch':self.epoch,'source_size':size,'treatment':self.cfg['preprocessing']['treatment']}
  if self.split=='train':
   weak,g=geometry(im,stable_rng(self.cfg['seed'],self.epoch,r['image'],'geometry'))
   if self.cfg['acquisition']:strong,a=acquisition(weak.copy(),stable_rng(self.cfg['seed'],self.epoch,r['image'],'acquisition'))
   else:strong=weak.copy();a={'identity':True}
   meta.update(geometry=g,acquisition=a)
   return tensor(weak),tensor(strong),int(r['label_index']),r['image'],r['lesion_id'],json.dumps(meta,sort_keys=True)
  if self.stress:im=apply_acquisition(im,STRESS[self.stress])
  return tensor(im),int(r['label_index']),r['image']
