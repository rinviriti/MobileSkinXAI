from common import *
from model import Student
from pipeline import *
from losses import Projector,supcon,js
import time,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from collections import Counter

def savefig(out,name,fig):
 for ext in ['png','svg','pdf']:fig.savefig(out/f'figures/{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def main(out):
 out=Path(out);torch.set_num_threads(4);cv2.setNumThreads(1);rows=read(out/'source_data/approved_train_validation.csv');train=[r for r in rows if r['split']=='train'];selected=[]
 for c in CLASSES:
  pool=sorted([r for r in train if r['label']==c],key=lambda r:r['image']);indices=np.random.default_rng(42).choice(len(pool),4,replace=False);selected.extend(pool[int(i)] for i in indices)
 write(out/'source_data/QA_sample_selection.csv',selected);assets=out/'source_data/QA';assets.mkdir(exist_ok=True);qas=[];auglog=[];examples=[]
 fig,axs=plt.subplots(16,5,figsize=(13,34),layout='constrained')
 for i,r in enumerate(selected):
  rgb,mask,size=preprocess(r['source'],'RGB');views=[rgb];tensor(rgb).numpy().tofile(assets/(r['image']+'.float32_nchw.bin'));np.save(assets/(r['image']+'.normalized.npy'),tensor(rgb).numpy());rgb.save(assets/(r['image']+'_RGB.png'))
  for treatment in ['CC','CLAHE']:
   view,_,_=preprocess(r['source'],treatment);views.append(view);view.save(assets/(r['image']+'_'+treatment+'.png'));a=np.asarray(rgb);b=np.asarray(view);lab0=cv2.cvtColor(a.astype('float32')/255,cv2.COLOR_RGB2LAB);lab1=cv2.cvtColor(b.astype('float32')/255,cv2.COLOR_RGB2LAB);de=np.linalg.norm(lab0-lab1,axis=2)[mask];gray0=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);gray1=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY);g0=cv2.magnitude(cv2.Sobel(gray0,cv2.CV_32F,1,0),cv2.Sobel(gray0,cv2.CV_32F,0,1));g1=cv2.magnitude(cv2.Sobel(gray1,cv2.CV_32F,1,0),cv2.Sobel(gray1,cv2.CV_32F,0,1));ssim=structural_similarity(a,b,data_range=255,channel_axis=2);edge=float(g1[mask].mean()/max(g0[mask].mean(),1e-8));clip0=np.mean((a[mask]==0)|(a[mask]==255));clip1=np.mean((b[mask]==0)|(b[mask]==255));qa={'image':r['image'],'class':r['label'],'split':'train','treatment':treatment,'DeltaE76_mean':float(de.mean()),'DeltaE76_p95':float(np.percentile(de,95)),'SSIM_full_canvas':float(ssim),'edge_energy_ratio_content':edge,'clipping_fraction_increase':float(clip1-clip0),'engineering_QA_pass':bool(de.mean()<=3 and np.percentile(de,95)<=10 and ssim>=.9 and .75<=edge<=1.25 and clip1-clip0<=.01),'clinical_review':'not performed','mask':'known letterbox content rectangle only; NOT a lesion segmentation'};qas.append(qa)
  weak,g=geometry(rgb.copy(),stable_rng(42,1,r['image'],'geometry'));strong,p=acquisition(weak.copy(),stable_rng(42,1,r['image'],'acquisition'));views.extend([weak,strong]);auglog.append({'image':r['image'],'class':r['label'],'geometry':g,'acquisition':p,'seed':42,'epoch':1});examples.append((tensor(weak),tensor(strong),CLASSES.index(r['label'])))
  for ax,im,label in zip(axs[i],views,['RGB','mild CC','Luminance CLAHE','Weak view','Acquisition view']):ax.imshow(im);ax.axis('off');ax.set_title((r['label']+' · '+label) if i%4==0 else label,fontsize=8)
 fig.suptitle('ISIC training-only preprocessing and acquisition audit · fixed samples\nNo model results or clinical-validity claims');savefig(out,'source_preprocessing_audit',fig)
 write(out/'source_data/preprocessing_QA_metrics.csv',qas);dump(out/'source_data/augmentation_audit_parameters.json',auglog)
 torch.save({'weak':torch.stack([x[0] for x in examples]),'strong':torch.stack([x[1] for x in examples]),'target':torch.tensor([x[2] for x in examples])},out/'source_data/smoke_training_tensors.pt')
 # CPU preprocessing timing on a fixed representative development subset; no labels/images from heldout.
 sample=np.random.default_rng(42).choice(len(train),64,replace=False);timings=[]
 for treatment in ['RGB','CC','CLAHE']:
  elapsed=[]
  for i in sample:
   start=time.perf_counter();preprocess(train[int(i)]['source'],treatment);elapsed.append(time.perf_counter()-start)
  timings.append({'treatment':treatment,'images':64,'mean_seconds_per_image':float(np.mean(elapsed)),'median_seconds_per_image':float(np.median(elapsed)),'p95_seconds_per_image':float(np.percentile(elapsed,95)),'scope':'raw JPEG decode + EXIF/RGB + letterbox + optional deterministic treatment; warm OS cache; single worker'})
 write(out/'reports/CPU_pipeline_timing.csv',timings)
 # Actual graph profile: no optimizer, no training or model-result metrics.
 records=[];stages=[]
 for key in ['C0','C2']:
  cfg=load(out/f'configs/{key}.json');net=Student(cfg['fusion'],pretrained=True).eval();mac=[0];handles=[]
  for name,module in net.named_modules():
   if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):
    def hook(m,inp,y):
     if isinstance(m,torch.nn.Conv2d):mac[0]+=y.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
     else:mac[0]+=y.numel()*m.in_features
    handles.append(module.register_forward_hook(hook))
  x=torch.zeros(1,3,224,224)
  with torch.no_grad():y=net(x)
  for h in handles:h.remove()
  if key=='C0':
   with torch.no_grad():assert torch.equal(y,net.backbone(x)),'Standard architecture mismatch'
  assert list(y.shape)==[1,4];records.append({'configuration':key,'inference_parameters':sum(p.numel() for p in net.parameters()),'ConvLinear_MACs':mac[0],'input':[1,3,224,224],'architecture':ARCH,'fusion':cfg['fusion'],'skin_lesion_weights_used':False})
  if key=='C2':
   x=net.backbone.bn1(net.backbone.conv_stem(x))
   for i,block in enumerate(net.backbone.blocks):
    x=block(x);stages.append({'module':f'backbone.blocks.{i}','shape':list(x.shape),'fusion_tap':i in [1,2,4]})
 base,fused=records;assert fused['inference_parameters']/base['inference_parameters']<1.04 and fused['ConvLinear_MACs']/base['ConvLinear_MACs']<1.01
 write(out/'source_data/architecture_stages.csv',stages);dump(out/'reports/model_complexity.json',{'models':records,'C1_equals':'C0','C3_C4_equals':'C2','training_only_projector_parameters':sum(p.numel() for p in Projector().parameters()),'MAC_convention':'one multiply-accumulate per conv/linear output dot-product term; excludes pooling, normalization, activation, residual adds and memory traffic; not phone latency','parameter_growth_percent':100*(fused['inference_parameters']/base['inference_parameters']-1),'MAC_growth_percent':100*(fused['ConvLinear_MACs']/base['ConvLinear_MACs']-1)})
 # Source metadata figure, not a performance/results figure.
 fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained');x=np.arange(4)
 for j,split in enumerate(['train','validation']):ax.bar(x+(j-.5)*.35,[sum(r['split']==split and r['label']==c for r in rows) for c in CLASSES],width=.34,label=split)
 ax.set(xticks=x,xticklabels=CLASSES,ylabel='Approved images',title='ISIC source-only development class distribution');ax.legend();savefig(out,'source_class_distribution',fig)
 # Architecture schematic derived from measured modules, not a methodology workflow.
 from matplotlib.patches import FancyBboxPatch
 fig,ax=plt.subplots(figsize=(13,6),layout='constrained');ax.set_xlim(0,13);ax.set_ylim(0,6);ax.axis('off')
 def box(x,y,w,h,text):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor='#DCEAF2',edgecolor='#34495e'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
 box(.2,3.8,2,1.1,'RGB224×224\nSingle MobileNetV4\nbackbone');box(3,4.4,2.2,1.1,'blocks.1\n64×28×28\nGAP→Linear32→ReLU6\n→Linear1280');box(3,2.8,2.2,1.1,'blocks.2\n96×14×14\nGAP→Linear32→ReLU6\n→Linear1280');box(3,1.1,2.2,1.1,'blocks.4\n960×7×7\nOriginal pooled head\n→1280');box(7,2.8,2.4,1.2,'Deep residual fusion\nd + sigmoid(ρ)Σαₖuₖ\nα=softmax(θ)');box(10.4,2.8,2.2,1.2,'Original Linear\n1280→4 logits\nNV/MEL/BCC/SCC')
 for yy in [4.9,3.3,1.6]:ax.annotate('',xy=(6.9,3.4),xytext=(5.3,yy),arrowprops={'arrowstyle':'->'});ax.annotate('',xy=(2.9,yy),xytext=(2.25,4.3),arrowprops={'arrowstyle':'->'})
 ax.annotate('',xy=(10.3,3.4),xytext=(9.5,3.4),arrowprops={'arrowstyle':'->'});ax.text(6.6,.25,'C2–C4 inference graph · measured stage shapes · schematic abstraction\nC4 projector1280→128→64 and all losses are training-only; excluded from export.',ha='center');savefig(out,'actual_multiscale_architecture',fig)
 # Mathematical invariants, including limited-SCC cases.
 zz=torch.randn(8,64,requires_grad=True);labels=torch.tensor([0,1,2,3]*2);loss,st=supcon(zz,labels);assert torch.isfinite(loss);loss.backward();assert zz.grad.norm()>0
 zz2=torch.randn(8,64,requires_grad=True);zero,st2=supcon(zz2,torch.zeros(8,dtype=torch.long));assert zero==0;zero.backward();assert torch.isfinite(zz2.grad).all()
 assert js(torch.ones(2,4)/4,torch.ones(2,4)/4).abs().max()<1e-7
 # Dataset split guard, deterministic stateless geometry/acquisition, no crop dimensions.
 cfg=load(out/'configs/C4.json')
 for split in ['test','external']:
  try:SourceDataset(out,cfg,split)
  except AssertionError:pass
  else:raise AssertionError('Held-out access guard failed')
 d0=SourceDataset(out,load(out/'configs/C0.json'),'train',epoch=1);d1=SourceDataset(out,load(out/'configs/C1.json'),'train',epoch=1)
 a=d0[0];b=d1[0];assert torch.equal(a[0],b[0]);assert json.loads(a[-1])['geometry']==json.loads(b[-1])['geometry']
 dump(out/'reports/CPU_IMPLEMENTATION_CHECKS.json',{'status':'PASS','class_singleton_positive_guard':st,'single_class_batch_zero_loss':True,'standard_timm_graph_parity':True,'development_partition_guards':True,'C0_C1_weak_geometry_parity':True,'stateless_views':True,'QA_optional_treatments_pass':dict(Counter((r['treatment'],r['engineering_QA_pass']) for r in qas)) if False else [{'treatment':t,'pass':sum(r['treatment']==t and r['engineering_QA_pass'] for r in qas),'total':16} for t in ['CC','CLAHE']],'clinical_lesion_mask_available':False,'ROI_used':False,'full_training_performed':False})
 print('CPU IMPLEMENTATION CHECKS PASS',records,flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
