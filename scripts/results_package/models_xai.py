"""Actual trained architecture profiles and standard gradient-weighted CAM; no training."""
import gc,math
from collections import Counter
from common import *
import torch
import timm
import torch.nn.functional as F
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

def load_net(c,k):
 cfg=c['models'][k]['config'];cp_path=source(c,f'checkpoints/{k}_selected_checkpoint.pt');assert sha(cp_path)==c['models'][k]['checkpoint_sha256'];cp=torch.load(cp_path,map_location='cpu',weights_only=True);net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=4);net.load_state_dict(cp['state_dict'],strict=True);net.eval();del cp;return net

def stage_paths(net):
 if hasattr(net,'stages'):
  return [('stem','Stem',type(net.stem).__name__)]+[(f'stages.{i}',f'Feature stage {i+1}',f"{len(s.blocks)} × {type(s.blocks[0]).__name__}") for i,s in enumerate(net.stages)]+[('head','Global pooling / normalized classifier',type(net.head).__name__)]
 paths=[('bn1','Stem: convolution + normalization',type(net.bn1).__name__)]
 for i,stage in enumerate(net.blocks):
  count=Counter(type(x).__name__ for x in stage);detail=', '.join(f'{v} × {t}' for t,v in count.items());paths.append((f'blocks.{i}',f'Feature stage {i+1}',detail))
 # Execution order is recovered by hooks, not assumed from this registration list.
 for p,label in [('global_pool','Global pooling'),('conv_head','Convolutional head'),('classifier','Four-class classifier')]:
  if hasattr(net,p):paths.append((p,label,type(getattr(net,p)).__name__))
 return paths

def diagram(c,k,records,path):
 nodes=[{'stage':'Input (normalized RGB)','detail':'NCHW; batch1 shape probe','output_shape':[1,3,224,224]}]+records;cols=4;rows=math.ceil(len(nodes)/cols);fig,ax=plt.subplots(figsize=(14,rows*2.05+1.0),layout='constrained');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');positions=[]
 for i,r in enumerate(nodes):
  row=i//cols;column=i%cols if row%2==0 else cols-1-i%cols;x=.13+column*.245;y=1-(row+.65)/(rows+.05);positions.append((x,y));width=.218;height=.65/(rows+.05)
  patch=FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle='round,pad=0.008',edgecolor=COLORS[k],facecolor=COLORS[k]+'22',lw=1.4);ax.add_patch(patch)
  shape=' × '.join(str(v) for v in r['output_shape']);detail='\n'.join(textwrap.wrap(r['detail'],32));ax.text(x,y,r['stage']+'\n'+detail+'\n'+shape,ha='center',va='center',fontsize=8)
 for (x,y),(xx,yy) in zip(positions,positions[1:]):
  if abs(y-yy)<1e-5:start=(x+.115*np.sign(xx-x),y);end=(xx-.115*np.sign(xx-x),yy)
  else:start=(x,y-.35/(rows+.05));end=(xx,yy+.35/(rows+.05))
  ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,color='#555555',lw=1))
 ax.set_title(NAMES[k]+'\nCode-derived major-stage architecture (schematic abstraction)',fontsize=12,pad=12)
 save_figure(c,'02_Model_Architectures',k+'_architecture',fig,NAMES[k]+' — actual architecture schematic','Stage order and NCHW shapes were measured from the actual trained timm model on one synthetic1×3×224×224tensor. Boxes abstract major stages; complete registered leaf-layer summaries and model repr are supplied. '+('The inference student is unchanged MobileNetV4-Conv-Small; ConvNeXt-Tiny, weak/strong view branches, gates and distillation exist only during training and are excluded from this inference graph.' if k in ['M5','M5-LR'] else 'The four-logit classifier maps to NV,MEL,BCC,SCC. No image from test/PAD is used.'),[path,source(c,f'model_definitions/{k}_model_repr.txt'),source(c,f'model_definitions/{k}_layers.csv')],split='architecture synthetic input only',models=[k],method='actual module hooks with schematic aggregation',script='models_xai.py')

def models(c):
 torch.set_num_threads(2);rows=[]
 for k in KEYS:
  net=load_net(c,k);leaf=[];handles=[];macs=[0];stages=[];definitions=source(c,'model_definitions');definitions.mkdir(exist_ok=True)
  def make_leaf(name,mod):
   def hook(module,inputs,output):
    outshape=list(output.shape) if torch.is_tensor(output) else str(type(output));value=0
    if isinstance(module,torch.nn.Conv2d):value=int(output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1])
    elif isinstance(module,torch.nn.Linear):value=int(output.numel()*module.in_features)
    macs[0]+=value;leaf.append({'module_path':name,'module_type':type(module).__name__,'output_shape':json.dumps(outshape),'direct_parameters':sum(p.numel() for p in module.parameters(recurse=False)),'counted_Conv2d_Linear_MACs':value})
   return hook
  for name,m in net.named_modules():
   if name and not list(m.children()):handles.append(m.register_forward_hook(make_leaf(name,m)))
  for name,label,detail in stage_paths(net):
   def stagehook(module,inputs,output,name=name,label=label,detail=detail):stages.append({'module_path':name,'stage':label,'detail':detail,'output_shape':list(output.shape),'stage_parameters':sum(p.numel() for p in module.parameters())})
   handles.append(net.get_submodule(name).register_forward_hook(stagehook))
  with torch.inference_mode():out=net(torch.zeros(1,3,224,224));assert list(out.shape)==[1,4]
  for h in handles:h.remove()
  total=sum(p.numel() for p in net.parameters());fine=sum(p.numel() for p in net.parameters() if p.requires_grad);head=sum(p.numel() for p in net.get_classifier().parameters());bufferbytes=sum(b.numel()*b.element_size() for b in net.buffers());parambytes=sum(p.numel()*p.element_size() for p in net.parameters())
  rows.append({'model':k,'model_name':NAMES[k],'architecture_identifier':c['models'][k]['config']['architecture'],'parameters_total':total,'fine_tuning_trainable_parameters':fine,'frozen_head_stage_trainable_parameters':head if k not in ['M5','M5-LR'] else 'no frozen-head stage','registered_leaf_modules':sum(1 for m in net.modules() if not list(m.children())),'counted_Conv2d_Linear_MACs':macs[0],'convention_FLOPs_2times_MACs':2*macs[0],'fp32_parameter_tensor_bytes':parambytes,'buffer_tensor_bytes':bufferbytes,'selected_full_training_checkpoint_bytes':c['models'][k]['checkpoint'].stat().st_size,'teacher_excluded_from_inference':k in ['M5','M5-LR'],'teacher_training_parameters_if_applicable':next(r['parameters_total'] for r in rows if r['model']=='M4') if k in ['M5','M5-LR'] else 0})
  write_csv(definitions/f'{k}_layers.csv',leaf);sp=write_csv(definitions/f'{k}_major_stages.csv',[{**r,'output_shape':json.dumps(r['output_shape'])} for r in stages]);write_json(definitions/f'{k}_major_stages.json',stages)
  reprpath=definitions/f'{k}_model_repr.txt'
  if not reprpath.exists():reprpath.write_text(str(net)+'\n')
  diagram(c,k,stages,sp);del net;gc.collect()
 complexity=write_csv(source(c,'model_complexity.csv'),rows)
 # Training-only modules: exactly the implemented enhanced-model computation, no inference teacher.
 graph=write_json(source(c,'enhancement_module_graph.json'),{'student':'MobileNetV4-Conv-Small, two train-view forwards','teacher':'fresh final ConvNeXt-Tiny, frozen eval and no_grad, two view forwards','input':'training-only weak/strong views with shared flip/rotation','gate':'confidence>=0.7, normalized mean entropy<=0.6, teacher viewJS<=0.1, both predictions equal training label','loss':'weightedCE+0.5 cross-viewKD(T4)+0.1 gated studentJS','gradients':'student only; teacher/reliability detached','inference':'one student forward, no teacher/gates/views/KD'})
 fig,ax=plt.subplots(figsize=(14,7),layout='constrained');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 nodes={'views':(.12,.55,'Training RGB224×224\nShared geometry → weak / strong views'),'teacher':(.40,.79,'ConvNeXt-Tiny teacher\nFrozen eval; two no-grad forwards'),'student':(.40,.32,'MobileNetV4-Conv-Small student\nTwo trainable forwards'),'gate':(.72,.79,'Training-only reliability gate\nConfidence, entropy, view JS\nBoth teacher predictions correct'),'loss':(.73,.32,'Weighted CE +0.5KD(T=4)+0.1JS\nCross-view targets / gated consistency'),'labels':(.73,.08,'Training labels only\nNV, MEL, BCC, SCC'),'update':(.95,.32,'Student-only\nAdamW update')}
 for key,(x,y,text) in nodes.items():
  width=.235 if key!='update' else .10;height=.15 if key!='labels' else .10;color='#9C755F' if key in ['teacher','gate'] else '#B07AA1';ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle='round,pad=0.01',edgecolor=color,facecolor=color+'20'));ax.text(x,y,text,ha='center',va='center',fontsize=8.5)
 for start,end in [((.24,.60),(.28,.79)),((.24,.50),(.28,.32)),((.52,.79),(.60,.79)),((.72,.70),(.72,.41)),((.52,.32),(.60,.32)),((.73,.14),(.73,.23)),((.85,.32),(.89,.32))]:ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,color='#555555'))
 ax.annotate('Training-label correctness',xy=(.83,.70),xytext=(.85,.12),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=.35'},fontsize=8)
 ax.text(.04,.94,'MobileSkinXAI-Enhanced: training-only module graph (code-derived schematic)',fontsize=13);ax.text(.04,.01,'Final inference: one MobileNetV4-Conv-Small student only. Teacher and loss branches are absent.',fontsize=10)
 save_figure(c,'02_Model_Architectures','enhanced_training_modules',fig,'MobileSkinXAI-Enhanced — teacher/student training-only modules','Abstraction of the actual code, not an end-to-end methodology workflow. Teacher/reliability quantities are detached; only the student is optimized. Both original and LR-control enhanced models use this same graph and four-logit inference student. The LR control changes only head/backbone LRs.',[graph,source(c,'recorded_runs/M5/frozen_configuration.json'),source(c,'recorded_runs/M5-LR/frozen_configuration.json')],split='training computation schematic only',models=['M5','M5-LR'],method='code-derived training architecture schematic',script='models_xai.py')
 metrics={r['model']:r for r in read_csv(source(c,'model_metrics.csv'))}
 for field,title,divisor,label in [('parameters_total','Inference model parameter count',1e6,'Parameters (millions)'),('counted_Conv2d_Linear_MACs','Convolution and linear MACs at224×224',1e6,'Conv2d/Linear MACs (millions)'),('selected_full_training_checkpoint_bytes','Selected PyTorch training-checkpoint sizes',2**20,'Full checkpoint size (MiB)')]:
  fig,ax=horizontal(c,[float(r[field])/divisor for r in rows],title,label);save_figure(c,'08_Model_Complexity',field,fig,title,'Actual four-class inference architecture at batch1, RGB224×224. '+('MAC convention: one multiply-accumulate=oneMAC; grouped/depthwise convolution and linear operations included. Bias additions, normalization, activation, residual addition, pooling, memory movement and functional operations are excluded. FLOPs=2×these countedMACs, not a full-operator FLOP total.' if 'MACs' in field else 'Training checkpoint sizes include optimizer/scaler/provenance and are not deployment weights or TFLite sizes.' if 'checkpoint' in field else 'All backbone/head parameters can train during fine-tuning; BN running statistics are frozen buffers. Teacher parameters are excluded from enhanced inference counts. Frozen-head trainable counts are in source data.'),[complexity],split='trained architecture / checkpoint metadata',models=KEYS,script='models_xai.py')
 # Explicit total vs trainable comparison (identical in full fine-tuning is expected).
 fig,ax=plt.subplots(figsize=(12,5.8),layout='constrained');y=np.arange(6)
 ax.barh(y-.17,[r['parameters_total']/1e6 for r in rows],.32,color=[COLORS[k] for k in KEYS],label='Total parameters');ax.barh(y+.17,[r['fine_tuning_trainable_parameters']/1e6 for r in rows],.32,color=[COLORS[k] for k in KEYS],alpha=.4,label='Trainable in fine-tuning');ax.set_yticks(y,[full_name(k) for k in KEYS]);ax.invert_yaxis();ax.set(xlabel='Parameters (millions)',title='Total and fine-tuning trainable parameters');ax.legend(frameon=False);save_figure(c,'08_Model_Complexity','total_trainable_parameters',fig,'Total and trainable inference-model parameters','Fine-tuning enables all student backbone/head parameters. Frozen BN running statistics are not parameters. Frozen-head stage counts are recorded separately; the enhanced models do not repeat that stage. No frozen teacher is counted as deployed student parameters.',[complexity],split='architecture/configuration only',models=KEYS,script='models_xai.py')
 for field,title,scale,label in [('training_wall_seconds','Measured training durations',60,'Training wall time (minutes)'),('peak_allocated_GiB','Measured peak CUDA allocation',1,'Peak PyTorch allocated GiB')]:
  fig,ax=horizontal(c,[float(metrics[k][field])/scale for k in KEYS],title,label);save_figure(c,'08_Model_Complexity',field,fig,title,'Actual recorded run values, not estimated phone latency. '+('Time includes epochs, validation and checkpoint writing. Enhanced times are additional runs; parent/teacher costs are reported in the narrative and must not be ignored.' if field=='training_wall_seconds' else 'Peak CUDA allocation is PyTorch allocator memory, excluding driver/desktop allocations; no unavailable peak reserved-memory value is fabricated.'),[source(c,'model_metrics.csv')],split='training resource measurements',models=KEYS,script='models_xai.py')
 fig,ax=plt.subplots(figsize=(10.5,7),layout='constrained')
 for k,r,marker in zip(KEYS,rows,['o','s','^','D','P','X']):ax.scatter(r['parameters_total']/1e6,float(metrics[k]['macro_f1']),color=COLORS[k],marker=marker,s=110,label=NAMES[k],edgecolor='black',linewidth=.5)
 ax.set(xlabel='Inference parameters (millions; logarithmic axis)',ylabel='Validation macro-F1',title='Inference parameter count versus validation macro-F1');ax.set_xscale('log');ax.set_ylim(.60,.76);fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=8);save_figure(c,'08_Model_Complexity','parameters_macro_f1_scatter',fig,'Parameter count versus selected-checkpoint validation macro-F1','Same-architecture enhanced students have identical parameter counts and different recorded validation scores. Marker shapes supplement the prescribed palette. Model selection, extra optimization and unequal teacher/parent compute preclude a simple causal parameter-efficiency claim; no phone latency is implied.',[complexity,source(c,'model_metrics.csv')],models=KEYS,script='models_xai.py')
 write_json(source(c,'MODELS_COMPLETE.json'),{'profiles':6,'MAC_convention':'Conv2d/Linear only, batch1x3x224x224, oneMAC=multiply+accumulate, FLOPs2MAC excluding otherops','input':'synthetic zeros; no test/PAD image','four_unique_inference_architectures':4})

def xai(c):
 torch.set_num_threads(2);allprobs=read_csv(source(c,'selected_validation_probabilities.csv'));val=c['validation'];allrows=[];missing=[];image_reads=[]
 assets=source(c,'xai_input_assets');assets.mkdir(exist_ok=True);rawfolder=source(c,'gradcam_maps');rawfolder.mkdir(exist_ok=True)
 for k in KEYS:
  net=load_net(c,k);net.requires_grad_(False);prefix='stages' if hasattr(net,'stages') else 'blocks';target_path=f'{prefix}.{len(getattr(net,prefix))-1}';target_layer=net.get_submodule(target_path);selected=[];rs=[r for r in allprobs if r['model']==k]
  for cl in CLASSES:
   for correct in [True,False]:
    pool=[r for r in rs if r['true_class']==cl and ((r['predicted_class']==cl)==correct)]
    if not pool:missing.append({'model':NAMES[k],'true_class':cl,'correct':correct,'reason':'No recorded validation example in this group'});continue
    confidence=np.array([float(r[r['predicted_class']]) for r in pool]);median=float(np.median(confidence));chosen=sorted(pool,key=lambda r:(abs(float(r[r['predicted_class']])-median),r['image']))[0];selected.append((cl,correct,chosen))
  panels={};modelrecords=[]
  for cl,correct,r in selected:
   imageid=r['image'];meta=val[imageid];path=ROOT/meta['processed_path'];assert meta['split']=='validation' and meta['release_eligible']=='True' and sha(path)==meta['processed_sha256'];asset=assets/(imageid+'.jpg')
   if not asset.exists():shutil.copyfile(path,asset)
   with Image.open(asset) as opened:im=opened.convert('RGB');assert im.size==(224,224);a=np.asarray(im).copy()
   image_reads.append({'model':NAMES[k],'image':imageid,'split':'validation','processed_sha256':sha(asset)})
   x=torch.from_numpy(a).permute(2,0,1).float().unsqueeze(0)/255;cfg=c['models'][k]['config'];mean=torch.tensor(cfg['normalization']['mean']).view(1,3,1,1);std=torch.tensor(cfg['normalization']['std']).view(1,3,1,1);x=((x-mean)/std).requires_grad_(True);capture={}
   def hook(module,inputs,output):capture['activation']=output
   handle=target_layer.register_forward_hook(hook)
   try:
    logits=net(x);p=logits.softmax(-1)[0];target=int(logits.argmax(1));saved=np.array([float(r[s]) for s in CLASSES]);diff=float(np.abs(p.detach().numpy()-saved).max());assert target==CLASSES.index(r['predicted_class']),'FP32 predicted class differs from recorded BF16 validation result';assert diff<=.02,'FP32/BF16 probability parity beyond predefined0.02absolute tolerance'
    activation=capture['activation'];assert activation.ndim==4 and min(activation.shape[-2:])>1;gradient=torch.autograd.grad(logits[0,target],activation)[0];assert torch.isfinite(gradient).all();alpha=gradient.mean((2,3),keepdim=True);cam=torch.relu((alpha*activation).sum(1))[0];raw=cam.detach().numpy();normal=raw-raw.min();maximum=float(normal.max());degenerate=maximum<=1e-12;normal=normal/maximum if not degenerate else np.zeros_like(normal);resized=F.interpolate(torch.from_numpy(normal).view(1,1,*normal.shape),size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy();heat=plt.colormaps['magma'](resized)[...,:3];overlay=np.clip(.60*(a/255)+.40*heat,0,1)
    camfile=rawfolder/f'{k}_{cl}_{"correct" if correct else "incorrect"}_{imageid}.csv';write_csv(camfile,[{'row':i,**{f'column{j}':float(normal[i,j]) for j in range(normal.shape[1])}} for i in range(normal.shape[0])]);npy=rawfolder/(camfile.stem+'_upsampled.npy')
    if not npy.exists():np.save(npy,resized.astype(np.float32))
    record={'model':k,'model_name':NAMES[k],'image':imageid,'true_class':cl,'predicted_class':CLASSES[target],'correct':correct,'selection':'closest to median saved top1 confidence within true-class/correctness group; lexicographic tie','target_score':'predicted class raw logit, not true-class target or softmax','target_layer':target_path,'feature_shape':json.dumps(list(activation.shape)),'gradient_L2':float(gradient.square().sum().sqrt()),'FP32_top1_probability':float(p[target].detach()),'saved_BF16_top1_probability':float(saved[target]),'max_abs_probability_difference':diff,'parity_absolute_tolerance':.02,'map_minmax_normalized_per_image':True,'zero_positive_map':degenerate,'map_csv':str(camfile.relative_to(c['out'])),'upsampled_map_npy':str(npy.relative_to(c['out'])),'input_asset':str(asset.relative_to(c['out'])),'input_sha256':sha(asset),'checkpoint_sha256':c['models'][k]['checkpoint_sha256'],**{f'FP32_p_{cl2}':float(p[j].detach()) for j,cl2 in enumerate(CLASSES)}};allrows.append(record);modelrecords.append(record);panels[(cl,correct)]=(a,overlay,record)
   except Exception as exc:missing.append({'model':NAMES[k],'true_class':cl,'correct':correct,'image':imageid,'reason':repr(exc)})
   finally:handle.remove()
  sp=write_csv(source(c,f'{k}_gradcam_examples.csv'),modelrecords) if modelrecords else write_json(source(c,f'{k}_gradcam_unavailable.json'),missing)
  fig,axes=plt.subplots(4,4,figsize=(13,11),layout='constrained')
  for i,cl in enumerate(CLASSES):
   for j,correct in enumerate([True,False]):
    axes[i,2*j].axis('off');axes[i,2*j+1].axis('off');pair=panels.get((cl,correct))
    if not pair:axes[i,2*j+1].text(.5,.5,'Unavailable: see source log',ha='center',va='center',fontsize=9);continue
    a,overlay,r=pair;axes[i,2*j].imshow(a);axes[i,2*j+1].imshow(overlay);axes[i,2*j].set_title(f'{cl}: '+('correct' if correct else 'incorrect')+'\n'+r['image'],fontsize=8);axes[i,2*j+1].set_title('Grad-CAM for '+r['predicted_class']+f"\np={r['FP32_top1_probability']:.3f}"+(' (zero map)' if r['zero_positive_map'] else ''),fontsize=8)
  fig.suptitle(NAMES[k]+'\nStandard Grad-CAM on approved validation examples',fontsize=13);sources=[sp]+[c['out']/r['map_csv'] for r in modelrecords]+[c['out']/r['input_asset'] for r in modelrecords]
  save_figure(c,'10_Explainability',k+'_GradCAM',fig,NAMES[k]+' — validation Grad-CAM','Actual trained-checkpoint standard Grad-CAM: alpha_k=mean_spatial(d predicted-class logit/d featureA_k), heatmap=ReLU(sum alpha_k A_k), then per-image minmax normalization and bilinear upsampling. Each true class has one correct and one incorrect example where available, selected near median confidence, not for pleasing maps. Original image and overlay are adjacent. CPUfloat32 predictions are parity-checked against saved BF16 probabilities (same top1, max absolute difference<=0.02). This is Grad-CAM, not Grad-CAM++, Score-CAM or a segmentation/clinical-localization validation. No Flutter XAI or faithfulness claim is made.',sources,models=[k],method='standard Grad-CAM on predicted raw logit; final spatial stage; CPUfloat32; no optimization',script='models_xai.py')
  del net;gc.collect();print('GRADCAM '+NAMES[k],flush=True)
 write_csv(source(c,'all_GradCAM_examples.csv'),allrows);write_csv(source(c,'XAI_image_access_log.csv'),image_reads);write_json(source(c,'XAI_availability.json'),{'requested_examples':48,'generated_examples':len(allrows),'unavailable':missing,'zero_positive_maps':sum(r['zero_positive_map'] for r in allrows),'model_parameter_updates':0,'input_splits':['validation'],'test_PAD_images_opened':False,'CAM_definition':'standard Grad-CAM, not a substitute alternative','no_segmentation_ground_truth_evaluation':True})
