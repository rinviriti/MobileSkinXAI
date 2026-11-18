"""CPU-only ImageNet MobileNetV4 export probe, NOT a final lesion model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.metadata as md,json,time,subprocess
import numpy as np
import torch
from core import PROJECT,ROOT,config,model,digest,seed_all
import litert_torch
from ai_edge_litert.interpreter import Interpreter

def main():
 torch.set_num_threads(2);seed_all();cfg=config('M3');net=model(cfg,pretrained=True).cpu().eval()
 # Four-class head is newly initialized; backbone comes only from official ImageNet.
 example=torch.randn(1,3,224,224);out=PROJECT/'export_probe/mobilenetv4_ImageNet_backbone_random4class_probe.tflite'
 assert not out.exists();start=time.perf_counter()
 report={'purpose':'ImageNet backbone + random4class head conversion feasibility ONLY; not final model or thesis performance','trained_skin_lesion_weights_loaded':False,'architecture':cfg['architecture'],'pretrained_provenance':json.loads((ROOT/'final_training/pretrained/M3/provenance.json').read_text()),'input':[1,3,224,224],'input_contract':'Normalized RGB NCHW float32; aspect-preserving letterbox and normalization outside exported graph','versions':{k:md.version(k) for k in ['torch','timm','litert-torch','ai-edge-litert','litert-converter']}}
 try:
  converted=litert_torch.convert(net,(example,));converted.export(str(out));interpreter=Interpreter(model_path=str(out),num_threads=2);interpreter.allocate_tensors();ins=interpreter.get_input_details();outs=interpreter.get_output_details();assert len(ins)==len(outs)==1
  errors=[]
  for seed in [42,43,44]:
   value=torch.randn((1,3,224,224),generator=torch.Generator().manual_seed(seed))
   with torch.no_grad():expected=net(value).numpy()
   inp=value.numpy()
   if list(ins[0]['shape'])==[1,224,224,3]:inp=inp.transpose(0,2,3,1)
   interpreter.set_tensor(ins[0]['index'],np.ascontiguousarray(inp));interpreter.invoke();actual=interpreter.get_tensor(outs[0]['index'])
   np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3);errors.append(float(np.max(np.abs(actual-expected))))
  ops=sorted({o['op_name'] for o in interpreter._get_ops_details()})
  report.update(status='PASS',artifact=str(out.relative_to(ROOT)),sha256=digest(out),bytes=out.stat().st_size,parity_inputs=3,max_abs_error=max(errors),absolute_tolerance=1e-4,relative_tolerance=1e-3,operators=ops,requires_flex_ops=any('Flex' in op for op in ops),path='Official LiteRT Torch -> TFLite -> LiteRT CPU interpreter',device_latency_measured=False,quantization_tested=False)
 except Exception as e:report.update(status='FAIL',error=repr(e));raise
 finally:
  report['wall_seconds']=time.perf_counter()-start;(PROJECT/'reports/export_pretrained_tflite.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
 (PROJECT/'reports/requirements_export.lock.txt').write_text(subprocess.check_output([str(ROOT/'.venv-export-py311/bin/python'),'-m','pip','freeze'],text=True))
if __name__=='__main__':main()
