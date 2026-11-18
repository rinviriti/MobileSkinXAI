"""Untrained architecture-only TFLite smoke. CPU, no clinical/skin weights or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
from model import Student
import torch,time,importlib.metadata as md
import litert_torch
from ai_edge_litert.interpreter import Interpreter

def main(out):
 out=Path(out);torch.set_num_threads(2);reports=[]
 for key,fusion in [('C0',False),('C2_C3_C4',True)]:
  path=out/f'conversion/{key}_UNTRAINED_ARCHITECTURE_ONLY.tflite';assert not path.exists();net=Student(fusion,pretrained=False).cpu().eval();example=torch.randn(1,3,224,224);start=time.perf_counter();r={'graph':key,'fusion':fusion,'trained_lesion_weights_used':False,'official_pretrained_weights_used':False,'purpose':'random architecture feasibility only; not final model/export or clinical evidence'}
  try:
   converted=litert_torch.convert(net,(example,));converted.export(str(path));interpreter=Interpreter(model_path=str(path),num_threads=2);interpreter.allocate_tensors();ins=interpreter.get_input_details();outs=interpreter.get_output_details();assert len(ins)==len(outs)==1;errors=[]
   for seed in [42,43,44]:
    value=torch.randn(1,3,224,224,generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():expected=net(value).numpy()
    inp=value.numpy()
    if list(ins[0]['shape'])==[1,224,224,3]:inp=inp.transpose(0,2,3,1)
    interpreter.set_tensor(ins[0]['index'],np.ascontiguousarray(inp));interpreter.invoke();actual=interpreter.get_tensor(outs[0]['index']);np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3);errors.append(float(abs(actual-expected).max()))
   ops=sorted({x['op_name'] for x in interpreter._get_ops_details()});assert not any('Flex' in x or x=='CUSTOM' for x in ops)
   r.update(status='PASS',input_shape=ins[0]['shape'].tolist(),output_shape=outs[0]['shape'].tolist(),operators=ops,requires_Flex=False,requires_custom_ops=False,max_absolute_error=max(errors),inputs_compared=3,atol=1e-4,rtol=1e-3,file=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,seconds=time.perf_counter()-start)
  except Exception as e:r.update(status='FAIL',error=repr(e),seconds=time.perf_counter()-start);reports.append(r);dump(out/'reports/TFLITE_CONVERSION_SMOKE.json',reports);raise
  reports.append(r);dump(out/'reports/TFLITE_CONVERSION_SMOKE.json',reports);print(json.dumps(r),flush=True)
 dump(out/'conversion/environment.json',{k:md.version(k) for k in ['torch','timm','litert-torch','ai-edge-litert','litert-converter']})
if __name__=='__main__':main(ROOT/sys.argv[1])
