"""Real CPU inference parity against preserved trained PyTorch tensors.

No training, calibration or held-out evaluation. Requires the verification env.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch,timm
from ai_edge_litert.interpreter import Interpreter
from preprocess_reference import preprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
 torch.set_num_threads(2);meta=json.loads((ROOT/'deployment/verification/MODEL_PROVENANCE.json').read_text());weights=ROOT/'deployment/verification/student_reference.pt';assert hashlib.sha256(weights.read_bytes()).hexdigest()==meta['pytorch_reference_sha256']
 checkpoint=torch.load(weights,map_location='cpu',weights_only=True);assert checkpoint['classes']==['NV','MEL','BCC','SCC'];net=timm.create_model(checkpoint['architecture'],pretrained=False,num_classes=4).eval();net.load_state_dict(checkpoint['state_dict'],strict=True)
 model=ROOT/meta['primary_model'];assert hashlib.sha256(model.read_bytes()).hexdigest()==meta['primary_sha256'];lite=Interpreter(model_path=str(model),num_threads=2);lite.allocate_tensors();inp=lite.get_input_details()[0];out=lite.get_output_details()[0];assert inp['shape'].tolist()==[1,224,224,3] and out['shape'].tolist()==[1,4]
 rows=json.loads((ROOT/'.private/golden_manifest.json').read_text());errors=[]
 for r in rows:
  x=preprocess(r['source']);lite.set_tensor(inp['index'],x);lite.invoke();actual=lite.get_tensor(out['index'])
  with torch.inference_mode():expected=net(torch.from_numpy(x.transpose(0,3,1,2).copy())).numpy()
  def softmax(a):e=np.exp(a-a.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
  ld=abs(actual-expected);pd=abs(softmax(actual)-softmax(expected));assert ld.max()<=.001 and ld.mean()<=.0001 and pd.max()<=.0001 and pd.mean()<=.00001 and actual.argmax()==expected.argmax();errors.append({'image':r['image'],'max_abs_logit':float(ld.max()),'max_abs_probability':float(pd.max())})
 print(json.dumps({'status':'PASS','images':errors,'training_performed':False},indent=2))
if __name__=='__main__':main()
