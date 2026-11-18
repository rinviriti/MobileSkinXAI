"""Create/verify a source-and-model archive; never modifies research outputs."""
from pathlib import Path
import subprocess,zipfile,hashlib,json,argparse
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'release_dist');p.add_argument('--include-ios-framework',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 subprocess.run(['python3',str(ROOT/'scripts/release_integrity.py')],check=True)
 version=(ROOT/'docs/RELEASE_VERSION').read_text().strip();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();target=a.output/f'MobileSkinXAI-{version}.zip'
 assert not target.exists(),'Preserve/version the existing archive before rebuilding'
 files=[ROOT/name for name in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if name!='app/flutter/assets/models/mobileskinxai.tflite']
 framework=ROOT/'app/flutter/packages/mobileskinxai_codec/ios/Frameworks/MobileSkinXaiCodec.xcframework'
 if a.include_ios_framework:
  assert framework.is_dir(),'CI-verified iOS XCFramework is required';files.extend(p for p in framework.rglob('*') if p.is_file())
 contents={}
 with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(set(files)):
   name=str(f.relative_to(ROOT));assert not any(x in f.parts for x in ['.private','.venv','dataset','Pods','.dart_tool']);data=f.read_bytes();z.writestr('MobileSkinXAI/'+name,data);contents[name]=hashlib.sha256(data).hexdigest()
  z.writestr('MobileSkinXAI/PACKAGE_PROVENANCE.json',json.dumps({'source_commit':sha,'version':version,'iOS_XCFramework_included':a.include_ios_framework,'file_SHA256':contents},indent=2))
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None
  name='MobileSkinXAI/app/flutter/assets/models/e2_lr_control_v20261118/model_float32.tflite';assert hashlib.sha256(z.read(name)).hexdigest()=='8b6e19f3d8d0dd9c3b798a2cf99c0525079b9de4ea2f5d1708dd928a718412f3'
 digest=hashlib.sha256(target.read_bytes()).hexdigest();(a.output/'SHA256SUMS.txt').write_text(digest+'  '+target.name+'\n');print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'sha256':digest,'files':len(contents)},indent=2))
if __name__=='__main__':main()
