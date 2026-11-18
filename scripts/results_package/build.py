"""Reproducible phased generator; never performs model training or test evaluation."""
import argparse,uuid
from common import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['init','prepare','plots','models','xai','finish']);ap.add_argument('--output');a=ap.parse_args()
 if a.phase=='init':
  version='v'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];out=ROOT/'research/thesis_assets/final_training_results'/version;out.mkdir(parents=True,exist_ok=False)
  for d in CATEGORIES:(out/d).mkdir()
  write_json(out/'00_Executive_Summary/PACKAGE_SCOPE.json',{'official_thesis_title':TITLE,'dataset_release':RELEASE.name,'validation_only_models':list(NAMES.values()),'no_training':True,'no_test_or_PAD_image_evaluation':True})
  print(rel(out));return
 assert a.output; c=load_context(a.output)
 if a.phase=='prepare':from prepare import prepare;prepare(c)
 elif a.phase=='plots':from plots import plots;plots(c)
 elif a.phase=='models':from models_xai import models;models(c)
 elif a.phase=='xai':from xai_cuda import generate;generate(c)
 elif a.phase=='finish':from finish import finish;finish(c)
if __name__=='__main__':main()
