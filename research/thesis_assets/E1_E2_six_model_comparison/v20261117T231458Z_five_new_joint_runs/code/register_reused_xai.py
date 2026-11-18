from common import *
from evaluate_amended import KEYS,NAMES
import shutil
OUT=ROOT/sys.argv[1];original=ROOT/'research/thesis_assets/final_training_results/v20261117T091430Z_704ed3cc';index=read(OUT/'FIGURE_INDEX.csv');known={r['figure_id'] for r in index}
for k in KEYS:
 source=original/f'11_Source_Data/{k}_CUDA_BF16_GradCAM_examples.csv';dest=OUT/f'09_Explainability/E1_{k}_CASES_REUSED.csv'
 if not dest.exists():shutil.copy2(source,dest)
 assert sha(source)==sha(dest)
 fid='E1_'+k+'_reused_GradCAM'
 if fid in known:continue
 paths={e:f'09_Explainability/E1_{k}_reused.{e}' for e in ['png','svg','pdf']}
 for e,path in paths.items():assert sha(OUT/path)==sha(original/f'10_Explainability/10_{k}_GradCAM_CUDA_BF16.{e}')
 index.append({'figure_id':fid,'title':'E1 '+NAMES[k]+' original ISIC validation Grad-CAM (reused)','caption':'Copied unchanged from the verified original E1 CUDA BF16 Grad-CAM package. Original ISIC development-validation cases, not current PAD test cases. These are separate illustration cohorts, not paired localization comparisons or clinical evidence. No on-device Flutter explainability claim. PAD study findings are post-hoc exploratory.','source_data':str(dest.relative_to(OUT)),'source_sha256':sha(dest),'script':'code/register_reused_xai.py','PNG_DPI':300,'thesis_shortlist':False,'original_generation_script':str((original/'11_Source_Data/generation_scripts/xai_cuda.py').relative_to(ROOT)),**paths});known.add(fid)
write(OUT/'FIGURE_INDEX.csv',index)
print('Registered',len(index),'total figure families, including seven reused E1/M3 families')
