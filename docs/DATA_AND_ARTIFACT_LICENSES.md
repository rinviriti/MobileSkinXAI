# Data and artifact rights / privacy inventory

No full datasets, archives, patient/demographic records, original patient/lesion IDs, virtualenvs, credentials or WSL system files are published. Frozen split digests are non-identifying image integrity/membership keys; actual medical grouping data remains local. Source code names metadata fields but includes no patient records.

## ISIC development fixtures — CC BY-NC4.0

Exactly four already fixed, anonymous ISIC2019 development-validation photographs are embedded in the test-only golden_cases.dart. They are NOT held-out test cases, a training archive or production assets. All four have noEXIF/JPEGcomment metadata and were visually reviewed as lesion closeups without faces, names or other visible identifying information. Hashes/numeric reference outputs are listed in GOLDEN_REFERENCE.json. The original image bytes are retained for parity; no extra images or private metadata are embedded. Restrict these fixtures/derivatives to the source noncommercial/attribution terms; no blanket commercial permission is asserted.

Official source/license: [ISIC Challenge2019](https://challenge.isic-archive.com/data/#2019), [CC BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/). Aggregate attribution: BCN20000/Department of Dermatology, Hospital Clínic de Barcelona; HAM10000/ViDIR Group, Medical University of Vienna; MSK dataset, credited as Anonymous by the publisher. Cite Tschandl/Rosendahl/Kittler(2018), HAM10000, Scientific Data5:180161, doi10.1038/sdata.2018.161; Codella etal.(2017), arXiv1710.05006; Hernández-Pérez etal.(2024), BCN20000, Scientific Data11:641. The official page provides the complete references.

## PAD training illustrations and genuine Grad-CAM — CC BY4.0

One metadata-stripped4-class training illustration grid and six existing development-validation Grad-CAM panels are included, not full clinical records. They were visually reviewed as lesion closeups without visible identifying text/faces. Public titles/filenames omit patientIDs. No identifier CSV, per-patient predictions or medical questionnaire is included. Changes:RGB letterboxing/224preparation and, for Grad-CAM, the actual trained-model map/overlay plus academic labels. These remain research illustrations, not clinical explanations or on-device Flutter Grad-CAM.

Attribution: Pacheco etal., PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones, Data in Brief32(2020)106221, doi10.1016/j.dib.2020.106221. [Official dataset](https://data.mendeley.com/datasets/zr7vgbcyr2/1), [publisher paper and open-data license](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/), [CC BY4.0](https://creativecommons.org/licenses/by/4.0/). Source studies supply consent/ethics statements; this project does not assert a new clinical consent or validation study.

## Code, weights and third parties

Native libjpeg-turbo3.2.0 retains its IJG/BSD/zlib notices; Pillow geometry retains its notice. See the native package LICENSE/THIRD_PARTY_NOTICES.md. Dependency licenses do not grant rights to unrelated project code/data. No new repository-wide permissive license is imposed. The project owner authorized distribution of the trained student artifacts; this is a nonclinical research release, not a grant of upstream data rights for commercial use.

The trained Float32 file and inference-only PyTorch reference preserve identical student weights; optimizer/RNG/training checkpoints are excluded. Failed Float16variants, uncalibratedINT8, extra weights and dataset archives are not shipped. Source clinical image licenses remain distinct from aggregate metric/plot and project-code rights. The local source research archive/Thesis Results is preserved; public copies omit private identifiers and raw metadata.
