# Dataset preparation

All dataset files live under `dataset/` on the Ubuntu Linux filesystem and are
excluded from Git. Raw images, archives and publisher metadata are preserved.

## Official sources and terms

- ISIC 2019 training release: <https://challenge.isic-archive.com/data/#2019>.
  License: CC BY-NC 4.0 (attribution and noncommercial use). Retain attribution
  to BCN20000/Hospital Clínic de Barcelona, HAM10000/ViDIR Medical University
  of Vienna, and MSK; the source page provides the required paper citations.
  Training archive: 9,771,618,190 bytes; labels: 1,291,479 bytes;
  metadata: 1,214,351 bytes. The official challenge test archive is not needed:
  internal development partitions come only from the training release.
- PAD-UFES-20 v1: <https://data.mendeley.com/datasets/zr7vgbcyr2/1>.
  DOI: 10.17632/zr7vgbcyr2.1; license: CC BY 4.0, attribution to Pacheco et al.
  Three image ZIPs and metadata total 3,592,712,972 bytes. The downloader records
  the publisher file listing and verifies its SHA-256 checksums. Data paper:
  <https://doi.org/10.1016/j.dib.2020.106221>.

File sizes and licenses were checked before download. WSL reported about 946 GiB
free, and its D: backing filesystem about 390 GiB free. Downloads and extraction
also check remaining disk space. Download receipts record source URLs, sizes and
SHA-256 hashes; ISIC hashes are locally calculated, not publisher signatures.

## Split provenance and inclusion

No historical split CSVs were found in the repository, Linux home or mounted
Windows dataset locations. The user explicitly authorized a new split if none
could be recovered. `scripts/create_isic2019_splits.py --create-new-split` was
therefore used with seed 42 and its original sorted lesion-group algorithm.
Existing split directories cannot be overwritten. Counts match the historical
protocol, but the original image assignments were unavailable for comparison.

| ISIC partition | NV | MEL | BCC | SCC | Total |
|---|---:|---:|---:|---:|---:|
| Train | 7,956 | 3,011 | 2,293 | 426 | 13,686 |
| Validation | 1,635 | 586 | 519 | 95 | 2,835 |
| Test | 1,735 | 588 | 511 | 107 | 2,941 |
| Missing lesion ID, excluded | 1,549 | 337 | 0 | 0 | 1,886 |

Known lesion groups: 10,006 (train 7,006, validation 1,500, test 1,500).
Known lesion overlap is zero. Class indices: NV=0, MEL=1, BCC=2, SCC=3.

PAD is reserved entirely for external evaluation, never model selection or
training. Map NEV to NV; preserve MEL/BCC/SCC; exclude ACK and SEK from the
closed-set four-class evaluation. The source SCC category includes Bowen disease
(SCC in situ), which is a label-definition limitation when comparing datasets.
PAD bare lesion IDs can recur across patients: use `(patient_id, lesion_id)` for
within-patient lesion consistency and retain both original fields. Dataset-local
IDs do not establish identities across ISIC and PAD.

## Reproduction and verification

Use the project `.venv`. These commands never invoke model training or download
pretrained weights. They stop rather than overwrite completed preparations.

```bash
.venv/bin/python scripts/download_datasets.py isic
.venv/bin/python scripts/download_datasets.py pad
.venv/bin/python scripts/extract_dataset_archives.py isic
.venv/bin/python scripts/extract_dataset_archives.py pad

# Recover existing split CSVs when available. Only for explicitly authorized
# creation of a new split, and only if the target split directory is absent:
# .venv/bin/python scripts/create_isic2019_splits.py --create-new-split

.venv/bin/python scripts/preprocess_isic2019.py --audit-only
.venv/bin/python scripts/audit_datasets.py
.venv/bin/python scripts/preprocess_isic2019.py
.venv/bin/python scripts/preprocess_padufes20.py
```

Raw integrity checks fully decode images, verify JPEG/PNG structure, compute
SHA-256 of files and oriented RGB pixels, and search 64-bit DCT pHash neighbors
at Hamming distance <=6. Candidate pairs include a dHash distance, labels,
lesion identifiers, split boundaries and inclusion flags. These are review
candidates, not proof of a shared lesion; the screen can miss crops/rotations.
No image is removed or reassigned merely because of a perceptual hash match.

Prepared images use the existing `prepare_image` function: EXIF orientation,
RGB, aspect-preserving LANCZOS resize, centered gray(114) letterbox, 224x224,
JPEG quality 95 with subsampling disabled. No augmentation or normalization is
applied to any preparation. Future augmentations must occur in the training
loader only. Validation, internal test and PAD external images stay separate.

Audit manifests, exclusions, class statistics, metadata issues and duplicate
candidates are saved in `dataset/audit/`. Each prepared dataset also has its own
manifest and summary. Raw hashes allow subsequent verification without changing
source files. Patient independence is not established for ISIC challenge metadata.
