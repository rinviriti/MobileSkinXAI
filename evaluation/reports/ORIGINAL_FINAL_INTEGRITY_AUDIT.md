# Final integrity audit

PASS with documented M1 controller deviation and explicit post-hoc weaker-teacher amendment.

| check | status | evidence |
| --- | --- | --- |
| Historical artifacts unchanged | PASS | All original preservation-manifest hashes verified, including E1 checkpoints, M3 E2 and original results/figures |
| Original and corrected controller seals | PASS | Both LOCK.json and CONTROLLER_FIX_LOCK.json match |
| Pre-evaluation integrity | PASS | Canonical release checksums plus every source/prepared held-out file hash checked, twelve checkpoint states loaded strictly with finite weights before inference; see FINAL_PRE_EVALUATION_INTEGRITY.json and frozen evaluation lock |
| Amendment and permanent historical preservation | PASS | All pre-amendment artifact hashes match; failed qualification remains false; amendment and teacher-use authorization seals intact |
| Frozen PAD identity and grouping | PASS | Original byte-identical split; 781/261/261 images; patient disjoint; [private group] retained only in validation |
| No held-out training membership | PASS | DEVELOPMENT_MANIFEST includes only fixed train/validation; evaluation manifests disjoint |
| New E2 run counts and stopping | PASS | Exactly five additional configurations; M1 recovered with preserved extra epochs; remaining four obey in-loop patience; all commit hashes and final optimizer/RNG states verify |
| All twelve checkpoint identities | PASS | Strict architecture loads, finite weights, four-class configs and frozen hashes; M3 original unchanged |
| Enhanced LR-only control | PASS | Only identity labels and prescribed LR fields differ; identical student/teacher, augmentation log hashes per common epoch, dataset, losses and gates |
| Independent numerical and aggregation reconciliation | PASS | All48 metric rows and192 class rows recomputed; exact frozen IDs/capture averaging; confusion counts reconcile; same1421ISIC/205PAD declaredlesions across all checkpoints |
| Figure/data integrity | PASS | 103 figure families decode with 300DPI PNG, parseable SVG and structurally valid PDF cross-references; all source data hashes match |
| Grad-CAM provenance | PASS | New frozen-model development-only predicted-logit gradients verified; raw inputs/maps/overlays preserved; M3 and E1 figures copied unchanged |

No original qualification was changed to pass. M1/M2/M3/M4 were not retrained during amended completion. Exact seeds/splits/gates/LR control preserved. No test tuning, Flutter changes or final export. See13_Reports/FINAL_INTEGRITY_AUDIT.json andTEACHER_PROTOCOL_AMENDMENT.md.
