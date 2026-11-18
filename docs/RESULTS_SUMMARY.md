# Frozen results summary

Primary scoring averages raw softmax probabilities over predeclared exact-capture representatives within each declared lesion, then uses argmax in NV/MEL/BCC/SCC order. ISIC1421declaredlesions/2703images; fixed PAD205declaredlesions/261images. Patient grouping limitations and the ambiguous validation-only SCC identity are preserved; known lesion independence is not biological patient independence when identifiers are unavailable.

|Model|E1 ISIC macro-F1|E1 fixed PAD macro-F1|E2 ISIC macro-F1|E2 fixed PAD macro-F1|
|---|---:|---:|---:|---:|
|MobileNetV3-Small|0.6541|0.3839|0.6380|0.6762|
|EfficientNet-Lite0|0.6583|0.3214|0.6403|0.7014|
|MobileNetV4-Conv-Small|0.6853|0.4032|0.6121|0.7100|
|ConvNeXt-Tiny|0.7168|0.3642|0.6813|0.6936|
|MobileSkinXAI-Enhanced|0.6927|0.4239|0.6791|0.6722|
|MobileSkinXAI-Enhanced (LR Control)|0.7103|0.4232|0.6639|0.6965|


The selected LR Control's E2 PAD accuracy is81.4634%, macro-F1 is0.6964749025 and balanced accuracy is64.4949%. MEL recall4/7(57.14%); SCC recall7/28(25.00%). Its ISIC accuracy is83.9550%, macro-F1 is0.6638767283 and balanced accuracy is65.7348%. Its PAD macro-F1 change versus its own E1 is+27.3315percentagepoints. E2 has labeled PAD exposure, so this is adaptation performance, not independent zero-shot confirmation.

MobileNetV4-Conv-Small has the highest E2 PAD macro-F1 point estimate0.7100417745, withMEL5/7 andSCC6/28. It was NOT retrospectively substituted for the development-selected candidate. Small melanoma support, poor SCC recognition, one seed in the E1/E2 study, previously examined PAD, unequal parent-training budgets and the failed-teacher amendment limit interpretation. Multi-seed CA-MACL evidence is a separate source-development study, not a controlled E2teacher ablation or proof of external superiority.

[All four-setting tables](../evaluation/metrics_tables/Table_05_Complete_Four_Setting_Classification_Comparison.md), [class-wise precision/recall/F1](../evaluation/metrics_tables/Table_07_PAD_Classwise_Precision_Recall_and_F1.md), [exact MEL/SCC counts](../evaluation/metrics_tables/Table_08_Melanoma_and_SCC_Exact_Detections.md), [model efficiency](../evaluation/metrics_tables/Table_09_Model_Efficiency_Training_and_Selected_Checkpoints.md), [figure index](../evaluation/FIGURE_INDEX.csv).

The full aggregate metrics/CMcells/classwise specificity/AUC/AP are in evaluation/source_data. Counts reconcile independently with48metric rows. Conditional patient-grouped bootstrap intervals are reported from original outputs, but no patient/group records are public; image bootstrap is not a substitute for training-seed replication. Historical full-PAD scores are inTable02and separately labeled comparative figures; only matched fixed-subset E1/E2 differences support the adaptation comparison.

Original model epochs, histories, measured runtimes and the MobileNetV3 controller recovery are retained. No new research training/test evaluation or score tuning was performed for publication. Methodological novelty, clinical safety and fresh independent smartphone superiority remain unproven.
