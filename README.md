# 🩺 MobileSkinXAI

**An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening**

MobileSkinXAI is a thesis-oriented Medical AI project focused on building a lightweight, explainable, and mobile-deployable skin lesion classification system.

The project aims to combine **deep learning**, **cross-dataset validation**, **Explainable AI (XAI)**, and **Flutter-based mobile deployment** to support research in trustworthy and accessible healthcare AI.

---

## 🎯 Project Goal

The goal of this project is to develop a lightweight AI framework that can classify skin lesion images, explain model decisions using visual XAI techniques, and run efficiently on mobile devices.

This project is being developed as an undergraduate thesis with the goal of preparing a publication-quality research manuscript.

---

## 🧠 Research Focus

* Skin lesion classification
* Cross-dataset generalization
* Lightweight deep learning
* Explainable AI using Grad-CAM
* TensorFlow Lite model conversion
* Flutter-based mobile deployment
* Medical image analysis

---

## 🗂️ Datasets

This project uses two public skin lesion datasets:

| Dataset         | Purpose                                                      |
| --------------- | ------------------------------------------------------------ |
| **ISIC 2019**   | Primary training, validation, and testing dataset            |
| **PAD-UFES-20** | External validation dataset for cross-dataset generalization |

> Dataset files are not included in this repository due to size and licensing considerations.

Expected local dataset structure:

```text
dataset/
├── ISIC2019/
│   ├── raw/
│   │   ├── images/
│   │   └── metadata/
│   ├── processed/
│   └── splits/
│
└── PADUFES20/
    ├── raw/
    │   ├── images/
    │   └── metadata/
    ├── processed/
    └── splits/
```

---

## 🏗️ Repository Structure

```text
MobileSkinXAI/
│
├── app/
│   └── flutter/
│
├── config/
│
├── dataset/
│   ├── ISIC2019/
│   └── PADUFES20/
│
├── docs/
│
├── evaluation/
│
├── experiments/
│   ├── exp01_baseline/
│   ├── exp02_preprocessing/
│   ├── exp03_attention/
│   └── exp04_tflite/
│
├── figures/
├── logs/
├── models/
├── notebooks/
├── paper/
├── preprocessing/
├── results/
├── thesis/
├── tflite/
├── training/
├── xai/
│
├── Research_Plan.md
├── requirements.txt
└── README.md
```

---

## 🔬 Proposed Methodology

```text
ISIC 2019 Dataset
        ↓
Data Cleaning and Preprocessing
        ↓
Class Distribution Analysis
        ↓
Train / Validation / Test Split
        ↓
Lightweight CNN Model Training
        ↓
Model Evaluation
        ↓
Grad-CAM Explainability
        ↓
External Validation on PAD-UFES-20
        ↓
TensorFlow Lite Conversion
        ↓
Flutter Mobile App Deployment
```

---

## 🤖 Planned Models

The project will compare multiple lightweight and mobile-friendly architectures:

* MobileNetV2
* MobileNetV3-Small
* MobileNetV3-Large
* EfficientNet-Lite
* EfficientNetV2-B0

The final model will be selected based on:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Model size
* Inference time
* Mobile deployment suitability

---

## 📊 Evaluation Plan

The model will be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve
* PR Curve
* AUC Score
* Cross-dataset performance
* Mobile inference latency
* Model size after TensorFlow Lite conversion

---

## 🔥 Explainable AI

MobileSkinXAI will include explainability through:

* Grad-CAM
* Grad-CAM visualization overlays
* Prediction confidence interpretation
* Class-wise explanation examples

The goal is to make the model prediction more transparent and interpretable.

---

## 📱 Mobile Deployment

The selected model will be converted into **TensorFlow Lite** format and integrated into a **Flutter mobile application**.

Planned app features:

* Upload image from gallery
* Capture image using camera
* Skin lesion prediction
* Confidence score
* Grad-CAM explanation
* Class description
* Medical disclaimer
* Offline inference support

---

## 🗓️ Two-Month Thesis Timeline

| Week   | Task                                                         |
| ------ | ------------------------------------------------------------ |
| Week 1 | Research design, dataset setup, literature review            |
| Week 2 | Dataset cleaning, preprocessing, class distribution analysis |
| Week 3 | Baseline model training and evaluation                       |
| Week 4 | Lightweight model comparison                                 |
| Week 5 | Explainable AI implementation and external validation        |
| Week 6 | TensorFlow Lite conversion and Flutter app integration       |
| Week 7 | Thesis writing, figures, tables, and result analysis         |
| Week 8 | Final thesis polishing and journal paper draft preparation   |

---

## 📌 Expected Contributions

* A lightweight skin lesion classification model suitable for mobile deployment
* Cross-dataset evaluation using ISIC 2019 and PAD-UFES-20
* Explainable AI visualizations for model transparency
* TensorFlow Lite model conversion
* Flutter-based mobile application prototype
* Thesis and publication-ready experimental documentation

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

It is **not a certified medical device** and must not be used for clinical diagnosis, treatment, or medical decision-making.

Always consult qualified healthcare professionals for medical advice.

---

## 👩‍💻 Developer

**Rinvi Jaman Riti**
Computer Science & Engineering
Daffodil International University

Research interests: Medical AI, Computer Vision, Explainable AI, Deep Learning, and Mobile Healthcare AI.
