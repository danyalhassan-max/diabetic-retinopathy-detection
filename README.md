# Diabetic Retinopathy Severity Grading Using DenseNet121

**Research Area:** Deep Learning · Medical Image Analysis · Ordinal Image Classification

**Dataset:** DDR Retinal Fundus Image Dataset

**Framework:** TensorFlow / Keras

**Project Status:** Advanced Multi-Stage Empirical Evaluation & Methodological Auditing

---

## 1. Executive Summary & Research Motivation

Diabetic Retinopathy (DR) remains a leading global cause of preventable blindness among working-age populations. Automated, computer-aided screening systems using retinal fundus images offer a viable pathway to expand diagnostic reach, reduce clinical backlogs, and ensure timely intervention.

This repository establishes an end-to-end, reproducible deep learning framework designed to grade DR severity across a standardized five-level ordinal scale (Grades 0 to 4) using the public **DDR dataset**.

However, building clinically and academically rigorous diagnostic models requires addressing severe methodological pitfalls commonly overlooked in baseline literature—specifically **data leakage caused by naive data augmentation splits** and **severe class imbalance**. This project systematically documents an iterative research progression: transitioning from a flawed preliminary setup with inflated metrics to a robust, leakage-free pipeline incorporating rigorous stratified partitioning, computed class-weighted loss functions, and multi-stage fine-tuning architectures.

---

## 2. Comprehensive Research Progression & Iterative Experiments

To maintain complete transparency and adhere to strict academic standards, this repository chronicles three distinct developmental phases. Each phase represents a deliberate effort to isolate methodological flaws, enforce data hygiene, and maximize generalizability.

### Phase 1: Experiment 1 — The Balanced Subset Baseline & Leakage Discovery

* **Objective:** Establish a quick proof-of-concept for five-grade classification using DenseNet121 on a controlled subset of the DDR dataset.
* **Methodology:**
* Selected 236 original images per severity grade, totaling 1,180 images.
* Applied spatial transformations (zoom, horizontal flips, vertical flips) to construct an augmented pool of 6,594 images.


* **The Flaw (Data Leakage):** Augmentation was executed *prior* to splitting the dataset into training, validation, and test sets. Consequently, distinct augmented variations originating from the same physical patient image leaked across partition boundaries.
* **Outcome & Critical Lesson:** The model achieved an artificially inflated test accuracy of **95.91%** and a validation Quadratic Weighted Kappa (QWK) of **0.9966**. While historically preserved for auditing purposes, this experiment serves as a textbook example of data leakage. It proves that naive splitting invalidates test metrics, reinforcing the principle that data hygiene supersedes raw score optimization.

---

### Phase 2: Experiment 2 — Full-Dataset Scale & Rigorous Partitioning

* **Objective:** Eliminate data leakage by utilizing the entire scale of the DDR dataset (12,522 image records) and enforcing strict partitioning prior to augmentation.
* **Methodology:**
* Implemented an 80/10/10 stratified split (`random_state=2019`) across all 12,522 images prior to batch generator loading.
* Maintained a strict partition barrier ensuring zero image-ID overlap between Training (10,017 images), Validation (1,252 images), and Test (1,253 images) splits.


* **Identified Limitations:** While the partition design was structurally sound, the preliminary run suffered from implementation oversights: DenseNet-specific pixel preprocessing was omitted, computed class weights were omitted from `model.fit()`, and final held-out test evaluations were omitted, resulting in a conservative baseline validation QWK of **0.3364**.

---

### Phase 3: Experiment 3 — Production-Ready Fine-Tuning & Advanced Backbones

* **Objective:** Resolve all preprocessing omissions, handle extreme class imbalance via algorithmic cost-sensitive learning, and maximize generalization using modern vision backbones.
* **Methodology:**
* **Zero Data Leakage:** Maintained strict pre-split partitioning with training-only data augmentation.
* **Cost-Sensitive Learning:** Integrated mathematically computed balanced class weights directly into the training loss calculation to counteract the severe underrepresentation of Mild (Grade 1: 5.03%) and Severe (Grade 3: 1.88%) categories.
* **Multi-Stage Fine-Tuning:**
* *Stage 1:* Trained classifier heads with frozen feature extraction backbones.
* *Stage 2:* Fully unfrozen end-to-end fine-tuning using micro-learning rates ($1\times 10^{-5}$) to adapt pre-trained weights to subtle retinal pathologies (microaneurysms, exudates, and hemorrhages).


* **Backbone Evolution:** Evaluated robust architectures including DenseNet121 and an elite **EfficientNetB3** configuration operating at an upgraded $300 \times 300$ resolution.


* **Final Performance Metrics:** The final production pipeline yielded an exceptional test **Quadratic Weighted Kappa (QWK) of 0.7532**, an overall test accuracy of **65%**, and high clinical sensitivity on minority classes (e.g., a **0.70 recall** on Severe DR cases).

---

## 3. Dataset Architecture & Class Distribution

The project utilizes the **DDR dataset**, mapped via a structured CSV file linking image identifiers to ordinal diagnostic targets.

### Severity Grading Scale

* **Grade 0:** No DR
* **Grade 1:** Mild Non-Proliferative Diabetic Retinopathy (NPDR)
* **Grade 2:** Moderate NPDR
* **Grade 3:** Severe NPDR
* **Grade 4:** Proliferative DR

### Full Dataset Statistical Breakdown (Experiment 2 & 3)

| Grade | Clinical Description | Image Count | Dataset Share |
| --- | --- | --- | --- |
| 0 | No DR | 6,266 | 50.04% |
| 1 | Mild NPDR | 630 | 5.03% |
| 2 | Moderate NPDR | 4,477 | 35.75% |
| 3 | Severe NPDR | 236 | 1.88% |
| 4 | Proliferative DR | 913 | 7.29% |
| **Total** | **Complete Corpus** | **12,522** | **100%** |

---

## 4. Repository Structure

```text
diabetic-retinopathy-detection/
│
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_balanced_subset_experiment.ipynb
│   ├── 02_full_dataset_experiment.ipynb
│   └── 03_efficientnet_production_pipeline.ipynb
└── assets/
    ├── confusion_matrix.png
    └── training_history_curves.png

```

---

## 5. Technical Stack & Reproducibility

Developed and tested under GPU-accelerated cloud environments (Kaggle Kernels), ensuring scalable execution for heavy computer vision tasks.

* **Core Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Numerical & Data Processing:** NumPy, pandas
* **Image Processing & Computer Vision:** OpenCV, Pillow, scikit-learn
* **Visualization:** Matplotlib, Seaborn, mlxtend
* **Reproducibility Controls:** Random seed fixed globally (`seed=2019`) across NumPy, TensorFlow, and partition splitting states.

---

## 6. Academic & Professional Utility

This project serves as an ideal centerpiece for:

1. **Master's Degree Admissions (Europe / International):** Demonstrates graduate-level command over experimental rigor, data hygiene, and handling real-world imbalanced medical datasets.
2. **Research Publication / Thesis Chaperones:** Provides an airtight ablation study demonstrating the exact mathematical and empirical impact of rectifying data leakage and applying cost-sensitive fine-tuning.

---

## 7. Responsible Use Statement

*This repository is structured strictly as an academic research and engineering portfolio project. It does not constitute a certified medical device, clinical diagnostic tool, or professional healthcare recommendation. Clinical deployment mandates extensive external validation, multi-center prospective trials, regulatory clearance (e.g., CE/FDA), and direct medical oversight.*
