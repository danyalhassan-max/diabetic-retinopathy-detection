# Diabetic Retinopathy Severity Grading Using Deep Learning

**Research Area:** Deep Learning · Medical Image Analysis · Computer Vision · Ordinal Classification

**Dataset:** DDR (Deep Diabetic Retinopathy) Retinal Fundus Image Dataset

**Architectures:** DenseNet121 · EfficientNetB3

**Framework:** TensorFlow / Keras

**Project Status:** Multi-Stage Experimental Evaluation & Methodological Auditing

---

## 1. Executive Summary & Research Motivation

Diabetic Retinopathy (DR) is a major cause of preventable vision impairment worldwide. Automated grading systems based on retinal fundus images have the potential to support screening workflows by assisting in the identification and classification of disease severity.

This repository presents an experimental deep learning framework for five-class Diabetic Retinopathy severity grading using the publicly available **DDR (Deep Diabetic Retinopathy) dataset**.

The project investigates convolutional neural network architectures, including **DenseNet121 and EfficientNetB3**, to classify retinal fundus images into five severity levels, ranging from No DR (Grade 0) to Proliferative DR (Grade 4).

The research progression focuses on improving experimental reliability through appropriate dataset partitioning, training-only data augmentation, class-imbalance handling, transfer learning, and systematic model evaluation.

Particular attention is given to identifying methodological weaknesses in preliminary experiments, including augmentation-induced data leakage, preprocessing inconsistencies, class imbalance, and limitations in evaluation methodology.

Rather than presenting only the highest-performing results, this repository preserves the experimental progression, including earlier flawed experiments, their limitations, subsequent methodological improvements, and recorded evaluation results.

The overall objective is to develop a reproducible experimental framework and investigate the effectiveness of deep learning architectures for automated DR severity grading.

---

## 2. Comprehensive Research Progression & Iterative Experiments

To maintain experimental transparency and support reproducibility, this repository documents three developmental phases. Each phase represents a progression in dataset utilization, experimental methodology, model development, and evaluation.

### Phase 1: Experiment 1 — Balanced Subset Baseline & Data Leakage Discovery

**Objective:** Establish an initial proof of concept for five-class DR severity classification using DenseNet121 on a controlled subset of the DDR dataset.

**Methodology:**

- Selected 236 original images per severity grade, resulting in a balanced subset of 1,180 original images.
- Applied spatial transformations, including zoom, horizontal flipping, and vertical flipping.
- Generated an augmented image pool containing 6,594 images.
- Trained a DenseNet121-based classification model on the resulting dataset.

**Identified Methodological Flaw — Data Leakage:**

Data augmentation was performed before partitioning the dataset into training, validation, and test sets.

Consequently, transformed versions originating from the same original retinal image could appear in multiple partitions. This allowed the model to encounter visually related samples during training and evaluation.

This introduced augmentation-induced data leakage and compromised the independence of the evaluation sets.

**Recorded Outcome:**

- Test accuracy: **95.91%**
- Validation QWK: **0.9966**

These unusually high metrics are preserved as historical experimental outputs but are not considered reliable indicators of model generalization because of the identified leakage.

**Research Lesson:**

This experiment demonstrated the importance of splitting original images before applying augmentation. It also established the need for independent evaluation data when developing medical image classification systems.

---

### Phase 2: Experiment 2 — Full-Dataset Training & Improved Partitioning

**Objective:** Move beyond the balanced subset and utilize the complete DDR dataset while addressing the augmentation-induced leakage identified in Experiment 1.

**Methodology:**

- Utilized the full dataset containing 12,522 image records.
- Implemented an 80/10/10 stratified train-validation-test split.
- Used `random_state=2019` for reproducibility.
- Performed dataset partitioning before batch-generator loading and training augmentation.
- Maintained separate training, validation, and test partitions.

**Dataset Partitioning:**

| Partition | Images | Percentage |
|---|---:|---:|
| Training | 10,017 | 80% |
| Validation | 1,252 | 10% |
| Testing | 1,253 | 10% |
| **Total** | **12,522** | **100%** |

**Identified Implementation Limitations:**

Although the experiment improved the partitioning methodology, subsequent auditing identified several implementation omissions:

- DenseNet121-specific image preprocessing was not correctly incorporated.
- Computed class weights were not passed into the model training procedure.
- Final held-out test evaluation was omitted.
- The experiment therefore did not provide a complete and consistently evaluated training pipeline.

**Recorded Outcome:**

- Validation QWK: **0.3364**

This result represents the recorded validation performance of the preliminary full-dataset experiment.

**Research Lesson:**

Using the complete dataset and partitioning before augmentation improves the experimental design, but correct model-specific preprocessing, class-imbalance handling, and complete evaluation are also essential.

**Important Limitation:**

The stratified image-level split establishes separation of the assigned image records, but does not independently establish patient-level separation or guarantee the absence of exact and near-duplicate images. These limitations must be considered when interpreting the results.

---

### Phase 3: Experiment 3 — EfficientNetB3-Based Training & Performance Evaluation

**Objective:** Develop an improved full-dataset experimental pipeline using EfficientNetB3, addressing earlier implementation limitations and investigating the performance of a different convolutional neural network architecture.

**Methodology:**

#### 1. Dataset Partitioning and Augmentation

- Utilized the DDR dataset containing 12,522 image records.
- Maintained separate training, validation, and test partitions.
- Applied augmentation during training rather than generating transformed samples before dataset partitioning.
- Used validation data for model selection and the held-out test set for final performance evaluation.

#### 2. Class-Imbalance Handling

The DDR dataset contains a substantial imbalance across DR severity categories, particularly for Mild and Severe DR.

To address this issue:

- Calculated class weights using the training data distribution.
- Incorporated cost-sensitive learning into the training process.
- Evaluated class-wise precision, recall, and F1-score to understand performance across underrepresented categories.

#### 3. Transfer Learning and Fine-Tuning

The experiment uses a two-stage transfer-learning strategy:

**Stage 1 — Feature Extraction**

- Initialized EfficientNetB3 with pretrained ImageNet weights.
- Froze the pretrained feature extraction backbone.
- Trained the classification head using the target DR dataset.

**Stage 2 — Fine-Tuning**

- Unfroze the feature extraction backbone for further adaptation.
- Used a small learning rate of approximately `1 × 10⁻⁵`.
- Fine-tuned the network to learn retinal-image features relevant to DR severity classification.

#### 4. Architecture and Input Resolution

EfficientNetB3 was selected to investigate compound-scaled CNN architecture performance on retinal fundus images.

- Architecture: EfficientNetB3
- Input resolution: `300 × 300`
- Framework: TensorFlow / Keras
- Transfer learning: ImageNet pretrained weights

DenseNet121 is retained in the earlier experimental phases, allowing the repository to document the development and evaluation of different CNN architectures.

#### 5. Recorded Performance

The uploaded EfficientNetB3 experiment contains the following recorded test results:

| Evaluation Metric | Recorded Result |
|---|---:|
| Test Accuracy | Approximately 65% |
| Quadratic Weighted Kappa (QWK) | 0.7532 |
| Macro F1-score | 0.51 |
| Weighted F1-score | 0.66 |
| Severe DR Recall (Grade 3) | 0.70 |

These results represent the saved execution outputs of the experiment and should be reproduced through a fresh execution before being treated as final research results.

**Class-Wise Performance:**

| DR Grade | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Grade 0 — No DR | 0.81 | 0.86 | 0.83 |
| Grade 1 — Mild | 0.14 | 0.49 | 0.22 |
| Grade 2 — Moderate | 0.87 | 0.35 | 0.50 |
| Grade 3 — Severe | 0.22 | 0.70 | 0.33 |
| Grade 4 — Proliferative | 0.61 | 0.77 | 0.68 |

**Performance Interpretation:**

The recorded QWK of 0.7532 indicates promising agreement between predicted and actual DR severity grades, considering the ordinal nature of the classification task.

However, the class-wise results reveal substantial differences in classification performance. In particular, Mild DR has low precision, while Moderate DR has relatively low recall.

These findings highlight the importance of evaluating models beyond overall accuracy and considering the performance of each severity category.

**Research Lesson:**

The third experiment investigates the use of EfficientNetB3, transfer learning, class-imbalance handling, and multi-stage fine-tuning as an alternative approach to the earlier DenseNet121 experiments.

The results should be interpreted in the context of the experimental split, evaluation protocol, and remaining dataset limitations.

---

## 3. Dataset Architecture & Class Distribution

The project utilizes the **DDR (Deep Diabetic Retinopathy) dataset**, a publicly available retinal fundus image dataset designed for research into diabetic retinopathy detection and grading.

The dataset is organized using image identifiers and corresponding diagnostic labels, which are mapped to five ordinal DR severity categories.

### DR Severity Grading Scale

| Grade | Clinical Description |
|---|---|
| 0 | No Diabetic Retinopathy |
| 1 | Mild Non-Proliferative Diabetic Retinopathy (NPDR) |
| 2 | Moderate Non-Proliferative Diabetic Retinopathy (NPDR) |
| 3 | Severe Non-Proliferative Diabetic Retinopathy (NPDR) |
| 4 | Proliferative Diabetic Retinopathy (PDR) |

### Full Dataset Statistical Breakdown

The following distribution represents the 12,522 image records used in the full-dataset experiments.

| Grade | Clinical Description | Image Count | Dataset Share |
|---|---|---:|---:|
| 0 | No DR | 6,266 | 50.04% |
| 1 | Mild NPDR | 630 | 5.03% |
| 2 | Moderate NPDR | 4,477 | 35.75% |
| 3 | Severe NPDR | 236 | 1.88% |
| 4 | Proliferative DR | 913 | 7.29% |
| **Total** | **Complete Dataset** | **12,522** | **100%** |

### Dataset Challenges

The DDR dataset presents several challenges relevant to deep learning research:

- **Class Imbalance:** No DR and Moderate DR account for a substantial proportion of the dataset, while Mild and Severe DR are underrepresented.
- **Visual Complexity:** Retinal lesions can be small, subtle, and difficult to distinguish.
- **Ordinal Relationships:** The five severity grades represent an ordered progression rather than unrelated categories.
- **Generalization:** Models must learn clinically relevant retinal features rather than memorizing image-specific characteristics.

These characteristics make DR grading a challenging medical image classification problem.

---

## 4. Repository Structure

The repository is organized into separate notebooks for the three experimental phases, with supporting assets and configuration files.

```text
diabetic-retinopathy-detection/
│
├── README.md
│   ├── 01_balanced_subset_experiment.ipynb
│   ├── 02_full_dataset_experiment.ipynb
│   └── 03_efficientnet_production_pipeline.ipynb
└── assets/
    ├── confusion_matrix.png
    └── training_history_curves.png
```

### Notebook Descriptions

| Notebook | Description |
|---|---|
| `01_balanced_subset_experiment.ipynb` | Initial DenseNet121 experiment using a balanced subset, including the investigation of augmentation-induced data leakage. |
| `02_full_dataset_experiment.ipynb` | Full-dataset DenseNet121 experiment with stratified dataset partitioning. |
| `03_efficientnet_production_pipeline.ipynb` | EfficientNetB3 experiment using transfer learning, fine-tuning, and performance evaluation. |

### Supporting Assets

- `confusion_matrix.png`: Visual representation of classification performance across DR severity grades.
- `training_history_curves.png`: Training and validation performance throughout model optimization.

---

## 5. Technical Stack & Reproducibility

The experiments were developed using Python and TensorFlow/Keras in GPU-accelerated cloud environments, including Kaggle Notebooks.

### Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| CNN Architectures | DenseNet121, EfficientNetB3 |
| Numerical Computing | NumPy |
| Data Processing | pandas |
| Image Processing | OpenCV, Pillow |
| Machine Learning Utilities | scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Model Evaluation | scikit-learn metrics, Quadratic Weighted Kappa |
| Execution Environment | Kaggle Notebooks |

### Reproducibility Considerations

The experiments incorporate reproducibility practices, including:

- Fixed random seeds where implemented.
- Stratified dataset partitioning.
- Separate training, validation, and test datasets.
- Training-only augmentation in the improved pipeline.
- Pretrained ImageNet model weights.
- Consistent metric-based evaluation.
- Recorded training and validation histories.

**Note:** Reproducibility also depends on the TensorFlow version, GPU environment, dataset version, library dependencies, and availability of pretrained weights. Exact numerical reproduction should be verified by executing the notebooks in a documented environment.

---

## 6. Experimental Evaluation & Research Contributions

This repository documents an iterative experimental investigation into deep learning-based DR severity grading.

### Key Research and Engineering Contributions

1. **Experimental Progression:** Documents the transition from a balanced-subset baseline to full-dataset experiments and alternative CNN architectures.

2. **Data Leakage Investigation:** Identifies augmentation-induced leakage in the initial experiment and demonstrates the importance of partitioning original images before augmentation.

3. **Full-Dataset Utilization:** Extends the experiments to the complete DDR dataset rather than relying exclusively on a balanced subset.

4. **Class-Imbalance Analysis:** Investigates cost-sensitive learning to address underrepresented DR severity categories.

5. **Transfer Learning:** Evaluates pretrained CNN architectures for retinal fundus image feature extraction and classification.

6. **Multi-Stage Fine-Tuning:** Investigates frozen-backbone feature extraction followed by end-to-end fine-tuning.

7. **Comprehensive Evaluation:** Examines accuracy, QWK, precision, recall, and F1-score to understand both overall and class-specific model performance.

### Academic and Professional Applications

This project demonstrates practical experience in:

- Medical image classification.
- Deep learning model development.
- Transfer learning and fine-tuning.
- Imbalanced dataset analysis.
- Experimental design and methodological auditing.
- Evaluation of ordinal classification tasks.
- Reproducible machine learning workflows.

It can serve as a foundation for further academic research, graduate research applications, thesis development, and comparative deep learning studies.

---

## 7. Limitations & Future Research

Despite the experimental improvements, several limitations remain relevant when interpreting the results.

### Dataset Limitations

- The experiments use a single publicly available dataset.
- The dataset's image-level labels do not establish independent patient identities.
- Image-level stratification does not guarantee patient-level separation.
- Exact and near-duplicate image detection should be performed to further investigate possible overlap between partitions.
- External validation on independent retinal datasets is required to evaluate cross-dataset generalization.

### Model Limitations

- Performance varies across DR severity categories.
- Mild and Moderate DR classification remain challenging.
- Overall accuracy does not fully represent performance on minority classes.
- Recorded results require independent reproduction before being considered final research findings.

### Future Research Directions

Potential future improvements include:

- Exact and perceptual near-duplicate image detection.
- Group-aware data partitioning where reliable grouping information is available.
- Additional CNN architecture comparisons.
- Ordinal-aware loss functions and classification methods.
- Hyperparameter optimization.
- Confidence interval estimation and repeated-seed experiments.
- Grad-CAM-based model interpretability.
- External validation using independent retinal fundus datasets.

---

## 8. Responsible Use Statement

This repository is intended strictly for academic research, educational purposes, and experimental development.

The models developed in this project are **not certified medical devices** and must not be used as standalone systems for clinical diagnosis, treatment decisions, or patient management.

Automated DR grading systems require extensive independent validation, assessment of clinical safety, evaluation across diverse patient populations, regulatory approval where applicable, and appropriate medical oversight before any clinical deployment.

The experimental results presented in this repository should not be interpreted as evidence of clinical effectiveness or diagnostic readiness.

---

## 9. Acknowledgements & References

The project uses the publicly available DDR retinal fundus image dataset.

**Dataset Reference:**

Li, T., Gao, Y., Wang, K., Guo, S., Liu, H., & Kang, H. (2019). *Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening.* Information Sciences, 501, 511–522.

**Architecture References:**

- Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* International Conference on Machine Learning (ICML).

---

**Disclaimer:** This project is an independent academic research implementation. All dataset rights, original model architectures, and third-party resources remain subject to their respective licenses and publication terms.
