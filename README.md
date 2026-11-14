# Diabetic Retinopathy Severity Grading Using DenseNet121

**Research area:** Deep Learning · Medical Image Analysis · Ordinal Image Classification  
**Dataset:** DDR retinal fundus image dataset  
**Framework:** TensorFlow / Keras  
**Status:** Two exploratory experiments documented; further methodological validation is required.

## Overview

This repository documents two experiments investigating five-level diabetic retinopathy (DR) severity grading from retinal fundus images using a DenseNet121-based convolutional neural network.

The first experiment used a class-balanced subset and applied augmentation before splitting the augmented images. It reported 95.91% test accuracy, but the split procedure creates a risk that augmented versions of the same original image occur in different partitions. Its test metrics are therefore reported as historical experimental results, not as a reliable estimate of generalization.

The second experiment used all 12,522 available image records and split the original image records into training, validation, and test partitions before applying augmentation to training batches. This is a better data-splitting design. However, the recorded run is preliminary: the saved best validation quadratic weighted kappa (QWK) was 0.3364, DenseNet-specific input preprocessing was not applied, calculated class weights were not used, and the held-out test set was not evaluated in the saved run.

The repository preserves both experiments to make the research progression and limitations transparent.

## Research objectives

- Investigate transfer learning with DenseNet121 for five-grade DR severity classification.
- Explore an ordinal target representation for the ordered severity labels.
- Compare a balanced-subset experiment with a full-dataset experiment.
- Monitor quadratic weighted kappa (QWK), which accounts for the ordinal relationship between severity grades.
- Document the data preparation, training configurations, recorded results, and methodological limitations.

## Dataset

The notebooks use the DDR dataset mounted in the Kaggle environment. The dataset CSV contains two fields:

| Field | Meaning |
|---|---|
| `id_code` | Retinal image filename |
| `diagnosis` | DR severity grade, encoded from 0 to 4 |

### Severity classes

| Grade | Class |
|---:|---|
| 0 | No DR |
| 1 | Mild NPDR |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | Proliferative DR |

### Full dataset inspected in Experiment 2

The notebook records 12,522 rows, zero duplicate image IDs, and zero missing image files.

| Grade | Class | Images | Percentage |
|---:|---|---:|---:|
| 0 | No DR | 6,266 | 50.04% |
| 1 | Mild NPDR | 630 | 5.03% |
| 2 | Moderate NPDR | 4,477 | 35.75% |
| 3 | Severe NPDR | 236 | 1.88% |
| 4 | Proliferative DR | 913 | 7.29% |
| **Total** | | **12,522** | **100%** |

The class distribution is imbalanced, with grades 1 and 3 particularly underrepresented.

**Dataset access:** Obtain the DDR dataset from its authorized distribution source. The dataset itself is not included in this repository. Check and follow the dataset's terms before using or redistributing it.

## Experiment 1 — Balanced subset

**Notebook:** `notebooks/01_balanced_subset_experiment.ipynb`  
**Historical notebook filename:** `diabetic-retinopathy-densenet121.ipynb`

### Data preparation

- Selected 236 original images from each of the five grades.
- Total original subset: 1,180 images.
- Applied zoom, horizontal-flip, and vertical-flip augmentation.
- The resulting augmented dataset contained 6,594 images.

| Grade | Original images |
|---:|---:|
| 0 | 236 |
| 1 | 236 |
| 2 | 236 |
| 3 | 236 |
| 4 | 236 |
| **Total** | **1,180** |

### Model and training

The notebook builds a DenseNet121 feature extractor with global average pooling, dropout, and a five-unit sigmoid output layer. It uses binary cross-entropy and Adam with a learning rate of 0.00005. Training was run for nine epochs with batch size 32. The notebook loads pretrained DenseNet121 weights without the top classification layer.

### Recorded validation and test results

The highest recorded validation QWK was **0.9966** at epoch 8.

The notebook reports the following metrics on 1,319 test images:

| Metric | Recorded value |
|---|---:|
| Accuracy | 95.91% |
| Macro precision | 96.24% |
| Micro precision | 95.91% |
| Weighted precision | 96.25% |
| Macro recall | 95.95% |
| Micro recall | 95.91% |
| Weighted recall | 95.91% |
| Macro F1-score | 95.95% |
| Micro F1-score | 95.91% |
| Weighted F1-score | 95.93% |

These values are reproduced from the notebook's saved outputs. They should not be interpreted as an unbiased estimate of performance on independent images.

### Important limitation: augmentation before splitting

In this experiment, augmentation was performed on the selected original images before the augmented dataset was divided into training, validation, and test partitions. Consequently, augmented variants derived from the same original image could be distributed across multiple partitions. This creates image-level data leakage risk and can inflate validation and test performance.

The 95.91% accuracy and 0.9966 validation QWK are therefore retained as historical results for this experiment, with this limitation clearly stated. They should not be presented as definitive model performance.

## Experiment 2 — Full dataset with split before augmentation

**Notebook:** `notebooks/02_full_dataset_experiment.ipynb`  
**Historical notebook filename:** `ddr-fyp-full-dataset.ipynb` (the uploaded combined notebook contains both experiments).

### Data split

The original 12,522 records were split using stratification by diagnosis and `random_state=2019`. Image-ID overlap assertions passed for the three partitions.

| Partition | Images | Share |
|---|---:|---:|
| Training | 10,017 | 80% |
| Validation | 1,252 | 10% |
| Test | 1,253 | 10% |
| **Total** | **12,522** | **100%** |

| Grade | Training | Validation | Test |
|---:|---:|---:|---:|
| 0 | 5,013 | 626 | 627 |
| 1 | 504 | 63 | 63 |
| 2 | 3,581 | 448 | 448 |
| 3 | 189 | 24 | 23 |
| 4 | 730 | 91 | 92 |

Images were resized to 224 × 224 RGB. Augmentation was applied through a generator using the training partition.

### Model architecture and configuration

| Component | Configuration |
|---|---|
| Backbone | Pretrained DenseNet121, `include_top=False` |
| Backbone trainability | Frozen |
| Input shape | 224 × 224 × 3 |
| Head | Global Average Pooling |
| Regularization | Dropout 0.5 |
| Output | Dense layer with 5 sigmoid units |
| Loss | Binary cross-entropy |
| Optimizer | Adam |
| Learning rate | 0.00005 |
| Batch size | 32 |
| Epochs | 9 |
| Total parameters | 7,042,629 |
| Trainable parameters | 5,125 |
| Non-trainable parameters | 7,037,504 |

The notebook saves a latest checkpoint, a best-validation-QWK checkpoint, and a CSV training log.

### Recorded training result

| Metric | Recorded value |
|---|---:|
| Best validation QWK | **0.3364** |
| Epoch of best validation QWK | 9 |
| Final training loss | 0.7793 |
| Final validation loss | 0.4388 |
| Keras final training `accuracy` | 0.9976 |
| Keras final validation `val_accuracy` | 1.0000 |

**Metric interpretation:** The model has five sigmoid outputs representing ordinal targets. The Keras `accuracy` metric in this notebook is computed over those output values; it is not equivalent to five-class grade accuracy. It must not be presented as 99.76% training grade accuracy or 100% validation grade accuracy.

No final test-set accuracy, precision, recall, F1-score, confusion matrix, or test QWK is recorded for Experiment 2. The test partition was created, but the saved training run does not contain a completed held-out test evaluation.

### Important limitations and audit notes

1. **DenseNet preprocessing:** The notebook loads RGB image arrays with values in the 0–255 range but does not apply the preprocessing transformation associated with the selected pretrained DenseNet121 weights.
2. **Class imbalance:** Balanced class weights are calculated in the notebook, but they are not applied in `model.fit` or in a custom weighted loss.
3. **Ordinal encoding output:** The current source code in the ordinal-label cell uses `i <= grade`, and the following assertion cell expects the corresponding five cumulative vectors. However, the saved output shown in the notebook still contains the earlier encoding values. The source code and saved output are therefore inconsistent; the saved assertion output is not proof that the edited source was rerun.
4. **Patient-level independence:** The CSV contains image filenames and diagnosis labels but no patient identifiers. The split is stratified at image level; patient-level separation cannot be verified.
5. **Test evaluation:** No final evaluation on the held-out test partition is recorded.

Experiment 2 improves the split design by separating original image records before training augmentation. It remains a preliminary run and is not yet a validated final model.

## Methodology summary

The experiments follow this broad workflow:

1. Load image identifiers and diagnosis labels from the dataset CSV.
2. Prepare images as RGB inputs at 224 × 224 pixels.
3. Encode the five ordered grades using cumulative ordinal targets.
4. Apply data augmentation to the training data.
5. Extract visual features using pretrained DenseNet121.
6. Apply global average pooling and dropout.
7. Predict five sigmoid outputs.
8. Train with binary cross-entropy and Adam.
9. Monitor validation QWK and save model checkpoints.
10. Evaluate the selected model on a held-out test set.

The two experiments differ in dataset selection and the timing of augmentation relative to data splitting. Their results should be interpreted separately.

## Repository structure

The current GitHub repository keeps the main project files at the root level:

```text
diabetic-retinopathy-detection/
│
├── README.md
├── ddr-fyp-full-dataset.ipynb
└── diabetic-retinopathy-densenet121.ipynb
```

- `diabetic-retinopathy-densenet121.ipynb` — original balanced-subset experiment.
- `ddr-fyp-full-dataset.ipynb` — newer notebook containing the full-dataset experiment and the earlier experiment cells.

This simple structure is appropriate for the current project. Separate `results/`, `models/`, or `docs/` folders can be added later if you decide to publish additional plots, reports, or model artifacts. Do not create empty folders or include files that are not actually present.

### Model files

The notebooks save Keras model files. The model artifacts may be kept in local backups or attached to a GitHub Release. If committing large model files, check GitHub's file-size limits and consider Git LFS. The repository does not need to include duplicate model checkpoints.

### Dataset files

Do not commit the full DDR dataset or extracted retinal images to this repository. Provide instructions for obtaining the dataset from its authorized source instead.

## Reproducibility

The notebooks were developed in Kaggle and use Kaggle-specific input paths for the dataset and pretrained DenseNet121 weights. Those paths must be adapted before running elsewhere.

The main libraries used include:

- Python
- TensorFlow / Keras
- NumPy
- pandas
- Pillow
- OpenCV
- scikit-learn
- Matplotlib
- tqdm
- mlxtend (used for confusion-matrix visualization in Experiment 1)

The notebooks set NumPy and TensorFlow random seeds to `2019` in the original experiment; the full-dataset split and training generator also use `random_state` / `seed` values of `2019`. Exact reproducibility can still depend on package versions, hardware, and GPU execution.

A pinned `requirements.txt` should be generated from the actual Kaggle environment before claiming environment-level reproducibility. Package versions are not fully specified in the notebooks.

## Recommended next research steps

Before using the work as evidence of model performance in an academic application, thesis, or research proposal:

1. Correct the full-dataset pipeline to use the appropriate DenseNet121 preprocessing for the selected pretrained weights.
2. Re-run the ordinal encoding verification so the saved outputs match the current source code.
3. Choose and implement a suitable strategy for class imbalance; do not describe the current calculated weights as applied.
4. Use patient-level splitting if patient identifiers can be obtained. If they cannot, state the image-level split limitation.
5. Train and select the model using training and validation data only.
6. Evaluate the selected model once on the untouched test set.
7. Report class-wise precision, recall, F1-score, confusion matrix, accuracy, and QWK.
8. Record package versions, exact pretrained-weight source, training configuration, and checkpoint details.
9. Compare against a baseline and, where feasible, report uncertainty estimates.

## Responsible-use statement

This repository is an academic research project. It is not a medical device and does not provide a clinical diagnosis. The models and results must not be used to make patient-care decisions. Any clinical application would require rigorous external validation, clinical oversight, representative data, and applicable regulatory review.

## Dataset and model citations

The supplied notebooks identify the DDR dataset and a Kaggle-hosted DenseNet121 pretrained-weight file, but do not provide complete bibliographic references. Add the original DDR dataset publication/provider citation and the exact pretrained-weight source citation before formal academic submission. Do not invent bibliographic details.

---

**Project status:** Two exploratory experiments documented. Experiment 1 has augmentation-before-split leakage risk. Experiment 2 has an improved image-level split design but requires preprocessing correction, ordinal-output verification, imbalance handling, and independent test evaluation before final performance claims can be made.
