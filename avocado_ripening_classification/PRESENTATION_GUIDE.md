# 30-Minute Project Presentation Guide

## Project overview

This project classifies orange images into four source-dataset labels:
`blackspot`, `canker`, `fresh`, and `grenning` (the spelling used in the
dataset). Despite the folder name, this is **orange-condition classification**,
not avocado-ripeness classification.

The project compares two approaches:

1. A classical machine-learning pipeline using color and texture features with
   a Support Vector Machine (SVM).
2. A deep-learning pipeline using transfer learning with MobileNetV2.

The objective is to compare their performance on the same held-out test set
and discuss the limitations of that evaluation.

## 30-minute agenda

| Time | Topic | Presentation focus |
|---|---|---|
| 0–2 min | Introduction | State the problem, four labels, and the two approaches. |
| 2–4 min | Purpose | Explain the potential fruit-sorting use case and clarify this is a prototype. |
| 4–7 min | Dataset | Show the folder structure, class counts, and original train/test split. |
| 7–9 min | Exploratory analysis | Discuss class balance, image readability, image dimensions, and lighting checks. |
| 9–11 min | Data preparation | Explain training, validation, and test separation. |
| 11–16 min | Classical model | Describe color histograms, LBP texture features, and the SVM. |
| 16–21 min | Deep-learning model | Explain MobileNetV2 transfer learning, preprocessing, augmentation, and training. |
| 21–24 min | Results | Compare metrics and interpret the confusion matrices. |
| 24–27 min | Demonstration | Run one image through both saved models. |
| 27–29 min | Limitations and future work | Discuss uncertainty, potential shortcuts, and improvements. |
| 29–30 min | Conclusion | Recap results and the main takeaway. |

## Detailed talk track

### 1. Introduction and objective — 0–4 minutes

Suggested opening:

> This project uses computer vision to classify images of oranges into four
> labels: blackspot, canker, fresh, and grenning. I implemented and compared a
> classical machine-learning approach and a deep-learning approach. The goal
> is to evaluate both on the same held-out test data and understand their
> strengths and limitations.

Be explicit that the task is orange-condition classification. The folder name
is misleading; the project does not classify avocado ripeness.

### 2. Dataset and exploratory analysis — 4–9 minutes

The dataset contains 1,090 images:

| Class | Training images | Test images | Total |
|---|---:|---:|---:|
| `blackspot` | 184 | 22 | 206 |
| `canker` | 179 | 22 | 201 |
| `fresh` | 281 | 33 | 314 |
| `grenning` | 347 | 22 | 369 |
| **Total** | **991** | **99** | **1,090** |

The exploratory data analysis checks class counts, image dimensions, whether
files can be read, and grayscale brightness as an indicator of lighting
variation. See
[`notebooks/01_eda_and_data_cleaning.ipynb`](notebooks/01_eda_and_data_cleaning.ipynb).

One important observation is that `blackspot`, `canker`, and `grenning`
images are 800×800 JPGs, while `fresh` images are variable-size PNGs. This
format and resolution difference could act as a dataset shortcut: a model
might learn to identify image characteristics rather than fruit condition.
This makes high test scores less certain as a measure of real-world
performance.

### 3. Data split — 9–11 minutes

The supplied test set is reserved for final evaluation. A stratified
validation set is drawn from the 991 training images, resulting in:

- 842 training images
- 149 validation images
- 99 test images

“Stratified” means the validation split preserves approximately the same
proportions of the different classes. Validation data supports development
decisions; the test set is held back for the final comparison.

The split is implemented in
[`src/dataset.py`](src/dataset.py), in `get_dataset_splits`.

### 4. Classical machine-learning approach — 11–16 minutes

The classical pipeline extracts hand-designed features from each image and
uses them to classify it:

1. Resize the image to 128×128.
2. Calculate color histograms from the HSV and LAB color spaces.
3. Calculate a Local Binary Pattern (LBP) histogram to represent texture.
4. Scale the features and pass them to an RBF-kernel Support Vector Machine.
5. Apply class weighting to help account for imbalanced class counts.

Suggested explanation:

> The color histograms summarize how much of different colors an image
> contains. The LBP feature summarizes local texture patterns. The SVM learns
> how combinations of those features correspond to the four labels.

The feature extraction and classifier are in
[`src/classical_model.py`](src/classical_model.py), and training is in
[`notebooks/02_approach_1_classical_ml.ipynb`](notebooks/02_approach_1_classical_ml.ipynb).

### 5. Deep-learning approach — 16–21 minutes

The second approach uses MobileNetV2, a convolutional neural network
architecture. It starts with ImageNet-pretrained weights, freezes the feature
extraction layers, and replaces the final classification layer to predict the
four project labels.

Images are resized to 224×224, converted to tensors, and normalized using
ImageNet means and standard deviations. Training uses augmentations such as
horizontal flips, small rotations, and mild brightness or contrast changes.
The report records 12 training epochs, with the lowest validation loss at
epoch 12.

Suggested explanation:

> Instead of learning every visual feature from scratch, the model reuses
> features learned from a large image dataset and trains a new final layer for
> these four orange classes.

The model definition is in
[`src/dl_model.py`](src/dl_model.py), image loading is in
[`src/torch_dataset.py`](src/torch_dataset.py), and training is in
[`notebooks/03_approach_2_deep_learning.ipynb`](notebooks/03_approach_2_deep_learning.ipynb).

### 6. Evaluation and results — 21–24 minutes

Both models were evaluated on the same 99 test images:

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| HSV/LAB + LBP + SVM | **96.97%** | **97.00%** | **96.59%** | **96.57%** |
| MobileNetV2 | 94.95% | 94.61% | 94.32% | 94.45% |

On this split, the SVM classified 96 of 99 images correctly. Its three errors
were `canker` images predicted as `blackspot`. MobileNetV2 classified 94 of
99 correctly: it confused some `blackspot` and `canker` images and predicted
one `canker` image as `fresh`.

Explain the metrics briefly:

- **Accuracy:** the proportion of all test images classified correctly.
- **Precision:** when the model predicts a class, how often that prediction
  is correct.
- **Recall:** how many actual examples of a class the model correctly finds.
- **Macro F1:** the average of the F1-scores across classes, giving each class
  equal weight.

Macro metrics are useful because the class counts differ. Results and
confusion matrices are generated by
[`notebooks/04_model_evaluation_and_demo.ipynb`](notebooks/04_model_evaluation_and_demo.ipynb)
and saved in [`reports/test_metrics.json`](reports/test_metrics.json).

Interpret the comparison cautiously: it does not prove the SVM will outperform
MobileNetV2 on new data. The test set is small, and the image-format
difference for `fresh` could influence results.

### 7. Live demonstration — 24–27 minutes

From the project directory, run the demo on a test image:

```powershell
py main_demo.py "data/raw/test/fresh/f (1).png"
```

The script loads both saved models, preprocesses the image for each approach,
and prints each model's prediction. See [`main_demo.py`](main_demo.py).

For a smooth recording:

- Open the project and demo image before starting the video.
- Use an image from the existing test set.
- Avoid retraining during the presentation; show predictions from the saved
  models.
- Explain that the output is a predicted class label, not a quality guarantee.

### 8. Limitations and future work — 27–29 minutes

- There are only 22–33 test images per class, so metric estimates are
  uncertain.
- The class distribution is imbalanced.
- Different image formats and sizes may provide shortcuts.
- Camera, lighting, background, and framing may change performance.
- Production performance on external images has not been established.
- Image classification cannot determine internal fruit quality.

With more time, collect a larger and more balanced dataset from different
sources, verify labels, test on an independent external dataset, and measure
performance on the intended hardware.

The MobileNetV2 timing in the report is about 20.17 ms mean per image for
inference on a preprocessed tensor. It excludes image decoding and resizing,
so it must not be presented as end-to-end production latency.

### 9. Conclusion — 29–30 minutes

Suggested closing:

> The project builds two approaches to orange-condition classification and
> evaluates them on the same held-out test set. The classical SVM scored
> higher on this particular split, while MobileNetV2 provides a
> transfer-learning comparison. Because the test set is small and may contain
> format-related shortcuts, the results are promising for this dataset but
> need broader validation before real-world use.

## Likely questions

**Why did the classical model outperform MobileNetV2?**

For this small dataset, the hand-designed color and texture features may be
enough to separate the classes. The result is specific to this split; more
external data is needed to know which approach generalizes better.

**Why keep a test set separate from validation?**

Validation data helps guide development decisions. A separate test set gives
a more independent final check, provided it is not repeatedly used to tune
the models.

**What does transfer learning mean here?**

The model starts with image features learned from ImageNet, freezes its
feature-extraction backbone, and trains a new final classifier for the four
orange labels.

**Is the model ready for production?**

No production claim is supported yet. It needs evaluation on larger,
independent data and on the target hardware and image-capture conditions.

**What is the main takeaway?**

Both approaches performed well on the supplied test set, but the dataset is
small and may contain an image-format shortcut. Treat the scores as
preliminary rather than proof of general performance.
