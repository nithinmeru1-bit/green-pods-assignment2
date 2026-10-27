# Orange Condition Classification

This project adapts the computer-vision assignment workflow to the supplied
orange image dataset. It classifies `blackspot`, `canker`, `fresh`, and
`grenning` (the spelling in the source dataset). These are orange-condition
labels, not avocado ripeness stages.

For a timed walkthrough, see the [30-minute presentation guide](PRESENTATION_GUIDE.md).

## Dataset

The original train/test folders are preserved under `data/raw/`:

```text
data/raw/
├── train/
│   ├── blackspot/
│   ├── canker/
│   ├── fresh/
│   └── grenning/
└── test/
    ├── blackspot/
    ├── canker/
    ├── fresh/
    └── grenning/
```

| Class | Train | Test | Total |
|---|---:|---:|---:|
| `blackspot` | 184 | 22 | 206 |
| `canker` | 179 | 22 | 201 |
| `fresh` | 281 | 33 | 314 |
| `grenning` | 347 | 22 | 369 |
| **Total** | **991** | **99** | **1,090** |

The notebooks make a stratified validation split from the supplied training
folders and reserve the supplied test folders for final evaluation. Images
remain in their original format and are not moved or relabeled during training.
With the current data, this produces 842 training, 149 validation, and 99 test
images (77.2% / 13.7% / 9.1% of the total). The supplied test split is kept
intact instead of being re-partitioned to an artificial 70/15/15 split.

## Setup

Use Python 3.8 or newer:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Place the dataset's existing `train/` and `test/` folders under `data/raw/`,
with one directory per class. For example, copy the contents of the supplied
`Orange Dataset` folder into `data/raw/`. The source dataset is not duplicated
in the code repository; large raw images and trained weights are excluded by
`.gitignore`. Confirm that you have permission to publish any dataset before
sharing it.

Open and run the notebooks in order:

1. `notebooks/01_eda_and_data_cleaning.ipynb` — class balance, image
   dimensions, and lighting checks.
2. `notebooks/02_approach_1_classical_ml.ipynb` — HSV/LAB color histograms,
   Local Binary Patterns, and an SVM.
3. `notebooks/03_approach_2_deep_learning.ipynb` — frozen MobileNetV2
   transfer-learning head, geometric/color augmentation, and validation-based
   early stopping.
4. `notebooks/04_model_evaluation_and_demo.ipynb` — held-out test comparison,
   classification metrics, confusion matrices, and single-image demo.

The deep-learning notebook downloads ImageNet MobileNetV2 weights the first
time it runs. If pretrained weights are unavailable, set `pretrained=False`
in the model-building cell and note that the run is not transfer learning.

The training run used a CPU-only environment. For a fresh VS Code environment,
select the Python environment created above as the notebook kernel. If Jupyter
reports a kernel launch error, install the project requirements into that
environment and restart the kernel.

## Train and run the demo

Run the training cells in notebooks 2 and 3 first; they save model files under
`models/`. Then classify an image from the test split:

```powershell
py main_demo.py "data/raw/test/fresh/f (1).png"
```

Optional model paths can be supplied with `--classical-model` and `--deep-model`.
The script reports predictions from both trained models.

## Deploy the project page to Vercel

The root `index.html` is a static project showcase. It presents the project
and links to the notebooks, results, and report; it does not run model
predictions or require a Python function.

In Vercel, import the repository and configure:

- **Root Directory:** `avocado_ripening_classification`
- **Framework Preset:** `Other`
- **Build Command:** leave blank
- **Output Directory:** leave blank (serve the project root)

Then deploy. If Vercel still reports that it cannot find a Python entrypoint,
check that the Root Directory is the project folder and that the Framework
Preset is not set to Python. The notebook and command-line demo are not
web-server entrypoints.

## Evaluation and dataset limitations

Notebook 4 reports accuracy, macro precision, macro recall, macro F1-score,
per-class results, confusion matrices, and a CPU inference-only timing
benchmark. It saves results to `reports/test_metrics.json`,
`reports/test_predictions.csv`, and `reports/figures/`. Rebuild the completed
Markdown-derived PDF report after rerunning model training/evaluation:

```powershell
py reports/build_orange_report.py
```

The generated report is `reports/Assignment2_Orange_Report.pdf`.

The provided test set has only 22–33 images per class, so its metrics have
substantial sampling uncertainty. Class counts are imbalanced (`grenning` has
the most samples); use macro-averaged metrics and inspect the confusion
matrices rather than relying on accuracy alone. The class `fresh` is stored as
variable-size PNG images while the other classes are 800x800 JPG images. This
format/resolution difference may act as a dataset shortcut; high test scores
may not generalize to images from other cameras or sources.

Do not claim production latency or model quality until measured on the target
hardware and held-out data. The current dataset represents orange images only;
it cannot answer questions about avocado ripeness or internal fruit quality.

## Colab, GitHub, and walkthrough submission

The notebooks can be run in Google Colab after uploading/cloning this project
and making the same `data/raw/train` and `data/raw/test` directory structure
available to the runtime. In Colab, install dependencies with
`%pip install -r requirements.txt`, select a GPU runtime if available, then run
notebooks 1–4 in order. Upload each finished notebook to Google Drive and set
its sharing permission as required by the instructor.

Before publishing to GitHub, create a repository in your own account, review
the dataset licence, and verify that no images, credentials, or model weights
are staged. The project `.gitignore` excludes data and trained weights. Add
your public GitHub and Colab URLs and the walkthrough video link to your final
submission; those account-sharing actions must be completed from your account.
