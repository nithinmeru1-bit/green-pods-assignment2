import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_DIR = Path(__file__).resolve().parent
METRICS_PATH = REPORT_DIR / "test_metrics.json"
OUTPUT_PATH = REPORT_DIR / "Assignment2_Orange_Report.pdf"
CLASS_COUNTS = [
    ("blackspot", 184, 22, 206),
    ("canker", 179, 22, 201),
    ("fresh", 281, 33, 314),
    ("grenning", 347, 22, 369),
]


def paragraph(text, style):
    return Paragraph(text, style)


def make_table(rows, widths, header=True):
    wrapped_rows = [
        [
            Paragraph(str(value), TABLE_CELL_STYLE)
            for value in row
        ]
        for row in rows
    ]
    table = Table(wrapped_rows, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9CA3AF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E46")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def build_report():
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(
            "Run notebooks/04_model_evaluation_and_demo.ipynb first; "
            "reports/test_metrics.json is missing."
        )

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    models = metrics["models"]
    latency = metrics["mobile_model_inference_benchmark"]
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F4E46"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#1F4E46"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    global TABLE_CELL_STYLE
    TABLE_CELL_STYLE = ParagraphStyle(
        "TableCell",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
    )

    story = [
        paragraph("Orange Condition Classification Report", styles["ReportTitle"]),
        paragraph(
            "Classify orange images into the source dataset labels "
            "<b>blackspot</b>, <b>canker</b>, <b>fresh</b>, and <b>grenning</b>. "
            "This is an orange-condition task, not avocado ripeness classification.",
            styles["BodyText"],
        ),
        paragraph("Dataset understanding and preparation", styles["SectionHeading"]),
        paragraph(
            "The original train/test folders contain 1,090 images. The 991 supplied "
            "training images were split with stratification into 842 training and 149 "
            "validation samples. The untouched supplied test set contains "
            "99 images.",
            styles["BodyText"],
        ),
        Spacer(1, 7),
        make_table(
            [["Class", "Train", "Test", "Total"]]
            + [[name, train, test, total] for name, train, test, total in CLASS_COUNTS]
            + [["Total", 991, 99, 1090]],
            [2.4 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
        ),
        Spacer(1, 8),
        paragraph(
            "EDA found no unreadable images. blackspot, canker, and grenning images "
            "are 800x800 JPGs, while fresh images are variable-size PNGs. This "
            "format/resolution difference may act as a dataset shortcut and limits "
            "generalization claims.",
            styles["BodyText"],
        ),
        paragraph("Methods", styles["SectionHeading"]),
        paragraph(
            "The classical baseline combines normalized HSV and CIELAB color "
            "histograms with Local Binary Pattern texture features and a "
            "class-balanced RBF SVM. The second model uses an ImageNet-pretrained "
            "MobileNetV2 with a replacement classification head and frozen feature "
            "backbone. Training augmentation and validation-loss early stopping "
            "were used. MobileNetV2 was trained for 12 epochs, reached 100% "
            "validation accuracy at epoch 10, and had 99.3% validation accuracy "
            "at epoch 12. Its minimum validation loss was approximately 0.0904 "
            "at epoch 12.",
            styles["BodyText"],
        ),
        paragraph("Held-out test results", styles["SectionHeading"]),
        paragraph(
            "Both models were evaluated on the same 99 supplied test images. "
            "The classical model classified 96/99 correctly; MobileNetV2 classified "
            "94/99 correctly.",
            styles["BodyText"],
        ),
        Spacer(1, 7),
        make_table(
            [
                ["Model", "Accuracy", "Macro precision", "Macro recall", "Macro F1"],
                *[
                    [
                        model_name,
                        "{:.2%}".format(values["accuracy"]),
                        "{:.2%}".format(values["macro_precision"]),
                        "{:.2%}".format(values["macro_recall"]),
                        "{:.2%}".format(values["macro_f1"]),
                    ]
                    for model_name, values in models.items()
                ],
            ],
            [1.55 * inch, 0.85 * inch, 1.15 * inch, 1.05 * inch, 0.8 * inch],
        ),
        Spacer(1, 8),
        paragraph(
            "The SVM's three errors were canker images predicted as blackspot. "
            "MobileNetV2 confused two blackspot images and two canker images with "
            "the other diseased class, and predicted one canker as fresh. Both "
            "models classified all grenning images correctly. The SVM's scores were "
            "higher on this split by 2.02 percentage points in accuracy and "
            "2.13 percentage points in macro F1; this small test set does not "
            "establish performance on a new domain.",
            styles["BodyText"],
        ),
        paragraph("Production performance", styles["SectionHeading"]),
        paragraph(
            "No production evaluation has been performed. On this CPU, MobileNetV2 "
            "inference for a preprocessed 224x224 tensor (batch size 1) measured "
            "mean {:.2f} ms, median {:.2f} ms, and p95 {:.2f} ms over 30 timed "
            "runs after five warmups. Image decoding and resizing are excluded; "
            "this is not end-to-end production latency. Evaluate on the intended "
            "hardware, camera, lighting, and external data before making production "
            "claims.".format(
                latency["mean_ms"], latency["median_ms"], latency["p95_ms"]
            ),
            styles["BodyText"],
        ),
        paragraph("Drawbacks and future improvements", styles["SectionHeading"]),
        paragraph(
            "The test split is small (22-33 images per class), class counts are "
            "imbalanced, and image format/resolution correlate with the fresh "
            "label. Camera, lighting, background, and framing can cause dataset "
            "shift; image appearance also cannot measure internal fruit quality. "
            "With more time, collect a larger balanced dataset across sources, "
            "verify labels, evaluate on an external test set, compare validation-"
            "selected fine-tuning/augmentation strategies, and benchmark the full "
            "pipeline on target hardware.",
            styles["BodyText"],
        ),
        paragraph("Time spent", styles["SectionHeading"]),
        paragraph(
            "No reliable time log was kept for dataset inspection, model "
            "experiments, and evaluation, so a measured duration cannot be reported. "
            "If required, enter your own truthful estimate from your work history "
            "rather than inventing a measurement.",
            styles["BodyText"],
        ),
        paragraph("Submission status", styles["SectionHeading"]),
        paragraph(
            "The four notebooks have executed successfully; trained models, "
            "evaluation outputs, and this report are present. Publishing the "
            "repository/notebooks with instructor-approved permissions and "
            "recording the requested walkthrough video still require the student's "
            "own accounts and actions. Confirm the instructor's rubric and required "
            "export format before submission.",
            styles["BodyText"],
        ),
    ]

    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Orange Condition Classification Report",
        author="",
    )
    document.build(story)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    build_report()
