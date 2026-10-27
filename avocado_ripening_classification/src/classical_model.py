from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_orange_features(image_path: str) -> np.ndarray:
    """Combine normalized HSV/LAB color histograms with an LBP texture histogram."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read image: {}".format(image_path))
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    histograms = [
        cv2.calcHist([hsv], [channel], None, [16], [0, 256]).flatten()
        for channel in (0, 1, 2)
    ]
    histograms.extend(
        cv2.calcHist([lab], [channel], None, [16], [0, 256]).flatten()
        for channel in (1, 2)
    )
    color_features = np.concatenate(histograms).astype(np.float32)
    color_features /= np.maximum(color_features.sum(), 1.0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    texture_features, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, 12), range=(0, 11)
    )
    texture_features = texture_features.astype(np.float32)
    texture_features /= max(float(texture_features.sum()), 1.0)
    return np.concatenate([color_features, texture_features])


def extract_feature_matrix(samples: Iterable[Tuple[Path, str]]):
    samples = list(samples)
    features = np.vstack([extract_orange_features(path) for path, _ in samples])
    labels = np.asarray([label for _, label in samples])
    return features, labels


def build_classical_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", class_weight="balanced")),
        ]
    )
