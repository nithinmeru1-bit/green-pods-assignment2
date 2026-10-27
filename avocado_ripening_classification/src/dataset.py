from pathlib import Path
from typing import Dict, List, Tuple, Union

from sklearn.model_selection import train_test_split

ImageSample = Tuple[Path, str]
PathLike = Union[str, Path]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover_images(split_dir: PathLike) -> List[ImageSample]:
    """Return image paths and their directory labels from one dataset split."""
    split_path = Path(split_dir)
    if not split_path.is_dir():
        raise FileNotFoundError("Dataset split directory does not exist: {}".format(split_path))

    samples = []
    for class_dir in sorted(path for path in split_path.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_dir.name))

    if not samples:
        raise ValueError("No supported images found in dataset split: {}".format(split_path))
    return samples


def get_dataset_splits(
    data_dir: PathLike = "data/raw",
    validation_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, List[ImageSample]]:
    """Use the supplied test set and stratify a validation set from training data."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be greater than 0 and less than 1.")

    root = Path(data_dir)
    train_samples = discover_images(root / "train")
    test_samples = discover_images(root / "test")
    class_names = sorted({label for _, label in train_samples})
    test_classes = {label for _, label in test_samples}
    if class_names != sorted(test_classes):
        raise ValueError(
            "Train and test class folders must match. Train: {}; test: {}.".format(
                class_names, sorted(test_classes)
            )
        )

    train_paths = [str(path) for path, _ in train_samples]
    train_labels = [label for _, label in train_samples]
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=train_labels,
    )
    test_samples = sorted(test_samples, key=lambda sample: (sample[1], str(sample[0])))

    return {
        "train": list(zip(map(Path, train_paths), train_labels)),
        "validation": list(zip(map(Path, validation_paths), validation_labels)),
        "test": test_samples,
        "class_names": class_names,
    }
