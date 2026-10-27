import argparse
from pathlib import Path

import joblib
import torch
from PIL import Image
from torchvision import transforms

from src.classical_model import extract_orange_features
from src.dl_model import build_orange_classifier


def predict_image(image_path, classical_model, dl_model, class_names, device):
    features = extract_orange_features(image_path).reshape(1, -1)
    classical_prediction = classical_model.predict(features)[0]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)
    dl_model.eval()
    with torch.no_grad():
        predicted_index = int(dl_model(batch).argmax(dim=1).item())
    return classical_prediction, class_names[predicted_index]


def main():
    parser = argparse.ArgumentParser(
        description="Classify an orange image using the classical and MobileNetV2 models."
    )
    parser.add_argument("image", type=Path, help="Path to an orange image")
    parser.add_argument(
        "--classical-model",
        type=Path,
        default=Path("models/orange_classical.joblib"),
    )
    parser.add_argument(
        "--deep-model",
        type=Path,
        default=Path("models/orange_mobilenet_v2.pth"),
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error("Image does not exist: {}".format(args.image))
    if not args.classical_model.is_file() or not args.deep_model.is_file():
        parser.error(
            "Train both models in the notebooks first; expected {} and {}.".format(
                args.classical_model, args.deep_model
            )
        )

    checkpoint = torch.load(args.deep_model, map_location="cpu")
    class_names = checkpoint["class_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_model = build_orange_classifier(len(class_names), pretrained=False)
    dl_model.load_state_dict(checkpoint["model_state_dict"])
    dl_model.to(device)
    classical_model = joblib.load(args.classical_model)
    predictions = predict_image(
        args.image, classical_model, dl_model, class_names, device
    )

    print("Image: {}".format(args.image))
    print("Classical SVM: {}".format(predictions[0]))
    print("MobileNetV2: {}".format(predictions[1]))


if __name__ == "__main__":
    main()
