import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from types import SimpleNamespace
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

from . import utils
from .model import NetworkCIFAR as Network
from .genotypes import Genotype


class GenericImageDataset(Dataset):
    def __init__(self, root, split="test", transform=None, is_grayscale=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.is_grayscale = is_grayscale
        csv_path = os.path.join(root, f"{split}.csv")
        image_dir = os.path.join(root, f"images_{split}")
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_file_name"])
        label = int(row["label"])
        image = Image.open(image_path).convert("L" if self.is_grayscale else "RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_on_test(dataset_name, best_architecture_params, save_dir):
    print("\n🧪 Evaluating final model on test set...")

    metadata_path = f"dataset_analysis_{dataset_name}.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    class_count = metadata["num_classes"]
    is_grayscale = metadata.get("is_grayscale", False)
    transform = utils.get_dynamic_transform(metadata, resize_to=32)

    test_data = GenericImageDataset(
        root=os.path.join("data", dataset_name),
        split="test",
        transform=transform,
        is_grayscale=is_grayscale
    )
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

    genotype = Genotype(
        normal=best_architecture_params["normal"],
        normal_concat=best_architecture_params["normal_concat"],
        reduce=best_architecture_params["reduce"],
        reduce_concat=best_architecture_params["reduce_concat"]
    )

    args = SimpleNamespace(
        init_channels=36,
        layers=20,
        auxiliary=False,
        arch="DARTS_FINAL",
        gpu=0
    )

    model = Network(args.init_channels, class_count, args.layers, args.auxiliary, genotype).cuda()
    model_path = os.path.join(save_dir, "weights.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ Test Accuracy: {acc:.4f}")

    report = classification_report(all_labels, all_preds, digits=4)
    log_file = os.path.join(save_dir, "log.txt")

    with open(log_file, "a") as f:
        f.write(f"\n✅ Test Accuracy: {acc:.4f}\n")
        f.write("\n📄 Classification Report:\n")
        f.write(report + "\n")

    return acc
def generate_test_predictions(dataset_name, best_architecture_params, save_dir):
    """
    Generates predictions on the unlabeled test set and saves them under
    project_root/data/exam_dataset/predictions.npy
    """
    print("\n🧪 Generating predictions for unlabeled test set...")

    metadata_path = f"dataset_analysis_{dataset_name}.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    class_count = metadata["num_classes"]
    is_grayscale = metadata.get("is_grayscale", False)
    transform = utils.get_dynamic_transform(metadata, resize_to=32)

    # Load test images (ignore labels)
    test_data = GenericImageDataset(
        root=os.path.join("data", dataset_name),
        split="test",
        transform=transform,
        is_grayscale=is_grayscale
    )
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

    # Build model
    genotype = Genotype(
        normal=best_architecture_params["normal"],
        normal_concat=best_architecture_params["normal_concat"],
        reduce=best_architecture_params["reduce"],
        reduce_concat=best_architecture_params["reduce_concat"]
    )

    args = SimpleNamespace(
        init_channels=36,
        layers=20,
        auxiliary=False,
        arch="DARTS_FINAL",
        gpu=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(args.init_channels, class_count, args.layers, args.auxiliary, genotype).to(device)

    model_path = os.path.join(save_dir, "weights.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for images, _ in test_loader:  # Ignore labels
            images = images.to(device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds.extend(logits.argmax(dim=1).cpu().numpy())

    # ✅ Save exactly like Optuna pipeline (project root relative)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_path = os.path.join(project_root, "data", "exam_dataset", "predictions.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(preds, dtype=np.int64))

    print(f"✅ Predictions saved to {save_path} — shape: {len(preds)}")
    return preds
