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

import utils
from .model import NetworkCIFAR as Network
from .genotypes import Genotype


class GenericImageDataset(Dataset):
    def __init__(self, root, split="test", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        csv_path = os.path.join(root, f"{split}.csv")
        image_dir = os.path.join(root, f"images_{split}")
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["file_name"])
        label = int(row["label"])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_on_test(dataset_name, best_architecture_params, save_dir):
    print("\n🧪 Evaluating final model on test set...")

    metadata_path = os.path.join("data", dataset_name, f"dataset_analysis_{dataset_name}.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    class_count = metadata["n_classes"]
    transform = utils.get_dynamic_transform(metadata, resize_to=32)

    test_data = GenericImageDataset(root=os.path.join("data", dataset_name), split="test", transform=transform)
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
            preds = outputs.argmax(dim=1)
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
