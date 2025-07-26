import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from feed_data import BalancedDataset
import pandas as pd
import json
import os

import sys  
from data_analyze import analyze_dataset  
from tqdm import tqdm
from PIL import Image
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import os
print("PYTORCH_CUDA_ALLOC_CONF =", os.getenv("PYTORCH_CUDA_ALLOC_CONF"))

class EfficientNetB4Custom(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.base_model = efficientnet_b4(weights=weights)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

class QuickTrain:
    def __init__(self, dataset_name="flowers", batch_size=16, epochs=3):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device:  {self.device}")

        analyze_dataset(self.dataset_name)
        self._load_metadata()
        self._prepare_data()
        self._init_model()

    def _load_metadata(self):
        metadata_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", f"dataset_analysis_{self.dataset_name}.json"))
        if not os.path.exists(metadata_path):
            print("📉 Metadata file not found — running data_analyze.py ...")
            import subprocess
            subprocess.run([
                sys.executable,
                os.path.join(os.path.dirname(__file__), "data_analyze.py")
            ])
            assert os.path.exists(metadata_path), f"❌ Metadata file still not found after analysis: {metadata_path}"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        print("📊 Dataset Metadata:")
        print(json.dumps(self.metadata, indent=2))

    def _prepare_data(self):
        base_path = os.path.expanduser(f"~/Documents/AutoML/data/{self.dataset_name}")
        df = pd.read_csv(os.path.join(base_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(base_path, "test.csv"))

        images_path = os.path.join(base_path, "images_train")
        test_images_path = os.path.join(base_path, "images_test")

        # Train with balancing
        train_dataset = BalancedDataset(df, images_path, self.metadata)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Test set (no augmentation)
        from utils import get_default_transforms
        test_transform = get_default_transforms(self.metadata)
        self.test_loader = DataLoader([
            (test_transform(Image.open(os.path.join(test_images_path, row['image_file_name'])).convert("RGB")), int(row['label']))
            for _, row in test_df.iterrows()
        ], batch_size=self.batch_size)

        print(f"📊 Training samples (balanced): {len(train_dataset)}")
        from collections import Counter

        # Get class counts from final dataset (including duplicates)
        final_counts = Counter(label for _, label, _ in train_dataset.data)
        print(f"🧮 Final training class distribution (after balancing): {dict(final_counts)}")

        print(f"🧪 Test samples: {len(self.test_loader.dataset)}")

    def evaluate_test(self):
        print("🧪 Evaluating on test set...")
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"✅ Test Accuracy: {acc:.4f}")
        return acc
    def _init_model(self):

        self.model = EfficientNetB4Custom(
            num_classes=self.metadata["num_classes"]
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)  # smaller LR for pretrained


    def train(self):
        print("🚀 Starting quick training run...")
        torch.cuda.empty_cache()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss, total_correct = 0, 0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()

            acc = total_correct / len(self.train_loader.dataset)
            avg_loss = total_loss / len(self.train_loader.dataset)
            print(f"📦 Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")
        
        
        # Access the actual dataset instance from the DataLoader
        train_dataset = self.train_loader.dataset
        # ✅ Save augmentation + duplication logs
        with open("augmentation_log.txt", "w") as f:
            f.write("🔁 Duplication Summary per Class:\n")
            for cls, count in train_dataset.duplication_log.items():
                f.write(f"Class {cls}: {count} duplicated samples\n")

            f.write("\n🧪 Unique Augmentations Applied:\n")
            for op in sorted(train_dataset.applied_ops_logger):
                f.write(f"{op}\n")

if __name__ == "__main__":
    trainer = QuickTrain()
    trainer.train()
    trainer.evaluate_test()
