
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from feed_data import BalancedDataset
import pandas as pd
import json
from utils import get_augmented_transforms
import os
import numpy as np
from model_builder import build_model_from_config
from data_analyze import analyze_dataset
from tqdm import tqdm
from PIL import Image
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import sys
import subprocess
import time
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils import get_default_transforms

class EfficientNetB4Custom(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.base_model = efficientnet_b4(weights=weights)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

class QuickTrain:
    def __init__(self, dataset_name="flowers", batch_size=32, epochs=6, model_name="efficientnet", config_path=None):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.LOG_FILE = f"{self.model_name}_{self.dataset_name}.log"
        with open(self.LOG_FILE, "w") as f:
            f.write(f"📝 Training Log for {self.model_name} on {self.dataset_name}\n\n")

        analyze_dataset(self.dataset_name)
        self._load_metadata()
        self._prepare_data()
        self._init_model()

        params = sum(p.numel() for p in self.model.parameters())
        with open(self.LOG_FILE, "a") as f:
            f.write(f"📏 Total Parameters: {params}\n")
            f.write(f"🛠️ Hyperparameters:\n")
            f.write(f"  • Learning Rate: 1e-4\n")
            f.write(f"  • Epochs: {self.epochs}\n")
            f.write(f"  • Batch Size: {self.batch_size}\n")
            f.write(f"  • Optimizer: Adam\n")
            f.write(f"  • Loss Function: CrossEntropyLoss\n\n")

    def _load_metadata(self):
        path = os.path.abspath(f"../../dataset_analysis_{self.dataset_name}.json")
        if not os.path.exists(path):
            print("📉 Metadata file not found — running data_analyze.py ...")
            subprocess.run([sys.executable, "data_analyze.py"])
        with open(path, "r") as f:
            self.metadata = json.load(f)

    def _prepare_data(self):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", self.dataset_name))
        df = pd.read_csv(os.path.join(base_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(base_path, "test.csv"))
        images_path = os.path.join(base_path, "images_train")
        test_images_path = os.path.join(base_path, "images_test")

        train_transform = get_augmented_transforms(self.metadata)
        train_dataset = BalancedDataset(df, images_path, self.metadata)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        print(f" Training on {len(self.train_loader.dataset)} samples.")
        test_transform = get_default_transforms(self.metadata)
        self.test_loader = DataLoader([
            (test_transform(Image.open(os.path.join(test_images_path, row['image_file_name'])).convert("RGB")), int(row['label']))
            for _, row in test_df.iterrows()
        ], batch_size=self.batch_size)

    def _get_best_custom_config(self):
        with open(self.config_path, "r") as f:
            trials = json.load(f)

        # Find best trial number from the last entry
        for entry in reversed(trials):
            if "best_so_far" in entry:
                best_trial_number = entry["best_so_far"]["trial"]
                break

        # Find the corresponding full trial entry
        for entry in trials:
            if entry.get("trial_number") == best_trial_number:
                return entry["parameters"]

        raise ValueError("Best trial not found in log file")

    def _init_model(self):
        if self.model_name == "efficientnet":
            self.model = EfficientNetB4Custom(num_classes=self.metadata["num_classes"]).to(self.device)
        elif self.model_name == "custom":
            assert self.config_path is not None, "Provide path to config JSON file for custom model."
            best_config = self._get_best_custom_config()

            # Parse block configs
            num_layers = best_config["num_layers"]
            blocks = []
            for i in range(num_layers):
                blocks.append({
                    "filters": best_config[f"filters{i}"],
                    "kernel": best_config[f"kernel{i}"],
                    "use_se": best_config[f"usese{i}"],
                    "use_residual": best_config[f"useresidual{i}"],
                    "expansion": best_config[f"expansion{i}"],
                    "use_depthwise": best_config[f"usedepthwise{i}"],
                    "downsample": False if i == 0 else True  # example logic
                })
            print(" Using architecture from trial:", best_config)

            self.model = build_model_from_config(
                blocks=blocks,
                dropout=best_config["dropout"],
                pool_type=best_config["pool_type"],
                num_classes=self.metadata["num_classes"],
                input_resolution=(512, 512)
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)


    def evaluate_test(self):
        print("🧪 Evaluating on test set...")
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"✅ Test Accuracy: {acc:.4f}")

        report = classification_report(all_labels, all_preds, digits=4)
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(xticks_rotation=90, cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.model_name}_{self.dataset_name}_confusion_matrix.png")
        plt.close()

        with open(self.LOG_FILE, "a") as f:
            f.write(f"\n✅ Test Accuracy: {acc:.4f}\n")
            f.write("\n📄 Classification Report:\n")
            f.write(report + "\n")

        return acc

    def train(self):
        print("🚀 Starting training...")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            with open(self.LOG_FILE, "a") as f:
                f.write(f"🖥️ GPU: {gpu_name} | Memory: {total_mem:.2f} GB\n\n")

        torch.cuda.empty_cache()
        start_time = time.time()
        train_acc_list, train_loss_list = [], []
        convergence_epoch = None

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
            train_acc_list.append(acc)
            train_loss_list.append(avg_loss)

            val_acc = self.evaluate_test()

            with open(self.LOG_FILE, "a") as f:
                f.write(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}\n")
                f.write(f"🧪 Validation Accuracy: {val_acc:.4f}\n")

            if acc == 1.0 and convergence_epoch is None:
                convergence_epoch = epoch + 1
                convergence_time = time.time() - start_time
                with open(self.LOG_FILE, "a") as f:
                    f.write(f"\n✅ Converged at epoch {convergence_epoch} after {convergence_time:.2f} seconds.\n")

        total_time = time.time() - start_time
        with open(self.LOG_FILE, "a") as f:
            f.write(f"\n⏱️ Total Training Time: {total_time:.2f} seconds\n")

        plt.figure()
        plt.plot(range(1, self.epochs + 1), train_acc_list, label="Accuracy")
        plt.plot(range(1, self.epochs + 1), train_loss_list, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"{self.model_name} Convergence Plot")
        plt.legend()
        plt.savefig(f"{self.model_name}_{self.dataset_name}_convergence.png")
        plt.close()



if __name__ == "__main__":
    trainer = QuickTrain(
        dataset_name="flowers",
        model_name="custom",
        config_path="trial_logs_optuna_search_flowers.json"
    )
    trainer.train()
    trainer.evaluate_test()