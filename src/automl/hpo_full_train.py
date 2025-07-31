
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
import sys
import subprocess
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report
from utils import get_default_transforms
import optuna


import json
class QuickTrain:
    def __init__(self, dataset_name="flowers", batch_size=32, epochs=6, model_name="custom", config_path=None, min_samples_per_class=150, learning_rate=1e-4, optimizer_name="Adam"):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name
        self.config_path = config_path
        self.learning_rate = learning_rate  # NEW
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_samples_per_class = min_samples_per_class
        self.LOG_FILE = "hpo_all_trials.log"

        with open(self.LOG_FILE, "a") as f:
            f.write(f"📝 Training Log for {self.model_name} on {self.dataset_name}\n\n")

        analyze_dataset(self.dataset_name, min_samples_per_class=self.min_samples_per_class)
        self._load_metadata()
        self._prepare_data()
        self._init_model()
        self.optimizer = self._init_optimizer(optimizer_name)
        params = sum(p.numel() for p in self.model.parameters())
        with open(self.LOG_FILE, "a") as f:
            f.write(f"📏 Total Parameters: {params}\n")
            f.write(f"🛠️ Hyperparameters:\n")
            f.write(f"  • Learning Rate: 1e-4\n")
            f.write(f"  • Epochs: {self.epochs}\n")
            f.write(f"  • Batch Size: {self.batch_size}\n")
            f.write(f"  • Optimizer: {self.optimizer}\n")
            f.write(f"  • Loss Function: CrossEntropyLoss\n\n")

    def _load_metadata(self):
        path = os.path.abspath(f"../../dataset_analysis_{self.dataset_name}.json")
        if not os.path.exists(path):
            print("📉 Metadata file not found — running data_analyze.py ...")
            self.metadata = analyze_dataset(self.dataset_name, save_to_file=True, min_samples_per_class=150)
        with open(path, "r") as f:
            self.metadata = json.load(f)


    def _init_optimizer(self, optimizer_name):
        if optimizer_name == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif optimizer_name == "RMSprop":
            return torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    
    def _prepare_data(self):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", self.dataset_name))
        
        # Load CSVs
        df = pd.read_csv(os.path.join(base_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(base_path, "test.csv"))
        images_path = os.path.join(base_path, "images_train")
        test_images_path = os.path.join(base_path, "images_test")

        # Step 1: Balance the training data
        full_train_dataset = BalancedDataset(df, images_path, self.metadata)
        balanced_df = full_train_dataset.df  # This is already balanced

        # Step 2: Train/Validation split
        train_df, val_df = train_test_split(
            balanced_df,
            test_size=0.2,
            stratify=balanced_df["label"],
            random_state=42
        )

        # Step 3: Create BalancedDataset objects
        train_dataset = BalancedDataset(train_df, images_path, self.metadata)
        val_dataset = BalancedDataset(val_df, images_path, self.metadata)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Step 4: Prepare fixed test set (for final evaluation only)
        test_transform = get_default_transforms(self.metadata)
        test_data = [
            (test_transform(Image.open(os.path.join(test_images_path, row['image_file_name'])).convert("RGB")), int(row['label']))
            for _, row in test_df.iterrows()
        ]
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        print(f"🧪 Training on {len(self.train_loader.dataset)} samples. Validation on {len(self.val_loader.dataset)} samples.")
    def evaluate_val(self):
        print("🧪 Evaluating on validation set...")
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"✅ Validation Accuracy: {acc:.4f}")
        return acc

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
        #print(" Using architecture from trial:", best_config)

        self.model = build_model_from_config(
            blocks=blocks,
            dropout=best_config["dropout"],
            pool_type=best_config["pool_type"],
            num_classes=self.metadata["num_classes"],
            input_resolution=(512, 512)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

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
def load_hpo_params(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # If it's from Optuna (wrapped under "params"), unwrap
    if "params" in data:
        return data["params"]
    return data  # Already flat dict

def objective(trial):
    # Sample hyperparameters
    print("this is trial")
    min_samples = trial.suggest_int("min_samples_per_class", 150, 400)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    epochs = 10
    print(f"🔎 Trial {trial.number} trying: min_samples={min_samples}, batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate:.2e}")

    try:
        shared_log_file = "hpo_optuna_trials_shared.log"
        trainer = QuickTrain(
            dataset_name="flowers",
            model_name=f"optuna_trial_{trial.number}",
            config_path="trial_logs_optuna_search_flowers.json",
            min_samples_per_class=min_samples,
            batch_size=batch_size,
            epochs = epochs,

            learning_rate=learning_rate,  # <-- pass to class
            optimizer_name = optimizer_name
        )
        trainer.train()
        acc = trainer.evaluate_val()
        print(f"✅ Trial {trial.number} finished with accuracy: {acc:.4f}")
        return acc
    except Exception as e:
        print(f"❌ Trial {trial.number} failed with exception: {e}")
        return 0.0


def compare_configs(default_path="default_hpo_params.json", optuna_path="hpo_best_trial_params.json"):
    results = []

    # Config 1: Default manual HPO
    default_params = load_hpo_params(default_path)
    trainer_default = QuickTrain(
        dataset_name="flowers",
        model_name="manual_best_config",
        config_path="trial_logs_optuna_search_flowers.json",
        min_samples_per_class=default_params["min_samples_per_class"],
        batch_size=default_params["batch_size"],
        epochs=30,
        learning_rate=default_params["learning_rate"],
        optimizer_name=default_params["optimizer"]
    )
    trainer_default.train()
    acc_default = trainer_default.evaluate_test()
    results.append(("Manual Best Config", acc_default))

    # Config 2: Optuna best trial
    optuna_params = load_hpo_params(optuna_path)
    trainer_optuna = QuickTrain(
        dataset_name="flowers",
        model_name="optuna_best_config",
        config_path="trial_logs_optuna_search_flowers.json",
        min_samples_per_class=optuna_params["min_samples_per_class"],
        batch_size=optuna_params["batch_size"],
        epochs=30,
        learning_rate=optuna_params["learning_rate"],
        optimizer_name=optuna_params["optimizer"]
    )
    trainer_optuna.train()
    acc_optuna = trainer_optuna.evaluate_test()
    results.append(("Optuna Best Trial", acc_optuna))

    # Summary log
    print("\n🔍 Test Accuracy Comparison:")
    print("{:<25} {:>10}".format("Configuration", "Test Acc"))
    for name, acc in results:
        print("{:<25} {:>10.4f}".format(name, acc))

    with open("comparison_summary.txt", "w") as f:
        f.write("Test Accuracy Comparison:\n")
        f.write("{:<25} {:>10}\n".format("Configuration", "Test Acc"))
        for name, acc in results:
            f.write("{:<25} {:>10.4f}\n".format(name, acc))



if __name__ == "__main__":
#============= HPO DISABLED ================
    compare_configs()


#============== HPO CODE ====================
    """
    study = optuna.create_study(
    direction="maximize",
    study_name="optuna_hpo_flowers",
    storage="sqlite:///optuna_hpo_flowers.db",
    load_if_exists=True
)
    study.optimize(objective, timeout=6 * 60 * 60)  # 6 hours in seconds

    print("\n✅ Best trial:")
  
    with open("hpo_best_trial_params.json", "w") as f:
        json.dump({
            "trial_number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params
        }, f, indent=2)
    print(f"Trial #{study.best_trial.number}")
    print(f"  Value: {study.best_trial.value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # 🏁 Final training with best parameters
    print("\n🚀 Retraining final model with best parameters...")
    epochs = 20
    final_trainer = QuickTrain(
        dataset_name="flowers",
        model_name="final_best_model",
        config_path="trial_logs_optuna_search_flowers.json",
        min_samples_per_class=study.best_trial.params["min_samples_per_class"],
        batch_size=study.best_trial.params["batch_size"],
        epochs = epochs,
        learning_rate=study.best_trial.params["learning_rate"],
        optimizer_name= study.best_trial.params["optimizer"]
    )
    final_trainer.train()
    final_trainer.evaluate_test()
"""