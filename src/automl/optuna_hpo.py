
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
from torch.utils.data import Dataset
import subprocess
import optuna.visualization.matplotlib as vis
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report
from utils import get_default_transforms
import optuna
import traceback
from random import sample
import json
from torch.utils.data import WeightedRandomSampler
class QuickTrain:
    def __init__(self, dataset_name, batch_size=32, epochs=3, model_name="custom", config_path=None, min_samples_per_class=200, learning_rate=1e-4, optimizer_name="Adam", weight_decay=0.0, architecture_params=None, test_mode=False):
        self.dataset_name = dataset_name
        self.architecture_params = architecture_params  # Best Arch found in NAS
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = model_name
        self.config_path = config_path
        self.learning_rate = learning_rate  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_samples_per_class = min_samples_per_class
        self.LOG_FILE = "hpo_all_trials.log"
     

        with open(self.LOG_FILE, "a") as f:
            f.write(f"📝 Training Log for {self.model_name} on {self.dataset_name}\n\n")

        analyze_dataset(self.dataset_name, min_samples_per_class=self.min_samples_per_class)
        self._load_metadata()
        self._prepare_data(test_mode=test_mode)
        self._init_model()
        self.optimizer = self._init_optimizer(optimizer_name)
        self.weight_decay = weight_decay  # passed in from trial
        params = sum(p.numel() for p in self.model.parameters())
        with open(self.LOG_FILE, "a") as f:
            f.write(f"📏 Total Parameters: {params}\n")
            f.write(f"🛠️ Hyperparameters:\n")
            f.write(f"  • Learning Rate: {self.learning_rate:.2e}\n")
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
        kwargs = dict(lr=self.learning_rate, weight_decay=getattr(self, "weight_decay", 0.0))
        if optimizer_name == "Adam":
            return torch.optim.Adam(self.model.parameters(), **kwargs)
        elif optimizer_name == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), **kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")


    
    from torch.utils.data import WeightedRandomSampler
    def _prepare_data(self, test_mode=False):
        """
        Prepares train, validation, and test loaders.
        Uses augmentation for undersampled classes and WeightedRandomSampler
        to balance batches in the training loader, while keeping the validation
        set stratified (balanced by class proportions).
        """


        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", self.dataset_name))        # ------------------------------------------------------
        # STEP 0: Load CSVs for training and testing
        # ------------------------------------------------------
        df = pd.read_csv(os.path.join(base_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(base_path, "test.csv"))
        images_path = os.path.join(base_path, "images_train")
        test_images_path = os.path.join(base_path, "images_test")

        # ------------------------------------------------------
        # STEP 1: Create balanced dataset with augmentation
        # BalancedDataset will oversample/augment undersampled classes
        # ------------------------------------------------------
        full_train_dataset = BalancedDataset(df, images_path, self.metadata)

        # Get label distribution after augmentation
        balanced_labels = [int(lbl) for _, lbl in [(p, l) for p, l, _ in full_train_dataset.data]]
        self.balanced_counts = dict(Counter(balanced_labels))

        # Log class distribution after augmentation
        with open("class_distribution.log", "a") as f:
            f.write("\n📊 Class distribution AFTER augmentation:\n")
            for cls, count in sorted(self.balanced_counts.items()):
                f.write(f"  Class {cls}: {count} samples\n")

        # ------------------------------------------------------
        # STEP 2: Balanced split for validation
        # ------------------------------------------------------
        val_indices = []
        train_indices = []

        labels = np.array(balanced_labels)
        unique_classes = np.unique(labels)
        min_class_count = min([np.sum(labels == cls) for cls in unique_classes])
        val_per_class = int(min_class_count * 0.2)  # 20% per class

        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            val_cls_idx = np.random.choice(cls_indices, size=val_per_class, replace=False)
            train_cls_idx = np.setdiff1d(cls_indices, val_cls_idx)
            val_indices.extend(val_cls_idx)
            train_indices.extend(train_cls_idx)

        train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)

        # ------------------------------------------------------
        # STEP 3: WeightedRandomSampler for balanced training batches
        # Gives each sample a weight inversely proportional to its class frequency
        # ------------------------------------------------------

        # Get the labels only for the training subset
        train_labels = [balanced_labels[i] for i in train_indices]

        # Count how many samples per class in the training set
        class_sample_count = np.array([train_labels.count(c) for c in sorted(set(train_labels))])

        # Assign weight per class = inverse frequency
        weight_per_class = 1.0 / class_sample_count

        # Map each training sample to its weight
        sample_weights = [weight_per_class[label] for label in train_labels]

        # Create the sampler that will sample *with replacement* according to weights
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),  # one pass over the dataset size
            replacement=True
        )

        # Create the training loader using the sampler (no shuffle needed)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)

        # Validation loader does not use a sampler — just plain stratified subset
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # ------------------------------------------------------
        # DEBUG: Log the first batch's class distribution for verification
        # ------------------------------------------------------
        first_batch_labels = []
        for _, labels_batch in self.train_loader:
            first_batch_labels = labels_batch.cpu().numpy().tolist()
            break  # only log the first batch

        print(f"[DEBUG] First batch class counts: {Counter(first_batch_labels)}")
        with open("weighted_sampler_debug.log", "w") as f:
            f.write(f"First batch classes: {first_batch_labels}\n")
            f.write(f"Class counts in first batch: {dict(Counter(first_batch_labels))}\n")

        # ------------------------------------------------------
        # STEP 4: Prepare test set
        # ------------------------------------------------------
        test_transform = get_default_transforms(self.metadata)
        if test_mode:
            # Test set without labels (for competition/test submission)
            self.test_loader = DataLoader(
                TestImageDataset(test_df, test_images_path, metadata=self.metadata, transform=test_transform),
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            # Test set with labels (for local evaluation)
            test_data = [
                (
                    test_transform(
                        Image.open(os.path.join(test_images_path, row['image_file_name']))
                        .convert("RGB" if self.metadata.get("num_channels", 3) == 3 else "L")
                    ),
                    int(row['label'])
                )
                for _, row in test_df.iterrows()
            ]
            self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        # ------------------------------------------------------
        # FINAL PRINT
        # ------------------------------------------------------
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
  
        if self.architecture_params is not None:
            best_config = self.architecture_params
        else:
            raise ValueError("Must provide architecture_params")
        # Parse block configs
        blocks = best_config["arch_blocks"]
        dropout = best_config["dropout"]
        pool_type = best_config["pool_type"]
        #print(" Using architecture from trial:", best_config)

        self.model = build_model_from_config(
            blocks=blocks,
            dropout=dropout,
            pool_type=pool_type,
            num_classes=self.metadata["num_classes"],
            input_resolution=tuple(self.metadata["image_resolution"])
        ).to(self.device)

        # Get class weights from balanced_counts
        total_samples = sum(self.balanced_counts.values())
        class_weights = torch.tensor(
            [total_samples / self.balanced_counts[cls] for cls in sorted(self.balanced_counts.keys())],
            dtype=torch.float
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

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

    def generate_test_predictions(self, save_path="data/exam_dataset/predictions.npy"):
        """
        Runs inference on self.test_loader and saves predictions.
        Assumes _prepare_data() has already set up the loader.
        """
        preds = []
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Generating predictions"):
                if isinstance(batch, (list, tuple)):  # (image, label) for labeled sets
                    images = batch[0]
                else:  # image only for unlabeled sets
                    images = batch
                images = images.to(self.device)
                outputs = self.model(images)
                preds.extend(outputs.argmax(1).cpu().numpy())

        preds = np.array(preds, dtype=np.int64)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, preds)
        print(f"✅ Predictions saved to {save_path} — shape: {preds.shape}")



    def full_train(self):
        print("🚀 Starting final full training...")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            with open(self.LOG_FILE, "a") as f:
                f.write(f"🖥️ GPU: {gpu_name} | Memory: {total_mem:.2f} GB\n\n")

        torch.cuda.empty_cache()
        start_time = time.time()
        train_acc_list, train_loss_list, val_acc_list = [], [], []
        convergence_epoch = None

        best_valid_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 5
        early_stopping_min_delta = 0.001

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
            train_acc_list.append(acc)
            train_loss_list.append(avg_loss)

            val_acc = self.evaluate_val()
            val_acc_list.append(val_acc)

            print(f"📦 Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")
            with open(self.LOG_FILE, "a") as f:
                f.write(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}\n")
                f.write(f"🧪 Validation Accuracy: {val_acc:.4f}\n")

            if acc == 1.0 and convergence_epoch is None:
                convergence_epoch = epoch + 1
                convergence_time = time.time() - start_time
                with open(self.LOG_FILE, "a") as f:
                    f.write(f"\n✅ Converged at epoch {convergence_epoch} after {convergence_time:.2f} seconds.\n")

            if val_acc - best_valid_acc > early_stopping_min_delta:
                best_valid_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    with open(self.LOG_FILE, "a") as f:
                        f.write(f"⏹️ Early stopping at epoch {epoch+1} — no improvement for {early_stopping_patience} epochs.\n")
                    break

        # Plot convergence
        plt.figure()
        epochs_range = range(1, len(train_acc_list) + 1)

        plt.plot(epochs_range, train_acc_list, label="Train Accuracy")
        plt.plot(epochs_range, val_acc_list, label="Validation Accuracy")
        plt.plot(epochs_range, train_loss_list, label="Train Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"{self.model_name} Convergence Plot")
        plt.legend()

        os.makedirs("plots", exist_ok=True)
        save_path = os.path.join("plots", f"{self.model_name}_{self.dataset_name}_convergence.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        with open(self.LOG_FILE, "a") as f:
            f.write(f"📊 Saved convergence plot to {save_path}\n")

        total_time = time.time() - start_time
        with open(self.LOG_FILE, "a") as f:
            f.write(f"\n⏱️ Total Training Time: {total_time:.2f} seconds\n")
        model_save_path = os.path.join("checkpoints", f"{self.model_name}_{self.dataset_name}_final.pth")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)

        with open(self.LOG_FILE, "a") as f:
            f.write(f"💾 Saved final trained model to {model_save_path}\n")

        return self
    
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
        val_acc_list = []

      
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

            val_acc = self.evaluate_val()
            val_acc_list.append(val_acc)
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


class TestImageDataset(Dataset):
    def __init__(self, csv_df, images_dir, metadata, transform=None):
        """
        Args:
            csv_df (DataFrame): test.csv dataframe
            images_dir (str): path to images_test folder
            metadata (dict): dataset metadata containing 'num_channels'
            transform: torchvision transforms
        """
        self.df = csv_df
        self.images_dir = images_dir
        self.metadata = metadata
        self.transform = transform

        # Decide image mode based on metadata channels
        self.image_mode = "RGB" if self.metadata.get("num_channels", 3) == 3 else "L"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.df.iloc[idx]['image_file_name'])
        image = Image.open(img_path).convert(self.image_mode)  # mode based on channels
        if self.transform:
            image = self.transform(image)
        return image


      
def load_hpo_params(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # If it's from Optuna (wrapped under "params"), unwrap
    if "params" in data:
        return data["params"]
    return data  # Already flat dict

def objective(trial, dataset_name, architecture_params, test_mode, median_count):
    # 1. Restrict optimizer choices based on CV domain knowledge


    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    # 2. Learning rate ranges tailored for Adam/AdamW
    if optimizer_name == "AdamW":
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    else:
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

    # 3. Weight decay for regularization
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # 4. Other relevant HPO parameters
    min_samples = trial.suggest_int("min_samples_per_class", max(200,median_count), 400)
    batch_size = trial.suggest_categorical("batch_size", [16,32])


    print(f"🔎 Trial {trial.number} trying: optimizer={optimizer_name}, lr={lr:.2e}, weight_decay={weight_decay:.1e}, min_samples={min_samples}, batch_size={batch_size}")

    try:
        trainer = QuickTrain(
            dataset_name=dataset_name,
            model_name=f"optuna_trial_{trial.number}",
            architecture_params=architecture_params,
            min_samples_per_class=min_samples,
            batch_size=batch_size,
            epochs=2,
            learning_rate=lr,
            optimizer_name=optimizer_name,
            test_mode=test_mode
        
        )

        # Inject weight decay into optimizer (requires modifying _init_optimizer)
        trainer.optimizer.param_groups[0]['weight_decay'] = weight_decay

        trainer.train()
        acc = trainer.evaluate_val()
        print(f"✅ Trial {trial.number} finished with accuracy: {acc:.4f}")
        return acc
    except Exception as e:
        log_path = "hpo_trial_errors.log"
        error_msg = (
        f"\n❌ Trial {trial.number} failed\n"
        f"Parameters: optimizer={optimizer_name}, lr={lr}, "
        f"weight_decay={weight_decay}, min_samples={min_samples}, batch_size={batch_size}\n"
        f"Error: {e}\n"
        f"Traceback:\n{traceback.format_exc()}\n"
    )
        print(error_msg)  # still show in console

        # Append to log file
        with open(log_path, "a") as log_file:
            log_file.write(error_msg)
        print(f"❌ Trial {trial.number} failed with exception: {e}")
        return 0.0


def run_hpo(dataset_name, architecture_params, test_mode):
    print("HPO OPTUNA")
    # compare_configs()
    # 🧪 HPO using Optuna
    study_name = f"optuna_hpo_{dataset_name}"
    storage_path = f"sqlite:///optuna_hpo_{dataset_name}.db"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True
    )

    # 🔁 Warm-start Optuna with a strong baseline
    study.enqueue_trial({
        "min_samples_per_class": 200,
        "batch_size": 32,
        "optimizer": "Adam",
        "lr": 1e-4,
        "weight_decay": 1e-4
    })
        # Get median from current dataset
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", dataset_name))
    train_csv_path = os.path.join(base_path, "train.csv")
    class_counts_before = dict(Counter(pd.read_csv(train_csv_path)["label"]))    
    median_count = int(np.median(list(class_counts_before.values())))


    # Start the HPO process
    study.optimize(lambda trial: objective(trial, dataset_name, architecture_params, test_mode,median_count), 
     n_trials=2) #2hours HPO

    # ✅ Log the best trial
    print("\n✅ Best trial:")
    print(f"Trial #{study.best_trial.number}")
    print(f"  Value: {study.best_trial.value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # 💾 Save best parameters to file
    with open("hpo_best_trial_params.json", "w") as f:
        json.dump({
            "trial_number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params
        }, f, indent=2)

    print("HPO ENDED") # DEBUG

    os.makedirs("plots", exist_ok=True)
    plt.rcParams['figure.figsize'] = (6, 4)

    print("📈 Saving HPO plots...")

    # Optimization history
    vis.plot_optimization_history(study)
    plt.gcf().savefig(f"plots/{dataset_name}_optuna_history.png", dpi=300, bbox_inches='tight')
    plt.clf()

    # Parameter importances
    #vis.plot_param_importances(study)
    #plt.gcf().savefig(f"plots/{dataset_name}_optuna_importance.png", dpi=300, bbox_inches='tight')
    #plt.clf()

    print("✅ HPO complete and visualizations saved.")

    return study.best_trial.params


