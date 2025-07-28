import os
import json
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

from feed_data import BalancedDataset
from data_analyze import analyze_dataset
from utils import get_default_transforms
from model_builder import build_model_from_config


def define_search_space(trial):
    """
    Define the architectural and training hyperparameter search space for Optuna.
    """
    num_layers = trial.suggest_int("num_layers", 6, 20)
    blocks = []
    for i in range(num_layers):
        blocks.append({
            "filters": trial.suggest_categorical(f"filters{i}", [32, 64, 96, 128, 160, 192]),
            "kernel": trial.suggest_categorical(f"kernel{i}", [3, 5, 7]),
            "use_se": trial.suggest_categorical(f"usese{i}", [True, False]),
            "use_residual": trial.suggest_categorical(f"useresidual{i}", [True, False]),
            "downsample": trial.suggest_categorical(f"downsample{i}", [True, False]),
            "expansion": trial.suggest_categorical(f"expansion{i}", [1, 3, 6]),
            "use_depthwise": trial.suggest_categorical(f"usedepthwise{i}", [True, False]),
        })

    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    pool_type = trial.suggest_categorical("pool_type", ["none", "max", "avg"])
    return blocks, dropout, pool_type


def objective(trial, dataset_name="flowers"):
    """
    Objective function to minimize (1 - validation accuracy) using Optuna.
    """
    # Analyze and fetch metadata
    analyze_dataset(dataset_name)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    metadata_path = os.path.join(project_root, f"dataset_analysis_{dataset_name}.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"[ERROR] Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Fidelity dimension: resolution reduction factor
    resize_factor = trial.suggest_float("resize_factor", 0.25, 1.0)
    original_res = metadata["image_resolution"]
    new_res = (int(original_res[0] * resize_factor), int(original_res[1] * resize_factor))
    metadata["image_resolution"] = new_res

    # Load dataset and split train/val
    project_root = Path(__file__).resolve().parents[2]
    base_path = project_root / "data" / dataset_name
    df = pd.read_csv(os.path.join(base_path, "train.csv"))
    images_path = os.path.join(base_path, "images_train")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    # Train dataset with augmentations
    train_dataset = BalancedDataset(train_df, images_path, metadata)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Validation dataset (no augmentation, just normalization)
    val_transform = get_default_transforms(metadata)
    val_dataset = [
        (val_transform(Image.open(os.path.join(images_path, row["image_file_name"])).convert("RGB")),
         int(row["label"]))
        for _, row in val_df.iterrows()
    ]
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blocks, dropout, pool_type = define_search_space(trial)
    model = build_model_from_config(blocks, dropout, pool_type, num_classes=metadata["num_classes"])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True))
    criterion = nn.CrossEntropyLoss()

    # Training loop (5 epochs)
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trial.report(loss.item(), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Validation evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return 1 - accuracy


def save_best_config(study, dataset_name):
    """
    Save the best trial's configuration to a JSON file.
    """
    best_params = study.best_trial.params
    config_path = f"best_config_{dataset_name}.json"
    with open(config_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"[INFO] Best configuration saved to: {config_path}")


def run_optuna_hyperband(dataset_name="flowers"):
    """
    Entry point to run the Optuna+Hyperband architectural and hyperparameter search.
    """
    storage_path = f"sqlite:///optuna_{dataset_name}.db"
    study = optuna.create_study(
        direction="minimize",
        study_name=f"optuna_search_{dataset_name}",
        storage=storage_path,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(lambda trial: objective(trial, dataset_name), n_trials=50)

    print("Best trial found:", study.best_trial.params)
    save_best_config(study, dataset_name)


if __name__ == "__main__":
    run_optuna_hyperband("flowers")
