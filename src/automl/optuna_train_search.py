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
import time
import math
from feed_data import BalancedDataset
from data_analyze import analyze_dataset
from utils import get_default_transforms
from model_builder import build_model_from_config
from torch.utils.data import random_split
import traceback
import random
optuna.logging.set_verbosity(optuna.logging.INFO)
def define_search_space(trial, ds_budget):
    """
    Define the architectural and training hyperparameter search space for Optuna.
    """
    num_layers = trial.suggest_int("num_layers",6, 12)
    blocks = []
    # Pick exactly ds_budget layers that will downsample
    all_indices = list(range(num_layers))
    ds_indices = set(random.sample(all_indices, min(ds_budget, num_layers)))  # <-- This is the trick
    #here if we have 8 layers we will randomly chose budget many layers. (Because it cannot apply downsampling in all of the layers)

    for i in range(num_layers):
        blocks.append({
            "filters": trial.suggest_categorical(f"filters{i}", [32, 64, 96, 128, 160, 192]),
            "kernel": trial.suggest_categorical(f"kernel{i}", [3, 5, 7]),
            "use_se": trial.suggest_categorical(f"usese{i}", [True, False]),
            "use_residual": trial.suggest_categorical(f"useresidual{i}", [True, False]),
            "downsample": i in ds_indices,
            "expansion": trial.suggest_categorical(f"expansion{i}", [1, 3, 6]),
            "use_depthwise": trial.suggest_categorical(f"usedepthwise{i}", [True, False]),
        })
        # Count how many downsample=True were chosen
    downsample_count = sum(block["downsample"] for block in blocks)
    print(f"[Trial {trial.number}] 🔻 Downsampling count: {downsample_count} (Budget: {ds_budget})")

    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    pool_type = trial.suggest_categorical("pool_type", ["none", "max", "avg"])
    return blocks, dropout, pool_type

import datetime

def compute_downsampling_budget(input_res, min_output_size=8):
    return math.floor(math.log2(input_res / min_output_size))

def print_current_trial(study, trial):
    trial_info = {
        "trial_number": trial.number,
        "value": trial.value,
        "parameters": trial.params,
        "best_so_far": {
            "trial": study.best_trial.number,
            "value": study.best_trial.value
        },
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 🔴 "duration_sec" intentionally removed
    }

    # Optional print to console
    if(trial.value is not None):
        print(f"\n🧪 [Trial {trial.number}] Value: {trial.value:.4f}")
        print(f"  → Parameters: {trial.params}")
        print(f"  → Best so far: Trial {study.best_trial.number} (value: {study.best_trial.value:.4f})")

    # Save safely to JSON log
    log_path = f"trial_logs_{study.study_name}.json"
    logs = load_existing_log_safely(log_path)
    logs.append(trial_info)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=4)


def load_existing_log_safely(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[⚠️ WARNING] Failed to decode {path}. Starting with empty log.")
        return []
    
def objective(trial, dataset_name="flowers"):
    """
    Objective function to minimize (1 - validation accuracy) using Optuna.
    """


    try:
        trial_start_time = time.time()
        # Analyze and fetch metadata
        analyze_dataset(dataset_name)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        metadata_path = os.path.join(project_root, f"dataset_analysis_{dataset_name}.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"[ERROR] Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Fidelity dimension: resolution reduction factor
        resize_factor = 0.5 # half of the resolution
        original_res = metadata["image_resolution"]
        new_res = (int(original_res[0] * resize_factor), int(original_res[1] * resize_factor))

        # Compute downsampling budget based on resized resolution
        target_min_output_size = 8  
        ds_budget = compute_downsampling_budget(min(new_res), target_min_output_size)

        # Load dataset and split train/val
        project_root = Path(__file__).resolve().parents[2]
        base_path = project_root / "data" / dataset_name
        df = pd.read_csv(os.path.join(base_path, "train.csv"))
        images_path = os.path.join(base_path, "images_train")

        #========== Dataset TRAIN / VALIDATION Split
        # Step 1: Create one big balanced dataset first
        balanced_dataset = BalancedDataset(df, images_path, metadata, resized_res=new_res)
        # Step 2: Split the indices

        val_ratio = 0.2
        val_size = int(len(balanced_dataset) * val_ratio)
        train_size = len(balanced_dataset) - val_size
        train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])
        #Step 3: Wrap in DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        #We dont shuffle the validation set because We want deterministic, reproducible evaluation.
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Model initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blocks, dropout, pool_type = define_search_space(trial, ds_budget=ds_budget)
        model = build_model_from_config(blocks, dropout, pool_type,
                                        num_classes=metadata["num_classes"], 
                                        input_resolution=new_res)   
    
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True))
        criterion = nn.CrossEntropyLoss()
        
        # Training loop (5 epochs)
        for epoch in range(6):
            start = time.time()
            print(f"trial : {trial.number} epoch : {epoch}")
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"⏱️ Epoch {epoch} duration: {time.time() - start:.2f}s")
            trial.report(loss.item(), epoch)
            #if trial.should_prune():
                #raise optuna.TrialPruned()

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
        trial.duration = time.time() - trial_start_time
        return 1 - accuracy
    except Exception as e:
        print(f"[⚠️ Trial {trial.number}] Failed with error:\n{traceback.format_exc()}")
        raise optuna.exceptions.TrialPruned()


def save_best_config(study, dataset_name):
    """
    Save the best trial's configuration to a JSON file.
    """
    best_params = study.best_trial.params
    config_path = f"best_config_{dataset_name}.json"
    with open(config_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"[INFO] Best configuration saved to: {config_path}")


def optuna_arch_search(dataset_name="flowers"):
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


    study.optimize(
    lambda trial: objective(trial, dataset_name),
    callbacks=[print_current_trial],
    timeout=18000  # in seconds 5 hours 
)
    print("Best trial found:", study.best_trial.params)
    save_best_config(study, dataset_name)
    return study.best_trial.params
