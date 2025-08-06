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
from collections import Counter
import numpy as np
import datetime
from torch.utils.data import WeightedRandomSampler
optuna.logging.set_verbosity(optuna.logging.INFO)

def define_search_space(trial, ds_budget):
    """
    Define the architectural and training hyperparameter search space for Optuna.

    This function builds the architecture block-by-block using fixed filters per stage,
    and samples other parameters (kernel size, SE, residual, depthwise, expansion) 
    for each block. The final block list and global config are stored as user_attrs.
    """
    num_layers = trial.suggest_int("num_layers", 9, 15)
    blocks = []

    # Step 1: Choose which layers will perform downsampling
    all_indices = list(range(num_layers))
    ds_indices = set(random.sample(all_indices, min(ds_budget, num_layers)))

    # Step 2: Define stage-wise filter sizes (fixed, not searched)
    filter_stage = [64, 96, 128, 160, 192]
    stage_size = max(1, num_layers // len(filter_stage))  # Prevent division by zero

    for i in range(num_layers):
        stage_idx = min(i // stage_size, len(filter_stage) - 1)
        filters = filter_stage[stage_idx]

        # Step 3: Search for other architectural decisions
        expansion = 6 if i < 3 else trial.suggest_categorical(f"expansion{i}", [3, 4, 6])
        use_residual = True if i >= num_layers // 2 else trial.suggest_categorical(f"useresidual{i}", [True, False])
        use_depthwise = trial.suggest_categorical(f"usedepthwise{i}", [True, False]) if i % 2 == 0 else False

        # Optional: Save filters for tracking
        trial.set_user_attr(f"filters{i}", filters)

        # Step 4: Build block config
        blocks.append({
            "filters": filters,
            "kernel": trial.suggest_categorical(f"kernel{i}", [3, 5, 7]),
            "use_se": trial.suggest_categorical(f"usese{i}", [True, False]),
            "use_residual": use_residual,
            "downsample": i in ds_indices,
            "expansion": expansion,
            "use_depthwise": use_depthwise,
        })

    # Step 5: Enforce at least 2 SE blocks
    if sum(block["use_se"] for block in blocks) < 2:
        for idx in random.sample(range(num_layers), 2):
            blocks[idx]["use_se"] = True

    downsample_count = sum(block["downsample"] for block in blocks)
    print(f"[Trial {trial.number}] 🔻 Downsampling count: {downsample_count} (Budget: {ds_budget})")

    # Step 6: Global hyperparameters
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    pool_type = trial.suggest_categorical("pool_type", ["max", "avg"])

    # ✅ Step 7: Save the full architecture info for later retrieval
    trial.set_user_attr("arch_blocks", blocks)
    trial.set_user_attr("dropout", dropout)
    trial.set_user_attr("pool_type", pool_type)

    return blocks, dropout, pool_type



def compute_downsampling_budget(input_res, min_output_size=8):
    return math.floor(math.log2(input_res / min_output_size))

def print_current_trial(study, trial):
    if not study.best_trials:  # no successful trials yet
        print(f"[Trial {trial.number}] No successful trial yet.")
        return
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
        # ===== 1. Stratified split before augmentation =====
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["label"],
            random_state=42
        )
         # ===== 2. Augmented training dataset =====
        train_dataset = BalancedDataset(
            train_df,
            images_path,
            metadata,
            resized_res=new_res,
            min_samples_per_class=200  # adjust if you want in NAS
        )
         # ===== 3. Plain validation dataset (no balancing) =====
        val_dataset = BalancedDataset(
            val_df,
            images_path,
            metadata,
            resized_res=new_res,
            min_samples_per_class=0  # disables oversampling
        )
        # ===== Log post-augmentation distribution =====
        with open("class_distribution.log", "a") as f:
            f.write("\n NAS Class distribution AFTER augmentation (NAS):\n")
            aug_labels = [int(lbl) for _, lbl, _ in train_dataset.data]
            for cls, count in sorted(Counter(aug_labels).items()):
                f.write(f"  Class {cls}: {count} samples\n")

        # ===== 4. Weighted sampler for training =====
        train_labels = [int(lbl) for _, lbl, _ in train_dataset.data]  # ensure int
        class_sample_count = np.array([train_labels.count(c) for c in sorted(set(train_labels))])
        weight_per_class = 1.0 / class_sample_count
        sample_weights = [weight_per_class[int(label)] for label in train_labels]  # index with int


        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)



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
        for epoch in range(2):
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
    best_trial = study.best_trial
    # Merge regular params and user attributes (e.g., filters0, filters1, ...)
    best_params = {**best_trial.params, **best_trial.user_attrs}

    config_path = f"best_config_{dataset_name}.json"
    output = {
        "params": best_params,
        "value": best_trial.value  # this is what Optuna minimized (1 - accuracy)
    }
    with open(config_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"[INFO] Best configuration and value saved to: {config_path}")


def optuna_arch_search(dataset_name):
    """
    Entry point to run the Optuna+Hyperband architectural and hyperparameter search.
    """
    storage_path = f"sqlite:///nas_optuna_{dataset_name}.db"
    study = optuna.create_study(
        direction="minimize",
        study_name=f"nas_optuna_{dataset_name}",
        storage=storage_path,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )


    study.optimize(
    lambda trial: objective(trial, dataset_name),
    callbacks=[print_current_trial],
    timeout=300 #3hours NAS
)
    print("Best trial found:", study.best_trial.params)
    save_best_config(study, dataset_name)

        # Merge all searched hyperparameters and user-defined architecture attributes
    merged_params = {**study.best_trial.params, **study.best_trial.user_attrs}
    return merged_params


