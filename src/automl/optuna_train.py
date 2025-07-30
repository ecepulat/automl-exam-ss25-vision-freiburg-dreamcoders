import os
# Set environment configs BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Allow dynamic CUDA memory expansion
# Below did not work run this in your terminal -> export ATEN_NO_NNPACK=1
# os.environ["ATEN_NO_NNPACK"] = "1" # Disable NNPACK to silence warnings
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import pandas as pd
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from model_builder import build_model_from_config
from data_analyze import analyze_dataset
from feed_data import BalancedDataset
import time

def load_best_trial_config(log_path):
    with open(log_path, "r") as f:
        logs = json.load(f)
    best_trial = min(logs, key=lambda t: t["value"])
    print(f"\n🏆 Loaded best trial: #{best_trial['trial_number']} (Val Accuracy: {1 - best_trial['value']:.4f})")
    return best_trial["parameters"]

def build_blocks_from_params(params):
    blocks = []
    for i in range(params["num_layers"]):
        blocks.append({
            "filters": params[f"filters{i}"],
            "kernel": params[f"kernel{i}"],
            "use_se": params[f"usese{i}"],
            "use_residual": params[f"useresidual{i}"],
            "expansion": params[f"expansion{i}"],
            "use_depthwise": params[f"usedepthwise{i}"],
            "downsample": False  # Avoid auto-generated DS strategy
        })
    return blocks

def main(dataset_name="flowers", num_epochs=50):
    # Analyze and load dataset metadata
    analyze_dataset(dataset_name)
    project_root = Path(__file__).resolve().parents[2]
    metadata_path = project_root / f"dataset_analysis_{dataset_name}.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    image_res = tuple(metadata["image_resolution"])
    num_classes = metadata["num_classes"]

    # Load best trial parameters
    # log_path = f"trial_logs_optuna_search_{dataset_name}.json"
    log_path = Path(__file__).resolve().parent / f"trial_logs_optuna_search_{dataset_name}.json"
    best_params = load_best_trial_config(log_path)
    blocks = build_blocks_from_params(best_params)

    # Load dataset
    base_path = project_root / "data" / dataset_name
    df = pd.read_csv(base_path / "train.csv")
    images_path = base_path / "images_train"
    full_dataset = BalancedDataset(df, images_path, metadata, resized_res=None)  # no resizing

    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) # Decreased from 32 to 4 due to CUDA out of memory error
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Build model
    model = build_model_from_config(
        blocks,
        best_params["dropout"],
        best_params["pool_type"],
        num_classes=num_classes,
        input_resolution=image_res
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Starting full training for {num_epochs} epochs...\n")

    scaler = GradScaler() # 🔧 Initialize mixed precision scaler


    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"[🌱 Epoch {epoch+1}] ------------------------")

        # Train
        accumulation_steps = 2  # Accumulate gradients over 2 steps

        model.train()
        total_loss = 0.0

        optimizer.zero_grad()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                print(f"[Debug] Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps  # undo division for logging

        # Only average once at the end
        avg_loss = total_loss / len(train_loader)


        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        duration = time.time() - start_time

        print(f"📊 Train Loss: {avg_loss:.4f}")
        print(f"✅ Val Accuracy: {val_acc:.4f}")
        print(f"⏱️ Epoch duration: {duration:.2f}s\n")
        torch.cuda.empty_cache()


    print("🎉 ✅ Full training completed successfully!")

if __name__ == "__main__":
    main()
