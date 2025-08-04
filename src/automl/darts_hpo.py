import os
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import optuna
import numpy as np
import json
from darts.genotypes import Genotype
from types import SimpleNamespace
  # Adjust import if needed
from darts.utils import accuracy, AvgrageMeter, _data_transforms_fashion
from .feed_data import BalancedDataset
from darts.model import NetworkCIFAR
from pathlib import Path
# Number of classes in your dataset
FASHION_CLASSES = 10

# Path to your saved genotype from the DARTS search phase
GENOTYPE_PATH = "./darts/search-EXP-20250719-154808/genotype.txt"  # Update with your actual path

# Dataset root path
DATA_ROOT = 'C:/Users/ecepu/Documents/AutoML/data/fashion'  # Update as needed
TRIAL_LOG_PATH = "trial_log.txt"

def load_genotype(path):
    print(f"[INFO] Loading genotype from: {path}")
    with open(path, "r") as f:
        genotype_str = f.read().strip()
    genotype = eval(genotype_str)
    print(f"[INFO] Genotype loaded: {genotype}")
    return genotype


def get_data_loaders(args):
    
    print("[INFO] Preparing data loaders...")

    train_transform, valid_transform = _data_transforms_fashion(None)
    dataset_name = "fashion"
    project_root = Path(__file__).resolve().parents[2]
    metadata_path = project_root / "dataset_analysis_fashion.json"

    if not os.path.exists(metadata_path):
        print(f"[WARNING] Metadata not found at {metadata_path}, falling back...")
        metadata_path = f"dataset_analysis_{dataset_name}.json"

    print(f"[INFO] Using metadata: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    df_path = os.path.join(args.data, dataset_name, "train.csv")
    images_path = os.path.join(args.data, dataset_name, "images_train")

    print(f"[INFO] Loading CSV from: {df_path}")
    print(f"[INFO] Loading images from: {images_path}")

    df = pd.read_csv(df_path)
    train_data = BalancedDataset(df, images_path, metadata)

    num_train = len(train_data)
    print(f"[INFO] Total samples: {num_train}")
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=6)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=6)
    return train_loader, valid_loader




def train_one_epoch(model, train_loader, criterion, optimizer, auxiliary_weight=0.4):
    print("[INFO] Starting training epoch...")
    model.train()
    for batch_idx, (inputs, targets) in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs, aux_outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        if aux_outputs is not None:
            aux_loss = criterion(aux_outputs, targets)
            loss += auxiliary_weight * aux_loss
            
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"[Train] Batch {batch_idx}, Loss: {loss.item():.4f}")


def validate(model, valid_loader, criterion):
    print("[INFO] Starting validation...")
    model.eval()
    objs = AvgrageMeter()
    top1 = AvgrageMeter()

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            prec1, _ = accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
    print(f"[INFO] Validation complete. Avg Loss: {objs.avg:.4f}, Top-1 Acc: {top1.avg:.2f}%")
    
    return top1.avg, objs.avg


def objective(trial):
    
    print(f"\n[TRIAL {trial.number}] Starting trial with Optuna...\n")
    
    args = SimpleNamespace(
        data='C:\\Users\\PC\\Desktop\\autoML\\Project\\data',
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        lr=trial.suggest_float("lr", 1e-4, 0.1, log=True),
        momentum=trial.suggest_float("momentum", 0.7, 0.99),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        drop_path_prob=trial.suggest_float("drop_path_prob", 0.0, 0.4),
        auxiliary=trial.suggest_categorical("auxiliary", [True, False]),
        auxiliary_weight=trial.suggest_float("auxiliary_weight", 0.1, 0.7),
        report_freq=50,
        gpu=0,
        epochs=70,
        init_channels=36,
        layers=20,
        model_path='saved_models',
        cutout=False,
        cutout_length=16,
        save='EXP',
        seed=0,
        arch='BEST_GENOTYPE',
        grad_clip=5,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        train_portion=0.7
    )

    # Load genotype fixed architecture

    genotype = load_genotype(GENOTYPE_PATH)


    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    print("[INFO] Initializing model...")
    # Initialize model with fixed genotype architecture
    model = NetworkCIFAR(C=16, num_classes=FASHION_CLASSES, layers=8, auxiliary=args.auxiliary, genotype=genotype)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-3)  # 5 epochs for tuning

    train_loader, valid_loader = get_data_loaders(args)
    for epoch in range(5):  # Use small number of epochs for quick evaluation
        train_one_epoch(model, train_loader, criterion, optimizer, auxiliary_weight=args.auxiliary_weight)
        scheduler.step()

    val_acc, val_loss = validate(model, valid_loader, criterion)
    print(f"[INFO] Trial {trial.number} - Validation Accuracy: {val_acc:.2f}%\n")
    
    # Logging trial result
    with open(TRIAL_LOG_PATH, "a") as f:
        f.write(
            f"Trial {trial.number:03d} | lr={args.lr:.5f}, momentum={args.momentum:.3f}, "
            f"weight_decay={args.weight_decay:.1e}, batch_size={args.batch_size} | "
            f"val_acc={val_acc:.2f}%\n"
        )
    # Memory cleanup
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # Optuna tries to maximize validation accuracy
    return val_acc


if __name__ == "__main__":
    
    print("[INFO] Starting Optuna hyperparameter search...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_trial = study.best_trial
    print("[INFO] Best trial complete:")
    print(f"Accuracy: {best_trial.value:.2f}%")
    print("Best params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
    with open("best_trial.txt", "w") as f:
        f.write("Best Trial\n")
        f.write(f"Validation Accuracy: {best_trial.value:.2f}%\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
