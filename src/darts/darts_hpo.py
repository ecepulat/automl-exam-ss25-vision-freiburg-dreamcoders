
import os
import json
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from types import SimpleNamespace
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import optuna
import numpy as np
from tqdm import tqdm

from . import utils
from .model import NetworkCIFAR as Network
from .genotypes import Genotype
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class GenericImageDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
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
        image_path = os.path.join(self.image_dir, row["image_file_name"])
        label = int(row["label"])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

def train(train_queue, model, criterion, optimizer, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = model(input)
        logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1 = utils.accuracy(logits, target, topk=(1,))[0]
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
    return top1.avg

def infer(valid_queue, model, criterion, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(input)
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits, target)
            prec1 = utils.accuracy(logits, target, topk=(1,))[0]
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
    return top1.avg


def run_darts_hpo(dataset_name, best_architecture_params, metadata_path=None):

    if metadata_path is None:
        metadata_path = f"dataset_analysis_{dataset_name}.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    # class_count = metadata["n_classes"]
    class_count = metadata.get("num_classes", metadata.get("n_classes"))

    transform = utils.get_dynamic_transform(metadata, resize_to=32)
    full_data = GenericImageDataset(root=os.path.join("data", dataset_name), split="train", transform=transform)
    train_indices, val_indices = train_test_split(
        list(range(len(full_data))),
        test_size=0.2,
        stratify=[full_data.df.iloc[i]["label"] for i in range(len(full_data))],
        random_state=42
    )
    train_data = Subset(full_data, train_indices)
    valid_data = Subset(full_data, val_indices)

    def objective(trial):
        print(f"[Optuna] Trial {trial.number + 1}")

        args = SimpleNamespace(
            batch_size = trial.suggest_categorical("batch_size", [64, 96]),
            lr = trial.suggest_float("lr", 1e-2, 0.1, log=True),  # Start from higher LR
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True),
            drop_path_prob=trial.suggest_float("drop_path_prob", 0.0, 0.4),
            grad_clip=5,
            report_freq=50,
            gpu=0,
            epochs=20 # Tried 5 for debug
        )

        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        torch.manual_seed(0)
        cudnn.enabled = True
        torch.cuda.manual_seed(0)

        genotype = Genotype(
            normal=best_architecture_params["normal"],
            normal_concat=best_architecture_params["normal_concat"],
            reduce=best_architecture_params["reduce"],
            reduce_concat=best_architecture_params["reduce_concat"]
        )

        model = Network(24, class_count, 20, False, genotype).cuda() # It was 36 before
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        train_queue = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        best_val_acc = 0
        for epoch in range(args.epochs):
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
            train_acc = train(train_queue, model, criterion, optimizer, args)
            val_acc = infer(valid_queue, model, criterion, args)
            best_val_acc = max(best_val_acc, val_acc)
            print(f"Epoch {epoch+1}/{args.epochs} | Val Acc: {val_acc:.2f}")
        return best_val_acc

    n_trials = 30 # tried 5 for debug
    pbar = tqdm(total=n_trials)

    def callback(study, trial):
        pbar.set_description(f"Trial {trial.number + 1} | Best: {study.best_value:.2f}")
        pbar.update(1)

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    pbar.close()


    print("\n🏆 Best DARTS HPO configuration:")
    for k, v in study.best_params.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    return study.best_params
