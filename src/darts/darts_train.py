import os
import sys
import time
import glob
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from types import SimpleNamespace
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from . import utils
from .model import NetworkCIFAR as Network
from .genotypes import Genotype
from .darts_test import evaluate_on_test


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

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input) #  to unpack the model's output if it's a tuple (which happens when auxiliary logits are returned)
            logits = output[0] if isinstance(output, tuple) else output

            loss = criterion(logits, target)

            prec1 = utils.accuracy(logits, target, topk=(1,))[0]
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def full_train(dataset_name, best_architecture_params, best_hpo_params, class_count):
    print("\n🚀 Retraining final model with best architecture + best hyperparameters (DARTS)...")

    torch.cuda.empty_cache()

    genotype = Genotype(
        normal=best_architecture_params["normal"],
        normal_concat=best_architecture_params["normal_concat"],
        reduce=best_architecture_params["reduce"],
        reduce_concat=best_architecture_params["reduce_concat"]
    )

    save_dir = f"final-train-{dataset_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    utils.create_exp_dir(save_dir, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    args = SimpleNamespace(
        data=os.path.join(os.getcwd(), "data"),
        batch_size=best_hpo_params["batch_size"],
        learning_rate=best_hpo_params["lr"],
        momentum=0.9,
        weight_decay=best_hpo_params["weight_decay"],
        report_freq=50,
        gpu=0,
        epochs=70,  # 🔧 Reduced from 70 for testing
        init_channels=36,
        layers=20,
        model_path='saved_models',
        auxiliary=False,
        auxiliary_weight=0.4,
        cutout=False,
        cutout_length=16,
        drop_path_prob=0.2,
        save=save_dir,
        seed=0,
        arch='DARTS_FINAL',
        grad_clip=5,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001
    )

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    model = Network(args.init_channels, class_count, args.layers, args.auxiliary, genotype).cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    metadata_path = os.path.join(f"dataset_analysis_{dataset_name}.json")

    with open(metadata_path) as f:
        metadata = json.load(f)

    transform = utils.get_dynamic_transform(metadata, resize_to=32)

    full_data = GenericImageDataset(root=os.path.join(args.data, dataset_name), split="train", transform=transform)

    train_indices, val_indices = train_test_split(
        list(range(len(full_data))),
        test_size=0.2,
        stratify=[full_data.df.iloc[i]["label"] for i in range(len(full_data))],
        random_state=42
    )

    train_data = Subset(full_data, train_indices)
    valid_data = Subset(full_data, val_indices)

    train_queue = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=6)
    valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    best_valid_acc = 0
    patience_counter = 0

    for epoch in tqdm(range(args.epochs), desc="Final Training"):
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, _ = train(train_queue, model, criterion, optimizer, args)
        valid_acc, _ = infer(valid_queue, model, criterion, args)

        logging.info('train_acc %f', train_acc)
        logging.info('valid_acc %f', valid_acc)

        scheduler.step()

        if valid_acc - best_valid_acc > args.early_stopping_min_delta:
            best_valid_acc = valid_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logging.info("Early stopping triggered at epoch %d", epoch)
                break

    print(f"\n✅ Final validation accuracy: {best_valid_acc:.2f}%")
    logging.info("Best validation accuracy: %.2f%%", best_valid_acc)
    
    # ✅ Save model weights before test evaluation
    torch.save(model.state_dict(), os.path.join(save_dir, "weights.pt"))

    # 🧪 Evaluate on test set
    evaluate_on_test(
        dataset_name=dataset_name,
        best_architecture_params=best_architecture_params,
        save_dir=save_dir
    )