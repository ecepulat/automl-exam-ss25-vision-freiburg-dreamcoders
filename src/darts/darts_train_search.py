import os
import sys
import time
import glob
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from pathlib import Path

from .model_search import Network # Use a relative import to locate the files
from .architect import Architect
from automl.feed_data import BalancedDataset
from types import SimpleNamespace
import pandas as pd
from . import utils


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    valid_iter = iter(valid_queue)

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        try:
            input_search, target_search = next(valid_iter)
        except StopIteration:
            valid_iter = iter(valid_queue)
            input_search, target_search = next(valid_iter)

        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def run_darts_arch_search(args, dataset_name):
    if not torch.cuda.is_available():
        logging.error('No GPU device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    logging.info(f"Running DARTS search on '{dataset_name}'")
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss().cuda()
    model = Network(args.init_channels, 10, args.layers, criterion).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_fashion(args)

    metadata_path = os.path.join(args.data, dataset_name, f"dataset_analysis_{dataset_name}.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    df = pd.read_csv(os.path.join(args.data, dataset_name, "train.csv"))
    images_path = os.path.join(args.data, dataset_name, "images_train")
    train_data = BalancedDataset(df, images_path, metadata)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=6)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    best_val_acc = 0
    best_genotype = None

    for epoch in tqdm(range(args.epochs), desc="DARTS Search Epochs"):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args)
        valid_acc, valid_obj = infer(valid_queue, model, criterion, args)

        logging.info('train_acc %f', train_acc)
        logging.info('valid_acc %f', valid_acc)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_genotype = model.genotype()

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    return best_val_acc, best_genotype

def darts_arch_search(dataset_name: str):
    project_root = Path(__file__).resolve().parents[2]
    data = str(project_root / "data")

    args = SimpleNamespace(
        data=os.path.join(os.getcwd(), "data"),
        batch_size=24,
        learning_rate=0.025,
        learning_rate_min=0.001,
        momentum=0.9,
        weight_decay=3e-4,
        report_freq=50,
        gpu=0,
        epochs=5,
        init_channels=8,
        layers=5,
        model_path='saved_models',
        cutout=False,
        cutout_length=16,
        drop_path_prob=0.2,
        save='EXP',
        seed=0,
        grad_clip=5,
        train_portion=0.3,
        unrolled=False,
        arch_learning_rate=3e-4,
        arch_weight_decay=1e-3,
        arch='DARTS',
        auxiliary=False,
        auxiliary_weight=0.4
    )

    args.save = f'search-{args.save}-{time.strftime("%Y%m%d-%H%M%S")}'
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    best_val_acc, best_genotype = run_darts_arch_search(args, dataset_name)

    genotype_dict = {
        "normal": best_genotype.normal,
        "normal_concat": best_genotype.normal_concat,
        "reduce": best_genotype.reduce,
        "reduce_concat": best_genotype.reduce_concat
    }

    print(f"\n🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print("Best genotype (DARTS):")
    print(json.dumps(genotype_dict, indent=2))

    genotype_path = os.path.join(args.save, f"best_genotype_{dataset_name}.json")
    with open(genotype_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "genotype": genotype_dict,
            "val_accuracy": best_val_acc,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    logging.info(f"[✓] Final genotype saved to {genotype_path}")
    return genotype_dict
