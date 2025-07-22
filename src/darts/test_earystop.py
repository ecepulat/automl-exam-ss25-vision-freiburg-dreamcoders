import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from types import SimpleNamespace
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# === Manually insert root to sys.path ===
sys.path.insert(0, os.path.abspath("src"))

# === Custom imports ===
from darts.model import NetworkCIFAR as Network
from darts.genotypes import BEST_GENOTYPE as genotype
from darts.utils import AvgrageMeter, accuracy, _data_transforms_fashion, load
from automl.datasets import FashionDataset

# === Configuration ===
FASHION_CLASSES = 10

args = SimpleNamespace(
    data="C:/Users/ecepu/Documents/AutoML/data",
    batch_size=24,
    report_freq=50,
    gpu=0,
    init_channels=36,
    layers=20,
    model_path="eval-EXP-20250722-133450/weights.pt",  # <-- Pointing to early-stopped model
    auxiliary=False,
    cutout=False,
    cutout_length=16,
    drop_path_prob=0.2,
    seed=0,
)

# === Logging ===
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt="%m/%d %I:%M:%S %p"
)

def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    logging.info("gpu device = %d", args.gpu)
    logging.info("args = %s", args)

    model = Network(args.init_channels, FASHION_CLASSES, args.layers, args.auxiliary, genotype).cuda()
    load(model, args.model_path)

    logging.info("param size = %fMB", sum(p.numel() for p in model.parameters()) / 1e6)

    criterion = nn.CrossEntropyLoss().cuda()
    _, test_transform = _data_transforms_fashion(args)

    test_data = FashionDataset(root=args.data, split="test", download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2
    )

    model.drop_path_prob = args.drop_path_prob
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info("FINAL test_acc: %f", test_acc)

def infer(test_queue, model, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    all_preds = []
    all_targets = []

    for step, (input, target) in enumerate(test_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(target.cpu().numpy())

        if step % args.report_freq == 0:
            logging.info("test %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    # Save CSV
    pd.DataFrame({'target': all_targets, 'predicted': all_preds}).to_csv("test_predictions.csv", index=False)
    logging.info("Saved predictions to test_predictions.csv")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    plt.close()
    logging.info("Saved confusion matrix to confusion_matrix.png")

    # Classification Report
    report = classification_report(all_targets, all_preds, digits=4)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    logging.info("Saved classification report to classification_report.txt")

    return top1.avg, objs.avg

if __name__ == "__main__":
    main()
