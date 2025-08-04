import os
import sys
import time
import glob
import numpy as np
import torch
import darts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.autograd import Variable
from darts.model_search import Network
from darts.architect import Architect
import torch.multiprocessing
from pathlib import Path
torch.cuda.empty_cache()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .feed_data import BalancedDataset
from .data_analyze import analyze_dataset
import pandas as pd
import json

FASHION_CLASSES = 10

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  torch.cuda.set_device(0)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss().cuda()
  model = Network(args.init_channels, FASHION_CLASSES, args.layers, criterion).cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_fashion(args)

  dataset_name = "fashion"
  analyze_dataset(dataset_name)

  project_root = Path(__file__).resolve().parents[2]
  metadata_path = project_root / "dataset_analysis_fashion.json"

  #metadata_path = os.path.join(args.data, dataset_name, f"dataset_analysis_{dataset_name}.json")
  if not os.path.exists(metadata_path):
      metadata_path = f"dataset_analysis_{dataset_name}.json"

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

  for epoch in tqdm(range(args.epochs), desc="Epochs"):
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)
    print(f"Epoch {epoch+1}/{args.epochs} | Learning Rate: {lr:.6f}")

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)
    print(f"Training   - Loss: {train_obj:.4f}, Accuracy: {train_acc:.2f}%")
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    print(f"Validation - Loss: {valid_obj:.4f}, Accuracy: {valid_acc:.2f}%")

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    
  # save genotype
  genotype = model.genotype()
  genotype_path = os.path.join(args.save, "genotype.json")
  with open(genotype_path, "w") as f:
      json.dump(genotype._asdict(), f, indent=2)
  print(f"Saved final genotype to {genotype_path}")


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  model.drop_path_prob = args.drop_path_prob
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

def infer(valid_queue, model, criterion):
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


if __name__ == '__main__':
  import torch.multiprocessing
  torch.multiprocessing.set_start_method('spawn', force=True)

  from types import SimpleNamespace
  args = SimpleNamespace(
      #data='C:/Users/ecepu/Documents/AutoML/data',
      data = 'C:\\Users\\PC\\Desktop\\autoML\\Project\\data',
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
      train_portion=0.3, #######################0.7
      unrolled=False,
      arch_learning_rate=3e-4,
      arch_weight_decay=1e-3,
      arch='DARTS',
      auxiliary=False,
      auxiliary_weight=0.4
  )

  args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  main()
