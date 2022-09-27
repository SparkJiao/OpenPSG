import argparse
import json
import os
import random
import time

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PSGTextDataset
from evaluator import CLIPEvaluator
from models import CLIPConcat
from trainer import CLIPTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='clip')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--infer_data', type=str, default='train')
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0")


model, preprocess = clip.load("ViT-B/32", device=device)
if args.model_name == "clip":
    pass
elif args.model_name == "clip_concat":
    model = CLIPConcat(model, embed_dim=model.text_projection.data.size(1))
    model.to(device)
else:
    raise ValueError(args.model_name)
print('Model Loaded...', flush=True)


# loading dataset
train_dataset = PSGTextDataset(stage='train', preprocess=preprocess)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=4)

val_dataset = PSGTextDataset(stage='val', preprocess=preprocess)
val_dataloader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=4)

test_dataset = PSGTextDataset(stage='test', preprocess=preprocess)
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=4)
print('Data Loaded...', flush=True)

# saving result!
print('Loading Best Ckpt...', flush=True)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint)
test_evaluator = CLIPEvaluator(model, k=args.top_k)

if args.infer_data == 'train':
    _dataloader = train_dataloader
elif args.infer_data == 'val':
    _dataloader = val_dataloader
else:
    _dataloader = test_dataloader
result = test_evaluator.submit(_dataloader, return_scores=True)

# save into the file
with open(f'{args.checkpoint[:-5]}_infer_{args.infer_data}.txt', 'w') as f:
    json.dump(result, f)

