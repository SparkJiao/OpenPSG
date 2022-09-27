import argparse
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
parser.add_argument('--epoch', type=int, default=36)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--linear_warmup', default=False, action='store_true')
parser.add_argument('--warmup_proportion', default=0.1, type=float)
parser.add_argument('--max_grad_norm', default=0.0, type=float)
parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--multi_label_cs', default=False, action='store_true')
# parser.add_argument('--context_len', type=int, default=77)
parser.add_argument('--add_prompt', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# savename = f'{args.model_name}_e{args.epoch}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_wp{args.warmup_proportion}_' \
#            f'mcs{args.multi_label_cs}_s{args.seed}'
savename = f'{args.model_name}_e{args.epoch}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}_wp{args.warmup_proportion}_' \
           f'prompt{args.add_prompt}_s{args.seed}'
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results', exist_ok=True)

device = torch.device("cuda:0")

# loading model
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
train_dataset = PSGTextDataset(stage='train', preprocess=preprocess, add_prompt=args.add_prompt)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

val_dataset = PSGTextDataset(stage='val', preprocess=preprocess, add_prompt=args.add_prompt)
val_dataloader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=4)

test_dataset = PSGTextDataset(stage='test', preprocess=preprocess, add_prompt=args.add_prompt)
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=4)
print('Data Loaded...', flush=True)

# loading trainer
trainer = CLIPTrainer(model,
                      train_dataloader,
                      learning_rate=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      epochs=args.epoch,
                      linear_warmup=args.linear_warmup,
                      warmup_proportion=args.warmup_proportion,
                      max_grad_norm=args.max_grad_norm,)
                      # multi_label_cs=args.multi_label_cs)
evaluator = CLIPEvaluator(model, k=3)

# train!
print('Start Training...', flush=True)
begin_epoch = time.time()
best_val_recall = 0.0
for epoch in range(0, args.epoch):
    train_metrics = trainer.train_epoch()
    val_metrics = evaluator.eval_recall(val_dataloader)

    # show log
    print(
        '{} | Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | mR {:.2f}'
        .format(savename, (epoch + 1), int(time.time() - begin_epoch),
                train_metrics['train_loss'], val_metrics['test_loss'],
                100.0 * val_metrics['mean_recall']),
        flush=True)

    # save model
    if val_metrics['mean_recall'] >= best_val_recall:
        torch.save(model.state_dict(), f'./checkpoints/{savename}_best.ckpt')
        best_val_recall = val_metrics['mean_recall']

print('Training Completed...', flush=True)

# saving result!
print('Loading Best Ckpt...', flush=True)
checkpoint = torch.load(f'checkpoints/{savename}_best.ckpt')
model.load_state_dict(checkpoint)
test_evaluator = CLIPEvaluator(model, k=3)
check_metrics = test_evaluator.eval_recall(val_dataloader)
if best_val_recall == check_metrics['mean_recall']:
    print('Successfully load best checkpoint with acc {:.2f}'.format(
        100 * best_val_recall),
        flush=True)
else:
    print('Fail to load best checkpoint')
result = test_evaluator.submit(test_dataloader)

# save into the file
with open(f'results/{savename}_{best_val_recall}.txt', 'w') as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
print('Result Saved!', flush=True)
