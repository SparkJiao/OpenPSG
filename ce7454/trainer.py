import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from clip.model import CLIP, convert_weights
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class BaseTrainer:
    def __init__(self,
                 net: nn.Module,
                 train_loader: DataLoader,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100) -> None:
        self.net = net
        self.train_loader = train_loader

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / learning_rate,
            ),
        )

    def train_epoch(self):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        # train_dataiter = iter(self.train_loader)

        # for train_step in tqdm(range(1, len(train_dataiter) + 1)):
        for batch in tqdm(self.train_loader, total=len(self.train_loader)):
            # for train_step in tqdm(range(1, 5)):
            # batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['soft_label'].cuda()
            # forward
            logits = self.net(data)
            loss = F.binary_cross_entropy_with_logits(logits,
                                                      target,
                                                      reduction='sum')
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            del target
            del data
            del batch

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['train_loss'] = loss_avg

        return metrics


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is None:
            continue
        p.grad.data = p.grad.data.float()


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    Refer to https://zhuanlan.zhihu.com/p/138117543.
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros((y_pred.size(0), 1), device=y_pred.device)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.sum(neg_loss + pos_loss)


def circle_loss(y_true: torch.Tensor, y_pred: torch.Tensor, m: float, gamma: float):
    sp = y_pred.clone()
    sn = y_pred.clone()

    ap = torch.clamp_min(-sp.detach() + 1 + m, min=0.)
    an = torch.clamp_min(sn.detach() + m, min=0.)

    delta_p = 1 - m
    delta_n = m

    logit_p = - ap * (sp - delta_p) * gamma
    logit_n = an * (sn - delta_n) * gamma

    logit_p[~y_true.bool()] = -1e6
    logit_n[y_true.bool()] = -1e6

    loss = nn.Softplus()(torch.logsumexp(logit_n, dim=-1) + torch.logsumexp(logit_p, dim=-1))

    return loss.mean()


class CLIPTrainer(BaseTrainer):
    def __init__(self,
                 net: CLIP,
                 train_loader: DataLoader,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100,
                 linear_warmup: bool = False,
                 warmup_proportion: float = 0.1,
                 max_grad_norm: float = 1.0, ) -> None:
        # multi_label_cs: bool = False) -> None:
        super().__init__(net, train_loader, learning_rate, momentum, weight_decay, epochs)

        self.optimizer = torch.optim.AdamW(
            net.parameters(),
            learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=weight_decay
        )
        self.scaler = GradScaler()
        self.max_grad_norm = max_grad_norm

        if linear_warmup:
            train_steps = epochs * len(train_loader)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_proportion * train_steps, train_steps)
        else:
            self.scheduler = None

        # self.multi_label_cs = multi_label_cs

    def train_epoch(self):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        # train_dataiter = iter(self.train_loader)

        # for train_step in tqdm(range(1, len(train_dataiter) + 1)):
        for batch in tqdm(self.train_loader, total=len(self.train_loader)):
            # for train_step in tqdm(range(1, 5)):
            # batch = next(train_dataiter)
            with autocast(dtype=torch.bfloat16):
                data = batch['data'].cuda()
                text = batch['text'][0].cuda()
                target = batch['soft_label'].cuda()
                # forward
                logits_per_image, logits_per_text = self.net(data, text)
                # if self.multi_label_cs:
                #     # loss = multilabel_categorical_crossentropy(target, torch.softmax(logits_per_image, dim=-1))
                #     loss = circle_loss(target, logits_per_image, m=0.25, gamma=256)
                # else:
                loss = F.binary_cross_entropy_with_logits(logits_per_image,
                                                          target,
                                                          reduction='sum')

            # backward
            self.net.zero_grad(set_to_none=True)
            # loss.backward()
            self.scaler.scale(loss).backward()

            if self.max_grad_norm > 0:
                if hasattr(self.optimizer, "clip_grad_norm"):
                    self.optimizer.clip_grad_norm(self.max_grad_norm)
                elif hasattr(self.net, "clip_grad_norm_"):
                    self.net.clip_grad_norm_(self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

            convert_models_to_fp32(self.net)

            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.optimizer.step()
            self.scheduler.step()

            convert_weights(self.net)

            del target
            del data
            del batch

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {
            'train_loss': loss_avg
        }

        return metrics


class CLIPMulTarTrainer(BaseTrainer):
    def __init__(self,
                 net: CLIP,
                 train_loader: DataLoader,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100,
                 linear_warmup: bool = False,
                 cosine_warmup: bool = False,
                 warmup_proportion: float = 0.1,
                 max_grad_norm: float = 1.0, ) -> None:
        super().__init__(net, train_loader, learning_rate, momentum, weight_decay, epochs)

        self.optimizer = torch.optim.AdamW(
            net.parameters(),
            learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=weight_decay
        )
        self.scaler = GradScaler()
        self.max_grad_norm = max_grad_norm

        if linear_warmup:
            train_steps = epochs * len(train_loader)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_proportion * train_steps, train_steps)
        elif cosine_warmup:
            train_steps = epochs * len(train_loader)
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_proportion * train_steps, train_steps)
        else:
            self.scheduler = None

    def train_epoch(self):
        self.net.train()  # enter train mode

        loss_avg = 0.0

        for batch in tqdm(self.train_loader, total=len(self.train_loader)):
            with autocast(dtype=torch.bfloat16):
                batch = {k: v.cuda() for k, v in batch.items()}
                # forward
                logits_per_image, _ = self.net(batch["data"], batch["text"])  # [batch, rel_num]

                pos_index = batch["pos_index"]  # [batch, max_pos_num]
                neg_index = batch["neg_index"]  # [batch, max_neg_num]
                pos_mask = batch["pos_mask"]  # [batch, max_pos_num]
                neg_mask = batch["neg_mask"]  # [batch, max_neg_num]
                labels = batch["labels"]
                labels[~pos_mask.bool()] = -100

                pos_scores = torch.gather(logits_per_image, dim=1, index=pos_index)
                neg_scores = torch.gather(logits_per_image, dim=1, index=neg_index)
                neg_scores[~neg_mask.bool()] = -10000
                pos_scores = pos_scores.unsqueeze(-1)  # [batch_size, max_pos_num, 1]
                neg_scores = neg_scores.unsqueeze(1).expand(-1, pos_index.size(1), -1)  # [batch, max_pos_num, max_neg_num]
                scores = torch.cat([pos_scores, neg_scores], dim=2)  # [batch, max_pos_num, max_neg_num + 1]
                loss = nn.CrossEntropyLoss(ignore_index=-100)(scores.reshape(-1, neg_index.size(1) + 1), labels.reshape(-1))

            # backward
            self.net.zero_grad(set_to_none=True)
            # loss.backward()
            self.scaler.scale(loss).backward()

            if self.max_grad_norm > 0:
                if hasattr(self.optimizer, "clip_grad_norm"):
                    self.optimizer.clip_grad_norm(self.max_grad_norm)
                elif hasattr(self.net, "clip_grad_norm_"):
                    self.net.clip_grad_norm_(self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

            convert_models_to_fp32(self.net)

            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            convert_weights(self.net)

            del batch

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {
            'train_loss': loss_avg
        }

        return metrics
