import io
import json
import logging
import os

import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torch.utils.data import Dataset, default_collate
import clip

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_transforms(stage: str):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    if stage == 'train':
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((1333, 800)),
            trn.RandomHorizontalFlip(),
            trn.RandomCrop((1333, 800), padding=4),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    elif stage in ['val', 'test']:
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((1333, 800)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])


class PSGClsDataset(Dataset):
    def __init__(
            self,
            stage,
            root='./data/coco/',
            num_classes=56,
    ):
        super(PSGClsDataset, self).__init__()
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if d['image_id'] in dataset[f'{stage}_image_ids']
        ]
        self.root = root
        self.transform_image = get_transforms(stage)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(0)
        soft_label[sample['relations']] = 1
        sample['soft_label'] = soft_label
        del sample['relations']
        return sample


class PSGTextDataset(PSGClsDataset):
    def __init__(self,
                 stage,
                 preprocess,
                 root='./data/coco/',
                 context_len=77,
                 add_prompt=False,
                 num_classes=56):
        super().__init__(stage, root, num_classes)

        with open('./data/psg/psg_cls_basic.json') as f:
            relations = json.load(f)["predicate_classes"]

        self.preprocess = preprocess
        self.relations = relations
        if add_prompt:
            self.relations = [f"Someone or some object is {rel} another person of object." for rel in self.relations]
        self.clip_text_inputs = clip.tokenize(self.relations, context_length=context_len)

        print(add_prompt, context_len)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        try:
            # with open(path, 'rb') as f:
            #     content = f.read()
            #     filebytes = content
            #     buff = io.BytesIO(filebytes)
            #     image = Image.open(buff).convert('RGB')
            #     sample['data'] = self.transform_image(image)
            sample['data'] = self.preprocess(Image.open(path))
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(0)
        soft_label[sample['relations']] = 1
        sample['soft_label'] = soft_label
        del sample['relations']
        sample["text"] = self.clip_text_inputs
        return sample


class PSGTextMultiTargetDataset(PSGClsDataset):
    def __init__(self,
                 stage,
                 preprocess,
                 root='./data/coco/',
                 context_len=77,
                 add_prompt=False,
                 num_classes=56):
        super().__init__(stage, root, num_classes)

        with open('./data/psg/psg_cls_basic.json') as f:
            relations = json.load(f)["predicate_classes"]

        self.preprocess = preprocess
        self.relations = relations
        if add_prompt:
            self.relations = [f"Someone or some object is {rel} another person of object." for rel in self.relations]
        self.clip_text_inputs = clip.tokenize(self.relations, context_length=context_len)

        print(add_prompt, context_len)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        try:
            data = self.preprocess(Image.open(path))
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        # Generate Soft Label
        # soft_label = torch.Tensor(self.num_classes)
        # soft_label.fill_(0)
        # soft_label[sample['relations']] = 1
        # sample['soft_label'] = soft_label
        # del sample['relations']
        # sample["text"] = self.clip_text_inputs

        # Generate a new tensor with size [batch, max_positive_num, max_negative_num + 1]
        # The loss can be calculating the softmax over the last dimension and obtain the average.

        return {
            "data": data,
            "text": self.clip_text_inputs,
            "relations": sample["relations"]
        }


def collate_fn(batch):
    batch_relations = [b.pop("relations") for b in batch]

    batch = default_collate(batch)
    batch["text"] = batch["text"][0]
    batch_size = len(batch_relations)

    max_rel_num = max(map(len, batch_relations))
    pos_index = torch.zeros(batch_size, max_rel_num, dtype=torch.long)
    pos_mask = torch.zeros(batch_size, max_rel_num, dtype=torch.long)
    neg_indices = []
    for b, item_relations in enumerate(batch_relations):
        pos_index[b, :len(item_relations)] = torch.tensor(item_relations, dtype=torch.long)
        pos_mask[b, :len(item_relations)] = 1
        neg_indices.append([i for i in range(56) if i not in item_relations])

    max_neg_num = max(map(len, neg_indices))
    neg_index = torch.zeros(batch_size, max_neg_num, dtype=torch.long)
    neg_mask = torch.zeros(batch_size, max_neg_num, dtype=torch.long)
    for b, item_neg_indices in enumerate(neg_indices):
        neg_index[b, :len(item_neg_indices)] = torch.tensor(item_neg_indices, dtype=torch.long)
        neg_mask[b, :len(item_neg_indices)] = 1

    labels = torch.zeros(batch_size, max_rel_num, dtype=torch.long)
    batch["pos_index"] = pos_index
    batch["pos_mask"] = pos_mask
    batch["neg_index"] = neg_index
    batch["neg_mask"] = neg_mask
    batch["labels"] = labels
    return batch
