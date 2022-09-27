import torch
from clip.model import CLIP
from typing import Union, Tuple
from torch import nn
from transformers.models.bert import BertPreTrainedModel


class CLIPConcat(nn.Module):
    def __init__(self, clip: CLIP, embed_dim: int):
        super().__init__()

        self.clip = clip
        self.proj_layer = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim),
                                        nn.Tanh(),
                                        nn.Linear(embed_dim, 1))
        self.proj_layer[0].weight.data.normal_(mean=0.0, std=0.02)
        self.proj_layer[0].bias.data.zero_()
        self.proj_layer[2].weight.data.normal_(mean=0.0, std=0.02)
        self.proj_layer[2].bias.data.zero_()

    def forward(self, image, text):
        image_features = self.clip.encode_image(image)  # [B, h]
        text_features = self.clip.encode_text(text)  # [T, h]

        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text

        image_features_ex = image_features[:, None, :].expand(-1, text_features.size(0), -1)
        text_features_ex = text_features[None, :, :].expand(image_features.size(0), -1, -1)
        features_cat = torch.cat([image_features_ex, text_features_ex], dim=-1)  # [B, T, h]
        logits_per_image = self.proj_layer(features_cat).squeeze(-1)  # [B, T]
        return logits_per_image, None
