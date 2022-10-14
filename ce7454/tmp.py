
# net - CLIP model
# pos_index[b, max_pos] - indices of the positive relations
# neg_index[b, max_neg] - indices of the negative relations
# pos_mask[b, max_pos]  - indicating true values in pos_index
# neg_mask[b, max_neg]  - indicating true values in neg_index

# forward
# [batch, rel_num]
logits_per_image, _ = net(batch["data"], batch["text"])
# [batch, max_pos]
labels = batch["labels"]
labels[~pos_mask.bool()] = -100  # Ignore padding positions

pos_scores = torch.gather(logits_per_image, 
                          dim=1, index=pos_index)
neg_scores = torch.gather(logits_per_image, 
                          dim=1, index=neg_index)
neg_scores[~neg_mask.bool()] = -10000
# [batch_size, max_pos_num, 1]
pos_scores = pos_scores.unsqueeze(-1)
# [batch, max_pos_num, max_neg_num]
neg_scores = neg_scores.unsqueeze(1)
neg_scores = neg_scores.expand(-1, pos_index.size(1), -1)
# [batch, max_pos_num, max_neg_num + 1]
scores = torch.cat([pos_scores, neg_scores], dim=2)

loss_fct = CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(scores.reshape(-1, neg_index.size(1) + 1),
                labels.reshape(-1))
