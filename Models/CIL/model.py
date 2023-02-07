from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
import math


class SentenceEncoder(nn.Module, ABC):
    def __init__(self, pretrained_path,
                 limit_size,
                 pos_dim,
                 embedding_size,
                 hidden_size,
                 dropout_rate):
        """
        Bert + BiLSTM + Ent-Pooling
        """
        super(SentenceEncoder, self).__init__()
        self.embedding = BertModel.from_pretrained(pretrained_path)
        for param in self.embedding.parameters():
            param.requires_grad = True

        self.pos1_embedding = nn.Embedding(2 * limit_size, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * limit_size, pos_dim)
        self.encoder = nn.LSTM(input_size=embedding_size + pos_dim * 2,
                               batch_first=True,
                               hidden_size=hidden_size,
                               bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)

    @staticmethod
    def entities_pooling(X, entity_masks):
        """
        :param X: entity calculate pooling
        :param entity_masks: like [[0,0,1,1,...,,0,1,1,0,0,..],..]
        :return:
        """
        entities_sum = torch.matmul(entity_masks, X)  # (batch_size,1, hidden_size)
        entities_sum = entities_sum.squeeze(1)
        entities_len = torch.sum(entity_masks, dim=-1).expand_as(entities_sum)
        entities_pooling = entities_sum / entities_len  # (batch_size, hidden_size)
        return entities_pooling

    def forward(self, token_idxes, att_masks, pos1es, pos2es, head_masks, tail_masks):
        """
        input: (batch_size, max_bag_size, max_sent_length)
        output: (batch_size, max_bag_size, hidden_dim)
        """

        Xe, _ = self.embedding(token_idxes, attention_mask=att_masks, output_all_encoded_layers=False)
        Xp0, Xp1 = self.pos1_embedding(pos1es), self.pos2_embedding(pos2es)
        X = torch.cat([Xe, Xp0, Xp1], dim=-1)
        X = self.dropout(X)

        bs, seq_len = token_idxes.size(0), token_idxes.size(1)
        input_lens = att_masks.sum(-1).cpu()
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(X, input_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = self.encoder(rnn_inputs)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=seq_len)
        out = self.dropout(rnn_outputs)

        head_out = SentenceEncoder.entities_pooling(out, head_masks.unsqueeze(1).float())
        tail_out = SentenceEncoder.entities_pooling(out, tail_masks.unsqueeze(1).float())

        out = torch.cat([head_out, tail_out], dim=-1)

        return out


class Net(nn.Module, ABC):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.sent_encoder = SentenceEncoder(opt.pretrained_path,
                                            opt.limit_size,
                                            opt.pos_dim,
                                            opt.embedding_size,
                                            opt.hidden_size,
                                            opt.dropout_rate)
        self.rel_embedding = nn.Embedding(opt.classes_num, opt.hidden_size * 4)
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.classes_num = opt.classes_num
        self.classifier = nn.Linear(opt.hidden_size * 4, opt.classes_num)

    @staticmethod
    def selective_attention(query, key, training=True):
        if training:
            query = query.view(1, -1)
        alphas = torch.matmul(query, key.T) / math.sqrt(query.size(-1))
        alphas = F.softmax(alphas, dim=-1)
        value = torch.matmul(alphas, key)
        return value

    @staticmethod
    def cl(rep, aug_rep, bag_rep, temperature):
        """
        :param rep: (B, bag, H)
        :param aug_rep: (B, bag, H)
        :param bag_rep: (B, H)
        :param temperature: float32
        :return:
        """
        # (B, bag, H)
        batch_size, bag_size, hidden_size = rep.size()
        aug_rep = aug_rep.view(batch_size, bag_size, hidden_size)
        # positive pairs
        # instance ~ augmented instance
        # (B, bag, H) ~ (B, bag, H) - (B, bag)
        pos_sim = F.cosine_similarity(rep, aug_rep, dim=-1)
        pos_sim = torch.exp(pos_sim / temperature)
        # negative pairs
        # instance ~ other bag representation
        # (B, H) - (B, bag, H)
        tmp_bag_rep = bag_rep.unsqueeze(1).repeat(1, bag_size, 1)
        # each instance ~ its own bag representation
        axis_sim = F.cosine_similarity(rep, tmp_bag_rep, dim=-1)  # (B, bag)
        tmp_bag_rep = bag_rep.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, B, H)
        # (B, bag, H) ~ (B, B, H) - (B, bag, B)
        tmp_rep = rep.permute((1, 2, 0))  # (bag, H, B)
        tmp_bag_rep = tmp_bag_rep.permute((1, 2, 0))  # (B, H, B)
        tmp_bag_rep = tmp_bag_rep.unsqueeze(1)  # (B, 1, H, B)
        # (bag, H, B) ~ (B, 1, H, B) - (B, bag, B)
        pair_sim = F.cosine_similarity(tmp_rep, tmp_bag_rep, dim=-2)  # (B, bag, B)
        # bug sum(2) ? any effect ?
        neg_sim = torch.exp(pair_sim / temperature).sum(2) - torch.exp(axis_sim / temperature)
        pos_sim = pos_sim.view(-1)
        neg_sim = neg_sim.view(-1)
        loss = -1.0 * torch.log(pos_sim / (pos_sim + neg_sim))
        loss = loss.mean()
        return loss

    def forward(self, scope, input1, input2=None, labels=None, training=True, temperature=None):
        flat = lambda x: x.view(-1, x.size(-1))
        if training:
            assert input2 is not None and labels is not None
            batch_size, bag_size, max_len = input1[0].size()
            token_idxes, att_masks, pos1es, pos2es, head_masks, tail_masks = [flat(_) for _ in input1]
            aug_token_idxes, aug_att_masks, aug_pos1es, aug_pos2es, aug_head_masks, aug_tail_masks = [flat(_) for _ in input2]
            # (batch_size, max_bag_size, max_len) -> (batch_size * max_bag_size, max_len)
            sent_reps = self.sent_encoder(token_idxes, att_masks, pos1es, pos2es, head_masks, tail_masks)
            aug_sent_reps = self.sent_encoder(aug_token_idxes, aug_att_masks, aug_pos1es, aug_pos2es, aug_head_masks, aug_tail_masks)

            bag_reps = []
            for idx, s in enumerate(scope):
                rel = labels[idx]
                ins_reps = sent_reps[s[0]:s[1]]
                rel_embed = self.rel_embedding(rel)  # (feature_size, )
                bag_rep = Net.selective_attention(rel_embed, ins_reps)  # (1, feature_size)
                bag_reps.append(bag_rep)
            bag_reps = torch.cat(bag_reps, dim=0)
            bag_out = self.classifier(self.dropout(bag_reps))
            sent_reps = sent_reps.view(batch_size, bag_size, -1)
            aug_sent_reps = aug_sent_reps.view(batch_size, bag_size, -1)
            cl_loss = Net.cl(sent_reps, aug_sent_reps, bag_reps, temperature)
            return bag_out, cl_loss
        else:
            logits = []
            token_idxes, att_masks, pos1es, pos2es, head_masks, tail_masks = [flat(_) for _ in input1]
            sent_reps = self.sent_encoder(token_idxes, att_masks, pos1es, pos2es, head_masks, tail_masks)
            for idx, s in enumerate(scope):
                ins_reps = sent_reps[s[0]:s[1]]
                # (bag_instance_num, feature_size)
                rels_embed = self.rel_embedding(torch.arange(self.classes_num).cuda())
                # (classes_num, feature_size)
                bags_reps = Net.selective_attention(rels_embed, ins_reps, training=False)
                # (classes_num, feature_size)
                out = self.classifier(bags_reps)
                # (classes_num, classes_num)
                logit = F.softmax(out, dim=-1).diag()  # line first
                logits.append(logit.view(1, -1))
            out = torch.cat(logits, dim=0)
            return out
