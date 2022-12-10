from abc import ABC

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class Net(nn.Module, ABC):
    def __init__(self, opt):
        """
        Bert + BiLSTM + Ent-Pooling
        """
        super(Net, self).__init__()
        self.embedding = BertModel.from_pretrained(opt.pretrained_path)
        for param in self.embedding.parameters():
            param.requires_grad = True

        self.pos1_embedding = nn.Embedding(2 * opt.limit_size, opt.pos_dim)
        self.pos2_embedding = nn.Embedding(2 * opt.limit_size, opt.pos_dim)
        self.encoder = nn.LSTM(input_size=opt.embedding_size + opt.pos_dim * 2,
                               batch_first=True,
                               hidden_size=opt.hidden_size,
                               bidirectional=True)
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(opt.hidden_size * 4, opt.hidden_size * 2),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(opt.hidden_size * 2, opt.classes_num))
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.zeros_(self.classifier[0].bias)
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

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

    def forward(self, batch_data):
        token_idxes, att_masks, pos1es, pos2es, head_masks, tail_masks = batch_data

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

        head_out = Net.entities_pooling(out, head_masks.unsqueeze(1).float())
        tail_out = Net.entities_pooling(out, tail_masks.unsqueeze(1).float())

        out = torch.cat([head_out, tail_out], dim=-1)
        out = self.classifier(out)

        return out
