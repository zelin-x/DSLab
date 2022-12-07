from abc import ABC

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class PieceWiseCNN(nn.Module, ABC):
    def __init__(self, input_size, filter_size, num_filters):
        super(PieceWiseCNN, self).__init__()
        mask_embedding = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding)
        self.hidden_size = num_filters
        self.Conv1d = nn.Conv1d(input_size, num_filters, filter_size, padding=1)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.mask_embedding.weight)
        nn.init.xavier_uniform_(self.Conv1d.weight)
        nn.init.zeros_(self.Conv1d.bias)

    def piece_wise_max_pooling(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, self.hidden_size * 3)

    def forward(self, X, X_mask):
        """
        :param X: embedding layer output
        :param X_mask: the location in sentence (before e1, between e1 and e2, after e2)
        :return: hidden layer output
        """
        out = self.Conv1d(X.transpose(1, 2)).transpose(1, 2)
        out = self.piece_wise_max_pooling(out, X_mask)
        return out


class PCNN_ONE(nn.Module, ABC):
    def __init__(self, opt):
        super(PCNN_ONE, self).__init__()
        if opt.use_plm:
            self.embedding = BertModel.from_pretrained(opt.pretrained_path)
            for param in self.embedding.parameters():
                param.requires_grad = True
        else:
            self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_size)
            nn.init.xavier_uniform_(self.embedding.weight)

        self.pos1_embedding = nn.Embedding(2 * opt.limit_size, opt.pos_dim)
        self.pos2_embedding = nn.Embedding(2 * opt.limit_size, opt.pos_dim)
        self.P_CNN = PieceWiseCNN(opt.embedding_size + 2 * opt.pos_dim, opt.filter_size, opt.num_filters)
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.classifier = nn.Linear(3 * opt.num_filters, opt.classes_num)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_idxes, att_masks, pos1s, pos2s, pos_mask, use_plm=True):
        if use_plm:
            Xe, _ = self.embedding(input_idxes, attention_mask=att_masks, output_all_encoded_layers=False)
        else:
            Xe = self.embedding(input_idxes)
        Xp0, Xp1 = self.pos1_embedding(pos1s), self.pos2_embedding(pos2s)
        X = torch.cat([Xe, Xp0, Xp1], dim=-1)
        X = self.dropout(X)
        out = self.P_CNN(X, pos_mask)
        out = self.dropout(out)
        out = self.classifier(out)
        return out
