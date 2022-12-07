from abc import ABC

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class PieceWiseCNN(nn.Module, ABC):
    def __init__(self, word_dim, pos_dim, filter_size, num_filters, lam):
        super(PieceWiseCNN, self).__init__()
        mask_embedding = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding)
        self.hidden_size = num_filters
        self.Conv1d = nn.Conv1d(word_dim * 3, num_filters, filter_size, padding=1)
        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(word_dim + 2 * pos_dim, 3 * word_dim)
        self.dropout = nn.Dropout(0.5)
        self.lam = lam
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.mask_embedding.weight)
        nn.init.xavier_uniform_(self.Conv1d.weight)
        nn.init.zeros_(self.Conv1d.bias)

    def piece_wise_max_pooling(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, self.hidden_size * 3)

    def forward(self, Xp, Xe, X_mask):
        """
        :param Xp: word embedding and pos embedding
        :param Xe: word embedding and entity embedding
        :param X_mask: the location in sentence (before e1, between e1 and e2, after e2)
        :return: hidden layer output
        """
        # A = torch.sigmoid(self.fc1(Xe / self.lam))
        A = torch.sigmoid(self.fc1(Xe))
        X = A * Xe + (1 - A) * torch.tanh(self.fc2(Xp))
        out = self.Conv1d(X.transpose(1, 2)).transpose(1, 2)
        out = self.piece_wise_max_pooling(out, X_mask)
        out = torch.tanh(out)
        return out


class SAN(nn.Module, ABC):
    def __init__(self, word_dim, pos_dim, lam=1.):
        super(SAN, self).__init__()
        self.fc1 = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2 = nn.Linear(word_dim + 2 * pos_dim, 3 * word_dim)
        self.fc1_att = nn.Linear(3 * word_dim, 3 * word_dim)
        self.fc2_att = nn.Linear(3 * word_dim, 3 * word_dim)
        self.lam = lam
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc1_att.weight)
        nn.init.xavier_uniform_(self.fc2_att.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc1_att.bias)
        nn.init.zeros_(self.fc2_att.bias)

    def forward(self, Xp, Xe):
        # embedding
        A = torch.sigmoid((self.fc1(Xe / self.lam)))
        X = A * Xe + (1 - A) * torch.tanh(self.fc2(Xp))
        # encoder
        A = self.fc2_att(torch.tanh(self.fc1_att(X)))
        P = torch.softmax(A, 1)
        X = torch.sum(P * X, 1)
        return self.dropout(X)


class Entity_Aware_Embedding(nn.Module, ABC):
    def __init__(self, pretrained_path, embedding_size, limit_size, pos_dim, drop_rate=0.5):
        super(Entity_Aware_Embedding, self).__init__()
        self.embedding = BertModel.from_pretrained(pretrained_path)
        for param in self.embedding.parameters():
            param.requires_grad = True

        self.pos1_embedding = nn.Embedding(2 * limit_size, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * limit_size, pos_dim)
        self.dropout = nn.Dropout(drop_rate)
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

    def forward(self, input_idxes, att_masks, e1_mask, e2_mask, pos1s, pos2s):
        Xs, _ = self.embedding(input_idxes, attention_mask=att_masks, output_all_encoded_layers=False)  # (bs, l, e)
        """word and pos embedding"""
        Xp0, Xp1 = self.pos1_embedding(pos1s), self.pos2_embedding(pos2s)  # (batch_size, max_len, embedding_size)
        Xp = torch.cat([Xs, Xp0, Xp1], dim=-1)   # (batch_size, max_len, embedding_size + 2 * pos_dim)
        """word and entity embedding"""
        Xe1 = Entity_Aware_Embedding.entities_pooling(Xs, e1_mask.unsqueeze(1).float()).unsqueeze(1).expand(Xs.shape)
        Xe2 = Entity_Aware_Embedding.entities_pooling(Xs, e2_mask.unsqueeze(1).float()).unsqueeze(1).expand(Xs.shape)
        Xe = torch.cat([Xs, Xe1, Xe2], dim=-1)  # (batch_size, max_len, embedding_size * 3)

        return self.dropout(Xp), self.dropout(Xe)

