from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class SentenceEncoder(nn.Module, ABC):
    def __init__(self, use_plm, pretrained_path, vocab_size, embedding_size,
                 limit_size, pos_dim, filter_size, num_filters, dropout_rate=0.5):
        super(SentenceEncoder, self).__init__()
        if use_plm:
            self.embedding = BertModel.from_pretrained(pretrained_path)
            for param in self.embedding.parameters():
                param.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            nn.init.xavier_uniform_(self.embedding.weight)

        self.pos1_embedding = nn.Embedding(2 * limit_size, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * limit_size, pos_dim)
        self.P_CNN = PieceWiseCNN(embedding_size + 2 * pos_dim, filter_size, num_filters)
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)

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

        return out


class PCNN_Att(nn.Module, ABC):
    def __init__(self, opt):
        super(PCNN_Att, self).__init__()
        self.sentence_encoder = SentenceEncoder(opt.use_plm, opt.pretrained_path, opt.vocab_size, opt.embedding_size,
                                                opt.limit_size, opt.pos_dim, opt.filter_size, opt.num_filters,
                                                opt.dropout_rate)
        self.rel_embedding = nn.Embedding(opt.classes_num, opt.num_filters * 3)
        # self.dropout = nn.Dropout(opt.dropout_rate)
        self.classes_num = opt.classes_num
        self.classifier = nn.Linear(opt.num_filters * 3, opt.classes_num)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.rel_embedding.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def selective_attention(query, key, training=True):
        if training:
            query = query.view(1, -1)
        alphas = torch.matmul(query, key.T) / math.sqrt(query.size(-1))
        alphas = F.softmax(alphas, dim=-1)
        value = torch.matmul(alphas, key)
        return value

    def forward(self, Xs, training=True):
        scope, token_idxes, att_masks, pos1es, pos2es, pos_masks, labels = Xs
        if training:
            """training:has relation label for query"""
            bags_feature = []
            for idx, s in enumerate(scope):
                w, m, p1, p2, pm = list(map(lambda x: x[s[0]:s[1]], [_ for _ in Xs[1:-1]]))
                rel = labels[idx]
                ins_features = self.sentence_encoder(w, m, p1, p2, pm)  # (bag_instance_num, feature_size)
                rel_embed = self.rel_embedding(rel)  # (feature_size, )
                _bags_feature = PCNN_Att.selective_attention(rel_embed, ins_features, training=training)    # (1, feature_size)
                bags_feature.append(_bags_feature)
            out = torch.cat(bags_feature, dim=0)
            # out = self.dropout(batch_bags_feature)
            out = self.classifier(out)
        else:
            """inference:has no relation label for query"""
            logits = []
            for idx, s in enumerate(scope):
                w, m, p1, p2, pm = list(map(lambda x: x[s[0]:s[1]], [_ for _ in Xs[1:-1]]))
                ins_features = self.sentence_encoder(w, m, p1, p2, pm)  # (bag_instance_num, feature_size)
                rels_embed = self.rel_embedding(torch.arange(self.classes_num).cuda())  # (classes_num, feature_size)
                _bags_feature = PCNN_Att.selective_attention(rels_embed, ins_features, training=training)  # (classes_num, feature_size)
                _out = self.classifier(_bags_feature)   # (classes_num, classes_num)
                _out = F.softmax(_out, dim=-1)          # line first
                _out = _out * torch.eye(self.classes_num, dtype=torch.float32).cuda()
                logit = torch.sum(_out, dim=-1)
                logits.append(logit.view(1, -1))
            out = torch.cat(logits, dim=0)
        return out

    def test_instance(self, Xs):
        token_idxes, att_masks, pos1es, pos2es, pos_masks = Xs
        out = self.sentence_encoder(token_idxes, att_masks, pos1es, pos2es, pos_masks)
        out = self.classifier(out)
        return out
