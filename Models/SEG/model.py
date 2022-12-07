from abc import ABC

import torch
import torch.nn as nn
from module import PieceWiseCNN, SAN, Entity_Aware_Embedding


class SEG(nn.Module, ABC):
    def __init__(self, opt):
        super(SEG, self).__init__()
        self.e_a_embedding = Entity_Aware_Embedding(pretrained_path=opt.pretrained_path,
                                                    embedding_size=opt.embedding_size,
                                                    limit_size=opt.limit_size,
                                                    pos_dim=opt.pos_dim,
                                                    drop_rate=opt.dropout_rate)
        self.PCNN = PieceWiseCNN(opt.embedding_size, opt.pos_dim, opt.filter_size, opt.num_filters, opt.pcnn_lambda)
        self.SAN = SAN(opt.embedding_size, opt.pos_dim, opt.san_lambda)
        self.dropout = nn.Dropout(opt.dropout_rate)

        self.fc1 = nn.Linear(3 * opt.embedding_size, 3 * opt.embedding_size)
        self.fc2 = nn.Linear(3 * opt.embedding_size, 3 * opt.num_filters)
        self.classifier = nn.Linear(3 * opt.num_filters, opt.classes_num)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.classifier.bias)

    def selective_gate(self, S, U, scope=None, infer=False):
        G = torch.sigmoid(self.fc2(torch.tanh(self.fc1(U))))  # FFN
        X = G * S
        if infer:
            # return X
            return S
        B = []
        for s in scope:
            # B.append(torch.mean(X[s[0]:s[1]], dim=0))
            B.append(torch.mean(S[s[0]:s[1]], dim=0))
        B = torch.stack(B)
        return B

    def forward(self, Xs, infer=False):
        if not infer:
            scope, token_idxes, att_masks, e1_masks, e2_masks, pos1es, pos2es, pos_masks = Xs
        else:
            scope, (token_idxes, att_masks, e1_masks, e2_masks, pos1es, pos2es, pos_masks) = None, Xs
        Xp, Xe = self.e_a_embedding(token_idxes, att_masks, e1_masks, e2_masks, pos1es, pos2es)
        # Encode
        S = self.PCNN(Xp, Xe, pos_masks)
        U = self.SAN(Xp, Xe)
        # Combine
        X = self.selective_gate(S, U, scope, infer=infer)
        X = self.dropout(X)
        # Output
        out = self.classifier(X)
        return out
