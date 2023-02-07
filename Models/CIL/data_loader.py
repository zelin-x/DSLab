from codecs import open

import sklearn
from torch.utils.data import Dataset
import torch.utils.data as data
import torch
import numpy as np
import json
import random
from utils import get_label2id, build_bert_vocab


class Dataset(data.Dataset):

    def __init__(self, filename, opt, training=True):
        super(Dataset, self).__init__()
        self.training = training
        self.filename = filename
        self.max_bag_size = opt.max_bag_size
        self.label2id = get_label2id(opt.label_path)
        self.max_len = opt.max_len
        self.limit_size = opt.limit_size
        self.char2id = build_bert_vocab(opt.pretrained_vocab_path)
        self.bags = []
        self.bag_names = []
        self.facts = {}
        self._preprocess()

    def re_tokenize(self, text, e1b, e1e, e2b, e2e):
        token_idx = [self.char2id.get(_, self.char2id['[UNK]']) for _ in text]
        att_mask = [1 for _ in range(len(token_idx))]
        if len(token_idx) > self.max_len - 2:
            token_idx = token_idx[:self.max_len - 2]
            att_mask = att_mask[:self.max_len - 2]
            token_idx = [self.char2id.get('[CLS]')] + token_idx + [self.char2id.get('[SEP]')]
            att_mask = [1] + att_mask + [1]
        else:
            token_idx = [self.char2id.get('[CLS]')] + token_idx + [self.char2id.get('[SEP]')]
            att_mask = [1] + att_mask + [1]
            token_idx += [self.char2id.get('[PAD]')] * (self.max_len - len(token_idx))
            att_mask += [0] * (self.max_len - len(att_mask))

        pos1s, pos2s = np.arange(self.max_len), np.arange(self.max_len)
        pos1s[:e1b + 1], pos1s[e1b + 1:e1e + 2], pos1s[e1e + 2:] = np.arange(-e1b - 1, 0), \
                                                                   0, \
                                                                   np.arange(1, self.max_len - e1e - 1)
        pos2s[:e2b + 1], pos2s[e2b + 1:e2e + 2], pos2s[e2e + 2:] = np.arange(-e2b - 1, 0), \
                                                                   0, \
                                                                   np.arange(1, self.max_len - e2e - 1)

        head_mask, tail_mask = np.zeros(self.max_len), np.zeros(self.max_len)
        head_mask[e1b + 1:e1e + 2] = 1
        tail_mask[e2b + 1:e2e + 2] = 1

        pos1s += self.limit_size
        pos2s += self.limit_size

        return token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask

    def _preprocess(self):
        print("Loading data file...")
        with open(self.filename, 'r', encoding='utf-8')as f:
            dicts = json.load(f)
            for idx, dic in dicts.items():
                text = dic['text']
                e1b, e1e = dic['head']['begin'], dic['head']['end']
                e2b, e2e = dic['tail']['begin'], dic['tail']['end']
                if max(e1e, e2e) >= self.max_len - 2:
                    continue
                token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask = self.re_tokenize(text, e1b, e1e, e2b, e2e)
                r = dic['relation']
                label = self.label2id[r]
                fact = (dic['head']['text'], dic['tail']['text'], r)

                if self.training:
                    aug_text = dic['aug_text']
                    aug_e1b, aug_e1e = dic['head']['aug_beg'], dic['head']['aug_end']
                    aug_e2b, aug_e2e = dic['tail']['aug_beg'], dic['tail']['aug_end']
                    if max(aug_e1e, aug_e2e) >= self.max_len - 2:
                        continue
                    aug_token_idx, aug_att_mask, aug_pos1s, aug_pos2s, aug_head_mask, aug_tail_mask = self.re_tokenize(
                        aug_text, aug_e1b, aug_e1e, aug_e2b, aug_e2e)
                    if fact in self.facts:
                        self.bags[self.facts[fact]].append([token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask,
                                                            aug_token_idx, aug_att_mask, aug_pos1s, aug_pos2s,
                                                            aug_head_mask, aug_tail_mask,
                                                            label])
                    else:
                        self.facts[fact] = len(self.facts)
                        self.bags.append([[token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask,
                                           aug_token_idx, aug_att_mask, aug_pos1s, aug_pos2s, aug_head_mask,
                                           aug_tail_mask,
                                           label]])
                else:
                    if fact in self.facts:
                        self.bags[self.facts[fact]].append(
                            [token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask, label])
                    else:
                        self.facts[fact] = len(self.facts)
                        self.bag_names.append(fact)
                        self.bags.append([[token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask, label]])

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = bag[0][-1]
        if self.training:
            bag_size = len(bag)
            if bag_size >= self.max_bag_size:
                np.random.shuffle(bag)
                resize_bag = bag[:self.max_bag_size]
            else:
                resize_bag = []
                while len(resize_bag) < self.max_bag_size:
                    resize_bag.append(bag[np.random.randint(0, bag_size)])
            bag = resize_bag
            bag = [ins[:-1] for ins in bag]   # without label
            bag = list(zip(*bag))
            token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask, \
            aug_token_idx, aug_att_mask, aug_pos1s, aug_pos2s, aug_head_mask, aug_tail_mask = [
                torch.tensor(d, dtype=torch.long).unsqueeze(0) for d in bag]
            label = torch.tensor(label, dtype=torch.long)
            return token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask, aug_token_idx, aug_att_mask, aug_pos1s, aug_pos2s, aug_head_mask, aug_tail_mask, label
        else:
            bag = list(zip(*bag))
            ent_pair = (self.bag_names[idx][0], self.bag_names[idx][1])
            token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask = [torch.tensor(d, dtype=torch.long) for d in bag[:-1]]
            label = torch.tensor(label, dtype=torch.long)
            return token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask, label, ent_pair

    def eval(self, pred_result):
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
        prec, rec = [], []
        correct = 0
        total = len(self.facts)
        for i, item in enumerate(sorted_pred_result):
            if (item['entpair'][0], item['entpair'][1], self.rel2id[item['relation']]) in self.facts:
                correct += 1
            prec.append(float(correct) / float(i + 1))
            rec.append(float(correct) / float(total))
        auc = sklearn.metrics.auc(x=rec, y=prec)
        np_prec = np.array(prec)
        np_rec = np.array(rec)
        f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
        mean_prec = np_prec.mean()
        return {'prec': np_prec, 'rec': np_rec, 'mean_prec': mean_prec, 'f1': f1, 'auc': auc}


def train_collate_fn(X):
    X = list(zip(*X))  # 解压
    scope = []  # 用来分包
    ind = 0
    for w in X[0]:
        scope.append((ind, ind + w.size(1)))
        ind += w.size(1)
    scope = torch.tensor(scope, dtype=torch.long)
    input1, input2, labels = X[:6], X[6:-1], X[-1]
    input1 = [torch.cat(_, 0) for _ in input1]
    input2 = [torch.cat(_, 0) for _ in input2]
    labels = torch.stack(labels)
    return scope, input1, input2, labels


def eval_collate_fn(X):
    X = list(zip(*X))  # 解压
    scope = []  # 用来分包
    ind = 0
    for w in X[0]:
        scope.append((ind, ind + len(w)))
        ind += len(w)
    scope = torch.tensor(scope, dtype=torch.long)
    input, labels, ent_pairs = X[:6], X[-2], X[-1]
    input = [torch.cat(_) for _ in input]
    labels = torch.stack(labels)
    return scope, input, labels, ent_pairs


def data_loader(data_file, opt, shuffle=True, training=True, num_workers=0):
    dataset = Dataset(data_file, opt, training)
    collate_fn = train_collate_fn if training else eval_collate_fn
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader
