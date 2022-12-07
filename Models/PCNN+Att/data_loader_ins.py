from codecs import open
from torch.utils.data import Dataset
import torch.utils.data as data
import torch
from collections import Counter
import pickle as pk
import numpy as np


class Dataset(data.Dataset):

    def __init__(self, filename, opt):
        super(Dataset, self).__init__()
        self.filename = filename
        self.label2id = Dataset.label2id(opt.label_path)
        self.use_plm = opt.use_plm
        self.max_len = opt.max_len
        self.limit_size = opt.limit_size
        self.data = []
        if self.use_plm:
            self.char2id = Dataset.build_bert_vocab(opt.pretrained_vocab_path)
            self._plm_preprocess()
        else:
            self.char2id = pk.load(open(opt.vocab_dict_path, "rb"))
            self._preprocess()

    @staticmethod
    def label2id(path):
        l2i = {}
        with open(path, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(f):
                lin = line.strip()
                l2i[lin.split()[0]] = int(lin.split()[-1])
        f.close()
        return l2i

    @staticmethod
    def build_bert_vocab(path):
        """Loads a vocabulary file into a dictionary."""
        vocab = {}
        index = 0
        with open(path, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def _preprocess(self):
        print("Loading data file...")
        with open(self.filename, 'r', encoding='UTF-8') as f:
            lines = [_.strip() for _ in f]
            for index, lin in enumerate(lines):
                if len(lin.split('\t')) == 3:
                    continue
                else:
                    e1, e2, r, sent = lin.split('\t')
                    e1_str, e1_pos = e1.split('&')
                    e2_str, e2_pos = e2.split('&')
                    e1b, e1e = int(e1_pos.split(':')[0]), int(e1_pos.split(':')[1])
                    e2b, e2e = int(e2_pos.split(':')[0]), int(e2_pos.split(':')[1])
                    if max(e1e, e2e) >= self.max_len:
                        continue

                    token_idx = [self.char2id.get(_, self.char2id['[UNK]']) for _ in sent]
                    att_mask = [1 for _ in range(len(token_idx))]
                    if len(token_idx) > self.max_len:
                        token_idx = token_idx[:self.max_len]
                        att_mask = att_mask[:self.max_len]
                        token_len = len(token_idx)
                    else:
                        att_mask = [1] + att_mask + [1]
                        token_len = len(token_idx)
                        token_idx += [self.char2id.get('[PAD]')] * (self.max_len - len(token_idx))
                        att_mask += [0] * (self.max_len - len(token_idx))

                    pos1s, pos2s = np.arange(self.max_len), np.arange(self.max_len)
                    pos1s[:e1b], pos1s[e1b:e1e + 1], pos1s[e1e + 1:] = np.arange(-e1b, 0), 0, np.arange(1,
                                                                                                        self.max_len - e1e)
                    pos2s[:e2b], pos2s[e2b:e2e + 1], pos2s[e2e + 1:] = np.arange(-e2b, 0), 0, np.arange(1,
                                                                                                        self.max_len - e2e)

                    pos1s += self.limit_size
                    pos2s += self.limit_size

                    pos_mask = np.array([0 for i in range(self.max_len)])
                    pos_mask[:min(e1e, e2e)] = 1
                    pos_mask[min(e1e, e2e):max(e1b, e2b)] = 2
                    pos_mask[max(e1b, e2b):token_len] = 3
                    pos_mask[token_len:] = 0

                    label = self.label2id[r]

                    assert len(token_idx) == self.max_len
                    assert len(att_mask) == self.max_len
                    assert len(pos1s) == self.max_len
                    assert len(pos2s) == self.max_len
                    assert len(pos_mask) == self.max_len

                    return token_idx, att_mask, pos1s, pos2s, pos_mask, label

    def _plm_preprocess(self):
        print("Loading data file...")
        with open(self.filename, 'r', encoding='UTF-8') as f:
            lines = [_.strip() for _ in f]
            for index, lin in enumerate(lines):
                if len(lin.split('\t')) == 3:
                    continue
                else:
                    e1, e2, r, sent = lin.split('\t')
                    e1_str, e1_pos = e1.split('&')
                    e2_str, e2_pos = e2.split('&')
                    e1b, e1e = int(e1_pos.split(':')[0]), int(e1_pos.split(':')[1])
                    e2b, e2e = int(e2_pos.split(':')[0]), int(e2_pos.split(':')[1])
                    if max(e1e, e2e) >= self.max_len - 2:
                        continue

                    token_idx = [self.char2id.get(_, self.char2id['[UNK]']) for _ in sent]
                    att_mask = [1 for _ in range(len(token_idx))]
                    if len(token_idx) > self.max_len - 2:
                        token_idx = token_idx[:self.max_len - 2]
                        att_mask = att_mask[:self.max_len - 2]
                        token_len = len(token_idx)
                        token_idx = [self.char2id.get('[CLS]')] + token_idx + [self.char2id.get('[SEP]')]
                        att_mask = [1] + att_mask + [1]
                    else:
                        token_idx = [self.char2id.get('[CLS]')] + token_idx + [self.char2id.get('[SEP]')]
                        att_mask = [1] + att_mask + [1]
                        token_len = len(token_idx)
                        token_idx += [self.char2id.get('[PAD]')] * (self.max_len - len(token_idx))
                        att_mask += [0] * (self.max_len - len(att_mask))

                    pos1s, pos2s = np.arange(self.max_len), np.arange(self.max_len)
                    pos1s[:e1b], pos1s[e1b:e1e + 1], pos1s[e1e + 1:] = np.arange(-e1b, 0), 0, np.arange(1,
                                                                                                        self.max_len - e1e)
                    pos2s[:e2b], pos2s[e2b:e2e + 1], pos2s[e2e + 1:] = np.arange(-e2b, 0), 0, np.arange(1,
                                                                                                        self.max_len - e2e)

                    pos1s += self.limit_size
                    pos2s += self.limit_size

                    pos_mask = np.array([0 for i in range(self.max_len)])
                    pos_mask[:min(e1e, e2e)] = 1
                    pos_mask[min(e1e, e2e):max(e1b, e2b)] = 2
                    pos_mask[max(e1b, e2b):token_len] = 3
                    pos_mask[token_len:] = 0

                    label = self.label2id[r]

                    assert len(token_idx) == self.max_len
                    assert len(att_mask) == self.max_len
                    assert len(pos1s) == self.max_len
                    assert len(pos2s) == self.max_len
                    assert len(pos_mask) == self.max_len

                    self.data.append((token_idx, att_mask, pos1s, pos2s, pos_mask, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _data = self.data[index]
        token_idxes, att_masks, pos1es, pos2es, pos_masks, labels = [torch.tensor(x, dtype=torch.long).unsqueeze(0) for x in _data]
        return token_idxes, att_masks, pos1es, pos2es, pos_masks, labels


def collate_fn(X):
    X = list(zip(*X))  # 解压
    token_idxes, att_masks, pos1es, pos2es, pos_masks, label = X

    token_idxes = torch.cat(token_idxes, 0)
    att_masks = torch.cat(att_masks, 0)
    pos1es = torch.cat(pos1es, 0)
    pos2es = torch.cat(pos2es, 0)
    pos_masks = torch.cat(pos_masks, 0)
    labels = torch.stack(label)

    return token_idxes, att_masks, pos1es, pos2es, pos_masks, labels.unsqueeze(1)


def instance_loader(data_file, opt, shuffle, num_workers=0):
    dataset = Dataset(data_file, opt)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader
