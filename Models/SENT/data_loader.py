from codecs import open
from torch.utils.data import Dataset
import torch.utils.data as data
import torch
import numpy as np
import json
from utils import get_label2id, build_bert_vocab


class Dataset(data.Dataset):

    def __init__(self, filename, opt):
        super(Dataset, self).__init__()
        self.filename = filename
        self.label2id = get_label2id(opt.label_path)
        self.max_len = opt.max_len
        self.limit_size = opt.limit_size
        self.char2id = build_bert_vocab(opt.pretrained_vocab_path)

        self.data = []
        self._plm_preprocess()

    def _plm_preprocess(self):
        print("Loading data file...")
        with open(self.filename, 'r', encoding='utf-8')as f:
            dicts = json.load(f)
            for idx, dic in dicts.items():
                sent = dic['text']
                r = dic['relation']
                e1b, e1e = dic['head']['begin'], dic['head']['end']
                e2b, e2e = dic['tail']['begin'], dic['tail']['end']
                if max(e1e, e2e) >= self.max_len - 2:
                    continue

                token_idx = [self.char2id.get(_, self.char2id['[UNK]']) for _ in sent]
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
                pos1s[:e1b+1], pos1s[e1b+1:e1e+2], pos1s[e1e+2:] = np.arange(-e1b-1, 0), 0, np.arange(1, self.max_len - e1e - 1)
                pos2s[:e2b+1], pos2s[e2b+1:e2e+2], pos2s[e2e+2:] = np.arange(-e2b-1, 0), 0, np.arange(1, self.max_len - e2e - 1)

                head_mask, tail_mask = np.zeros(self.max_len), np.zeros(self.max_len)
                head_mask[e1b+1:e1e+2] = 1
                tail_mask[e2b+1:e2e+2] = 1

                pos1s += self.limit_size
                pos2s += self.limit_size

                label = self.label2id[r]

                assert len(token_idx) == self.max_len
                assert len(att_mask) == self.max_len
                assert len(pos1s) == self.max_len
                assert len(pos2s) == self.max_len
                assert len(head_mask) == self.max_len
                assert len(tail_mask) == self.max_len

                self.data.append((int(idx), token_idx, att_mask, pos1s, pos2s, head_mask, tail_mask, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _data = self.data[index]
        token_idxes, att_masks, pos1es, pos2es, head_mask, tail_mask = \
            [torch.tensor(x, dtype=torch.long).unsqueeze(0) for x in _data[1:-1]]
        idx = torch.tensor(_data[0], dtype=torch.long)
        label = torch.tensor(_data[-1], dtype=torch.long)
        return idx, token_idxes, att_masks, pos1es, pos2es, head_mask, tail_mask, label


def collate_fn(X):
    X = list(zip(*X))  # 解压
    idxes, token_idxes, att_masks, pos1es, pos2es, head_masks, tail_masks, labels = X
    token_idxes = torch.cat(token_idxes, 0)
    att_masks = torch.cat(att_masks, 0)
    pos1es = torch.cat(pos1es, 0)
    pos2es = torch.cat(pos2es, 0)
    head_masks = torch.cat(head_masks, 0)
    tail_masks = torch.cat(tail_masks, 0)
    idxes = torch.stack(idxes)
    labels = torch.stack(labels)

    return idxes, token_idxes, att_masks, pos1es, pos2es, head_masks, tail_masks, labels


def data_loader(data_file, opt, shuffle, num_workers=0):
    dataset = Dataset(data_file, opt)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return loader
