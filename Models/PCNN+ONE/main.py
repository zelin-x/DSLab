"""
PCNN + ONE
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch.nn.functional as F
import numpy as np
import transformers
import os
import random

from model import PCNN_ONE
from data_loader import data_loader
from config import Config
from utils import calculate_metrics, print_metrics, get_id2label, calculate_avg_F

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def train(train_data_loader, test_loader, opt):
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    model = PCNN_ONE(opt)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    if opt.use_plm:
        bert_params = set(model.embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-5},
            {'params': [p for n, p in model.embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.lr,
             'weight_decay': 1e-5}
        ]
        optimizer = transformers.AdamW(param_optimizer, lr=opt.bert_lr, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    updates_total = len(train_data_loader) // opt.batch_size * opt.epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=0.2 * updates_total,
                                                             num_training_steps=updates_total)
    not_best_count = 0
    best_F1 = -1
    best_epoch = -1
    for epoch in range(opt.epochs):
        model.train()
        print("\n=== Epoch %d train ===" % epoch)
        epoch_loss = 0.0
        tp = 0
        pos_tot = 0
        pos_tp = 0
        for i, data in enumerate(train_data_loader):
            if torch.cuda.is_available():
                for d in range(len(data)):
                    data[d] = data[d].cuda()

            token_idxes, att_masks, pos1es, pos2es, pos_masks, labels = select_instance(model, data, opt.use_plm)
            output = model(token_idxes, att_masks, pos1es, pos2es, pos_masks, opt.use_plm)
            loss = criterion(output, labels)
            epoch_loss += loss.item()

            _, pred = torch.max(output, -1)
            # acc of all
            tp += (pred == labels).sum().item()
            # acc of not na
            pos_tot += (labels != 0).sum().item()
            pos_tp += ((pred == labels) & (labels != 0)).sum().item()
            # Log
            sys.stdout.write('\rstep: {0} / {1} | loss: {2:.5f}, acc: {3:.5f}, pos_acc: {4:.5f}'.
                             format(i + 1,
                                    len(train_data_loader),
                                    epoch_loss / (i + 1) * opt.batch_size,
                                    tp / ((i + 1) * opt.batch_size),
                                    pos_tp / pos_tot if pos_tot != 0 else 0.0)
                             )
            sys.stdout.flush()
            # Optimize
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (pos_tp / pos_tot) >= 0.5:
            print("\n=== Epoch %d val ===" % epoch)
            metric_dicts, cnt_dicts = valid(test_loader, model, opt.use_plm)
            id2label = get_id2label(opt.label_path)
            print_metrics(metric_dicts, id2label)
            micro_f, pos_micro_f = calculate_avg_F(cnt_dicts)
            print("MICRO F1={0:.5f}, POS_MICRO_F1={1:.5f}".format(micro_f, pos_micro_f))
            if pos_micro_f > best_F1:
                print("Best result!")
                best_F1 = pos_micro_f
                torch.save(model.state_dict(), opt.save_model_path)
                not_best_count = 0
                best_epoch = epoch
            else:
                not_best_count += 1
            if not_best_count >= opt.patients:
                print("Early stop!")
                break
    print('Finish training! The best epoch=' + str(best_epoch) + "The best F1=" + str(best_F1))


def select_instance(model, batch_data, use_plm=True):
    model.eval()
    scope, token_idxes, att_masks, pos1es, pos2es, pos_masks, labels = batch_data
    n_token_idxes, n_att_masks, n_pos1es, n_pos2es, n_pos_maks, n_labels = (), (), (), (), (), ()
    for idx, s in enumerate(scope):
        w = token_idxes[s[0]:s[1]]
        m = att_masks[s[0]:s[1]]
        p1 = pos1es[s[0]:s[1]]
        p2 = pos2es[s[0]:s[1]]
        pm = pos_masks[s[0]:s[1]]
        label = labels[idx]
        out = model(w, m, p1, p2, pm, use_plm)

        max_ins_id = torch.max(out[:, label], 0)[1].item()  # 返回最大的为label概率的instance索引

        n_token_idxes += (w[max_ins_id],)
        n_att_masks += (m[max_ins_id],)
        n_pos1es += (p1[max_ins_id],)
        n_pos2es += (p2[max_ins_id],)
        n_pos_maks += (pm[max_ins_id],)

    token_idxes = torch.stack(n_token_idxes, 0)
    att_masks = torch.stack(n_att_masks, 0)
    pos1es = torch.stack(n_pos1es, 0)
    pos2es = torch.stack(n_pos2es, 0)
    pos_masks = torch.stack(n_pos_maks, 0)

    model.train()

    return token_idxes, att_masks, pos1es, pos2es, pos_masks, labels


def valid(test_loader, model, use_plm=True):
    model.eval()
    pred_y = []
    true_y = []
    pred_prob = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            scope, token_idxes, att_masks, pos1es, pos2es, pos_masks, labels = data
            for idx, s in enumerate(scope):
                w = token_idxes[s[0]:s[1]]
                m = att_masks[s[0]:s[1]]
                p1 = pos1es[s[0]:s[1]]
                p2 = pos2es[s[0]:s[1]]
                pm = pos_masks[s[0]:s[1]]
                out = model(w, m, p1, p2, pm, use_plm)
                out = F.softmax(out, 1)
                max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
                max_ins_prob = max_ins_prob.tolist()
                max_ins_label = max_ins_label.tolist()
                max_pred_prob = max(max_ins_prob)
                max_y_hat = max_ins_label[max_ins_prob.index(max_pred_prob)]

                pred_y.append(max_y_hat)
                true_y.append(labels[idx].item())
                pred_prob.append(max_pred_prob)

    metric_dicts, cnt_dicts = calculate_metrics(true_y, pred_y, out.size(-1))

    return metric_dicts, cnt_dicts


if __name__ == '__main__':
    opt = Config()
    train_loader = data_loader(opt.train_path, opt, shuffle=True, training=True)
    test_loader = data_loader(opt.test_path, opt, shuffle=False, training=False)
    train(train_loader, test_loader, opt)
