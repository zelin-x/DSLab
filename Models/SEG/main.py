"""
SEG
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import transformers
import os
import random

from model import SEG
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

    model = SEG(opt)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    if opt.use_plm:
        bert_params = set(model.e_a_embedding.embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.e_a_embedding.embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-3},
            {'params': [p for n, p in model.e_a_embedding.embedding.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': 0.0,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': opt.lr,
             'weight_decay': 1e-3}
        ]
        optimizer = transformers.AdamW(param_optimizer, lr=opt.bert_lr, weight_decay=1e-3)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    updates_total = len(train_data_loader) // opt.batch_size * opt.epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=0.02 * updates_total,
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
            labels = data[-1]
            output = model(data[:-1], infer=False)
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            _, pred = torch.max(output, -1)
            # acc of all
            tp += (pred == labels).sum().item()
            # acc of not na
            pos_tot += (labels != 0).sum().item()
            pos_tp += ((pred == labels) & (labels != 0)).sum().item()
            # Log
            if i % 500 == 0 or i == len(train_data_loader) - 1:
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
                # calculate_pr_curve_and_save(true_y, pred_y, pred_prob, opt.classes_num, opt.pr_curve_result_path, 0.02)
                not_best_count = 0
                best_epoch = epoch
            else:
                not_best_count += 1
            if not_best_count >= opt.patients:
                print("Early stop!")
                break
    print('Finish training! The best epoch=' + str(best_epoch) + "The best F1=" + str(best_F1))


def valid(test_loader, model, use_plm=True):
    model.eval()
    pred_y = []
    true_y = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            labels = data[-1]
            out = model(data[:-1], infer=False)
            _, pred = torch.max(out, -1)
            pred, labels, _ = list(map(lambda x: x.data.cpu().numpy().tolist(), [pred, labels, _]))
            pred_y += pred
            true_y += labels

    metric_dicts, cnt_dicts = calculate_metrics(true_y, pred_y, out.size(-1))

    return metric_dicts, cnt_dicts


if __name__ == '__main__':
    opt = Config()
    train_loader = data_loader(opt.train_path, opt, shuffle=True, training=True)
    test_loader = data_loader(opt.test_path, opt, shuffle=False, training=False)
    train(train_loader, test_loader, opt)
