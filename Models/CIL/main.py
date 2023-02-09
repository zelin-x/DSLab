"""
CIL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import transformers
import os
import random
from sklearn.metrics import average_precision_score

from model import Net
from data_loader import data_loader
from config import Config
from utils import get_id2label

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def train(train_loader, test_loader, opt):
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    model = Net(opt)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    bert_params = set(model.sent_encoder.embedding.parameters())
    other_params = list(set(model.parameters()) - bert_params)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [
        {'params': [p for n, p in model.sent_encoder.embedding.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'lr': opt.bert_lr,
         'weight_decay': 1e-5},
        {'params': [p for n, p in model.sent_encoder.embedding.named_parameters() if
                    any(nd in n for nd in no_decay)],
         'lr': 0.0,
         'weight_decay': 0.0},
        {'params': other_params,
         'lr': opt.lr,
         'weight_decay': 1e-5}
    ]
    optimizer = transformers.AdamW(param_optimizer, lr=opt.bert_lr, weight_decay=1e-5)

    updates_total = len(train_loader) // opt.batch_size * opt.epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=0.2 * updates_total,
                                                             num_training_steps=updates_total)
    not_best_count = 0
    best_auc = -1
    best_epoch = -1
    for epoch in range(opt.epochs):
        model.train()
        print("\n=== Epoch %d train ===" % epoch)
        epoch_loss = 0.0
        tp = 0
        pos_tot = 0
        pos_tp = 0
        global_step = 0  # warmup
        for i, data in enumerate(train_loader):
            if torch.cuda.is_available():
                scope, input1, input2, labels = data
                scope = scope.cuda()
                input1 = [_.cuda() for _ in input1]
                input2 = [_.cuda() for _ in input2]
                labels = labels.cuda()
            output, cl_loss = model(scope, input1, input2, labels, training=True, temperature=opt.temperature)
            cl_loss = cl_loss.mean()
            ce_loss = criterion(output, labels)

            global_step += 1
            warmup_steps = updates_total * 0.2
            p = float(global_step) / (warmup_steps * 2.0)
            alpha = 2. / (1. + np.exp(-2. * p)) - 1
            loss = ce_loss + alpha * cl_loss * 10.0

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
                                    len(train_loader),
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
            id2rel = get_id2label(opt.label_path)
            result = eval(test_loader, model, id2rel)
            p = result['prec']
            print(
                "auc: %.4f \n p@100: %.4f p@200: %.4f p@300: %.4f pmean:%.4f" %
                (result['auc'], result['p100'], result['p200'], result['p300'], result['pmean'])
            )
            if result['auc'] > best_auc:
                print("Best result!")
                best_auc = result['auc']
                torch.save(model.state_dict(), opt.save_model_path)
                np.save(opt.prec_save_path, result['prec'])
                np.save(opt.rec_save_path, result['rec'])
                not_best_count = 0
                best_epoch = epoch
            else:
                not_best_count += 1
            if not_best_count >= opt.patients:
                print("Early stop!")
                break
    print('Finish training! The best epoch=' + str(best_epoch) + "The best auc=" + str(best_auc))


def eval(test_loader, model, id2rel):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            scope, input, labels = data
            if torch.cuda.is_available():
                scope = scope.cuda()
                input = [_.cuda() for _ in input]
            logits = model(scope, input1=input, training=False)   # (batch_size, class_num) already softmax
            y_true.append(labels[:][1:])
            y_pred.append(logits[:][1:])
        y_true = torch.cat(y_true).reshape(-1).detach().cpu().numpy()
        y_pred = torch.cat(y_pred).reshape(-1).detach().cpu().numpy()

    # AUC
    auc = average_precision_score(y_true, y_pred)  # PR-AUC

    # P@N
    order = np.argsort(-y_pred)
    p100 = (y_true[order[:100]]).mean() * 100
    p200 = (y_true[order[:200]]).mean() * 100
    p300 = (y_true[order[:300]]).mean() * 100
    pmean = (p100 + p200 + p300) / 3

    # PR
    order = np.argsort(y_pred)[::-1]
    correct = 0.
    total = y_true.sum()
    precision = []
    recall = []
    for i, o in enumerate(order):
        correct += y_true[o]
        precision.append(float(correct) / (i + 1))
        recall.append(float(correct) / total)
    precision = np.array(precision)
    recall = np.array(recall)
    result = {'prec': precision, 'rec': recall, 'p100': p100, 'p200': p200, 'p300': p300, 'pmean': pmean,
              'auc': auc}

    return result


if __name__ == '__main__':
    opt = Config()
    train_loader = data_loader(opt.train_path, opt, shuffle=True, training=True)
    test_loader = data_loader(opt.test_path, opt, shuffle=False, training=False)
    train(train_loader, test_loader, opt)
