"""
PCNN + ATT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
import transformers
import os
import random
from sklearn.metrics import auc

from model import PCNN_Att
from data_loader import data_loader
from config import Config
from utils import calculate_metrics, print_metrics, get_id2label, calculate_avg_F, calculate_pr_curve_and_save

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train(train_data_loader, test_loader, opt):
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    model = PCNN_Att(opt)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    if opt.use_plm:
        bert_params = set(model.sentence_encoder.embedding.parameters())
        other_params = list(set(model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = [
            {'params': [p for n, p in model.sentence_encoder.embedding.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': opt.bert_lr,
             'weight_decay': 1e-5},
            {'params': [p for n, p in model.sentence_encoder.embedding.named_parameters() if any(nd in n for nd in no_decay)],
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
    best_auc = -1
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
            output = model(data, training=True)
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
            id2rel = get_id2label(opt.label_path)
            result = eval(test_loader, model, id2rel)
            p = result['prec']
            print(
                "auc: %.4f f1: %.4f \n p@100: %.4f p@200: %.4f p@300: %.4f p@500: %.4f p@1000: %.4f p@2000: %.4f" % (
                    result['auc'], result['f1'], p[100], p[200], p[300], p[500], p[1000], p[2000]))
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


def valid(test_loader, model, use_plm=True):
    model.eval()
    pred_y = []
    true_y = []
    pred_prob = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            labels = data[-1]
            out = model(data, training=False)  # already softmax!
            _, pred = torch.max(out, -1)
            pred, labels, _ = list(map(lambda x: x.bags.cpu().numpy().tolist(), [pred, labels, _]))
            pred_y += pred
            true_y += labels
            pred_prob += _

    metric_dicts, cnt_dicts = calculate_metrics(true_y, pred_y, out.size(-1))

    return metric_dicts, cnt_dicts


def eval(test_loader, model, id2rel, use_plm=True):
    model.eval()
    pred_result = []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            ent_pairs = data[-1]
            if torch.cuda.is_available():
                data = [x.cuda() for x in data[:-1]]
            logits = model(data, training=False)
            class_num = logits.size(-1)
            logits = logits.cpu().numpy()

            for i in range(len(logits)):
                for rid in range(class_num):
                    if rid != 0:
                        pred_result.append({
                            'ent_pair': ent_pairs[i],
                            'relation': id2rel[rid],
                            'score': logits[i][rid]
                        })

    sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
    prec, rec = [], []
    correct = 0
    facts = test_loader.dataset.facts
    total = len(facts)
    for i, item in enumerate(sorted_pred_result):
        if (item['ent_pair'][0], item['ent_pair'][1], item['relation']) in facts:
            correct += 1
        prec.append(float(correct) / float(i + 1))
        rec.append(float(correct) / float(total))

    _auc = auc(x=rec, y=prec)
    np_prec = np.array(prec)
    np_rec = np.array(rec)
    f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
    mean_prec = np_prec.mean()
    result = {'prec': np_prec, 'rec': np_rec, 'mean_prec': mean_prec, 'f1': f1, 'auc': _auc}

    return result


if __name__ == '__main__':
    opt = Config()
    train_loader = data_loader(opt.train_path, opt, shuffle=True, training=True)
    test_loader = data_loader(opt.test_path, opt, shuffle=False, training=False)
    train(train_loader, test_loader, opt)
