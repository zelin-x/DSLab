"""
SENT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import transformers
import os
import random
import json

from model import Net
from data_loader import data_loader
from config import Config
from utils import calculate_metrics, get_id2label, calculate_avg_F, print_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def valid(test_loader, model):
    model.eval()
    pred_y = []
    true_y = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            labels = data[-1]
            out = model(data[1:-1])
            _, pred = torch.max(out, -1)
            pred, labels, _ = list(map(lambda x: x.bags.cpu().numpy().tolist(), [pred, labels, _]))
            pred_y += pred
            true_y += labels

    metric_dicts, cnt_dicts = calculate_metrics(true_y, pred_y, out.size(-1))

    return metric_dicts, cnt_dicts


def filter_and_relabel(model, _train_data_loader, id2label, cur_data_path, filter_data_path):
    model.eval()
    dynamic_threshold = 0.0
    sample_num = 0
    total_idxes, total_labels, total_pred, total_prob = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(_train_data_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            idxes, labels = data[0], data[-1]
            out = model(data[1:-1])
            out = F.softmax(out, dim=-1)
            prob, pred = torch.max(out, -1)
            total_idxes.append(idxes)
            total_labels.append(labels)
            total_pred.append(pred)
            total_prob.append(prob)
            dynamic_threshold += torch.sum(prob).item()
            sample_num += out.size(0)

    dynamic_threshold = dynamic_threshold / sample_num
    new_labels = {}
    relabel_cnt = 0
    for i in range(len(total_idxes)):
        idxes, labels, preds, probs = [_[i] for _ in (total_idxes, total_labels, total_pred, total_prob)]
        for j in range(len(idxes)):
            idx, label, pred, prob = list(map(lambda x: x[j].item(), (idxes, labels, preds, probs)))
            if prob >= dynamic_threshold:
                new_label = id2label[pred]
                if pred != label:
                    relabel_cnt += 1
            else:
                new_label = id2label[label]
            new_labels[idx] = new_label

    data_dict = json.load(open(cur_data_path, 'r', encoding='utf-8'))
    for k, v in new_labels.items():
        try:
            data_dict[str(k)]['relation'] = v
        except KeyError:
            continue

    with open(filter_data_path, 'w', encoding='utf-8')as f:
        json.dump(data_dict, f, ensure_ascii=False)

    print("Current Filter Threshold={0:.5f}\tRelabeled Count={1}".format(dynamic_threshold, relabel_cnt))
    print("Filtered data saved to", filter_data_path)


def main(opt):
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
    bert_params = set(model.embedding.parameters())
    other_params = list(set(model.parameters()) - bert_params)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [
        {'params': [p for n, p in model.embedding.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'lr': opt.bert_lr,
         'weight_decay': 1e-3},
        {'params': [p for n, p in model.embedding.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': 0.0,
         'weight_decay': 0.0},
        {'params': other_params,
         'lr': opt.lr,
         'weight_decay': 1e-3}
    ]
    optimizer = transformers.AdamW(param_optimizer, lr=opt.bert_lr, weight_decay=1e-3)

    train_data_loader = data_loader(opt.train_path, opt, shuffle=True)
    test_loader = data_loader(opt.test_path, opt, shuffle=False)

    updates_total = len(train_data_loader) // opt.batch_size * opt.epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=0.02 * updates_total,
                                                             num_training_steps=updates_total)
    not_best_count = 0
    best_F1 = -1
    best_epoch = -1
    id2label = get_id2label(opt.label_path)
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
            output = model(data[1:-1])
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            _, pred = torch.max(output, -1)
            # acc of all
            tp += (pred == labels).sum().item()
            # acc of not na
            pos_tot += (labels != 0).sum().item()
            pos_tp += ((pred == labels) & (labels != 0)).sum().item()
            # log
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
            metric_dicts, cnt_dicts = valid(test_loader, model)
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

        # filter and relabel
        if epoch < opt.M_epoch:
            continue
        cur_data_path = opt.train_path if epoch == opt.M_epoch else opt.filter_train_path
        filter_and_relabel(model, train_data_loader, id2label, cur_data_path, opt.filter_train_path)
        train_data_loader = data_loader(opt.filter_train_path, opt, shuffle=True)

    print('Finish training! The best epoch=' + str(best_epoch) + "The best F1=" + str(best_F1))


if __name__ == '__main__':
    opt = Config()
    main(opt)
