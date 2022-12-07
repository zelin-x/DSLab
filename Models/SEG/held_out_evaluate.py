import torch
import torch.nn.functional as F
from config import Config
from data_loader_ins import instance_loader
from utils import calculate_pr_curve_and_save
from model import SEG

import numpy as np
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def held_out_evaluation(test_loader, model, opt):
    """held-out-evaluation and draw pr-curve"""
    model.eval()
    golden_list = []
    pred_prob_list = []
    total, tp, pos_total, pos_tp = 0, 0, 0, 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            labels = data[-1]
            out = model(data[:-1], infer=True)
            prob = F.softmax(out, 1)      # (batch_size, classes_num)
            _, pred = torch.max(out, dim=1)

            total += len(labels)
            tp += (labels == pred).sum().item()
            pos_total += (labels != 0).sum().item()
            pos_tp += ((labels == pred) & (labels != 0)).sum().item()

            golden_list.append(labels)
            pred_prob_list.append(prob)
    # calculate_pr_curve_and_save(golden_list, pred_prob_list, opt.classes_num, opt.pr_curve_result_path, 0.02)
    acc = tp / total if total != 0 else 1.
    pos_acc = pos_tp / pos_total if total != 0 else 1.

    return acc, pos_acc


if __name__ == '__main__':
    opt = Config()
    model_name = "model_2022_11_14_16_27_55.ckpt"
    opt.model_name = model_name
    opt.model_path = "checkpoints/" + model_name
    opt.pr_curve_result_path = "result/" + model_name + ".txt"
    opt.batch_size = 32

    instance_loader = instance_loader(opt.test_path, opt, shuffle=False)
    model = SEG(opt)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(opt.model_path))
    acc, pos_acc = held_out_evaluation(instance_loader, model, opt)
    print("acc={0:.5f} pos_acc={1:.5f}".format(acc, pos_acc))
