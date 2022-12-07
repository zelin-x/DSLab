import torch
import torch.nn.functional as F
from config import Config
from data_loader_ins import instance_loader
from utils import calculate_pr_curve_and_save
from model import PCNN_Att

import numpy as np
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def held_out_evaluation(test_loader, model, opt):
    """held-out-evaluation and draw pr-curve"""
    model.eval()
    true_y = []
    pred_prob = []
    total, tp, pos_total, pos_tp = 0, 0, 0, 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            out = model.test_instance(data[:-1])
            prob = F.softmax(out, 1)      # (batch_size, classes_num)
            _, pred = torch.max(out, dim=1)

            total += len(labels)
            tp += (labels == pred).sum().item()
            pos_total += (labels != 0).sum().item()
            pos_tp += ((labels == pred) & (labels != 0)).sum().item()

            pred, labels = [x.data.cpu().numpy().tolist() for x in (pred, labels)]
            true_y += labels
            pred_prob.append(prob)
    pred_prob = torch.cat(pred_prob, dim=0).data.cpu().numpy()
    calculate_pr_curve_and_save(np.array(true_y), pred_prob, opt.classes_num, opt.pr_curve_result_path, 0.02)
    acc = tp / total if total != 0 else 1.
    pos_acc = pos_tp / pos_total if total != 0 else 1.

    return acc, pos_acc


if __name__ == '__main__':
    opt = Config()
    model_name = "######"
    opt.model_name = model_name
    opt.model_path = "checkpoints/" + model_name
    opt.pr_curve_result_path = "result/" + model_name + ".txt"

    instance_loader = instance_loader(opt.test_path, opt, shuffle=False)
    model = PCNN_Att(opt)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(opt.model_path))
    acc, pos_acc = held_out_evaluation(instance_loader, model, opt)
    print("acc={0:.5f} pos_acc={1:.5f}".format(acc, pos_acc))
