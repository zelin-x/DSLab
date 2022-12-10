from tqdm import tqdm


def get_label2id(label_path):
    l2i = {}
    with open(label_path, 'r', encoding='utf-8')as f:
        for line in f:
            lin = line.strip()
            if lin:
                lin = lin.split()
                l2i[lin[0]] = int(lin[1])
    return l2i


def get_id2label(label_path):
    i2l = {}
    with open(label_path, 'r', encoding='utf-8')as f:
        for line in f:
            lin = line.strip()
            if lin:
                lin = lin.split()
                i2l[int(lin[1])] = lin[0]
    return i2l


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


def calculate_metrics(golden_list, pred_list, classes_num):
    cnt_dicts = {}
    metric_dicts = {}
    for _ in range(classes_num):
        cnt_dicts[_] = {'tp': 0, 'fp': 0, 'fn': 0}
        metric_dicts[_] = {'R': 0.0, 'P': 0.0, 'F': 0.0, 'Support': 0}

    for i in range(len(golden_list)):
        if golden_list[i] == pred_list[i]:
            cnt_dicts[golden_list[i]]['tp'] += 1
        else:
            cnt_dicts[golden_list[i]]['fn'] += 1
            cnt_dicts[pred_list[i]]['fp'] += 1

    for k, v in cnt_dicts.items():
        recall = v['tp'] * 1.0 / (v['tp'] + v['fn']) if v['tp'] + v['fn'] != 0 else 0.0
        precision = v['tp'] * 1.0 / (v['tp'] + v['fp']) if v['tp'] + v['fp'] != 0 else 0.0
        metric_dicts[k]['R'] = recall
        metric_dicts[k]['P'] = precision
        metric_dicts[k]['F'] = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0.0
        metric_dicts[k]['Support'] = v['tp'] + v['fn']

    return metric_dicts, cnt_dicts


def calculate_avg_F(cnt_dicts):
    """micro"""
    tp, fp, fn = 0, 0, 0
    na_tp, na_fp, na_fn = 0, 0, 0
    for k, v in cnt_dicts.items():
        if k == 0:
            na_tp += v['tp']
            na_fp += v['fp']
            na_fn += v['fn']
        tp += v['tp']
        fp += v['fp']
        fn += v['fn']
    pos_tp, pos_fp, pos_fn = tp - na_tp, fp - na_fp, fn - na_fn
    r = tp / (tp + fn) if tp + fn != 0 else 0.0
    p = tp / (tp + fp) if tp + fp != 0 else 0.0
    micro_f = 2 * p * r / (p + r) if p + r != 0 else 0.0

    r = pos_tp / (pos_tp + pos_fn) if pos_tp + pos_fn != 0 else 0.0
    p = pos_tp / (pos_tp + pos_fp) if pos_tp + pos_fp != 0 else 0.0
    pos_micro_f = 2 * p * r / (p + r) if p + r != 0 else 0.0

    return micro_f, pos_micro_f


def print_metrics(metric_dicts, id2label: dict):
    print("\n" + "=" * 2 + "Evaluating Result" + "=" * 2)
    metric_ = sorted(metric_dicts.items(), key=lambda x: x[0])
    template = "{0} : R:{1:.4f} P:{2:.4f} F:{3:.4f} Support={4}"
    for k, v in metric_:
        print(template.format(id2label[k], v['R'], v['P'], v['F'], v['Support']))
    print("=" * 2 + "Evaluating Result" + "=" * 2)


def calculate_pr_curve_and_save(golden_list, pred_prob_list, class_num, save_curve_dir, interval=0.02) -> None:
    """
    :param golden_list: label
    :param class_num: num of classes
    :param pred_prob_list: the probability distribution of prediction (batch_size, classes_num)
    :param save_curve_dir: Precision Recall curve save path
    :param interval: calculate every threshold
    """

    def calculate_avg_pr_each_class_with_threshold(golden_list, pred_probs, classes_num, threshold):
        cnt_dicts = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(classes_num)}
        for i in tqdm(range(classes_num)):
            for j in range(len(golden_list)):
                golden_y, pred_prob = golden_list[j], pred_probs[j]
                pred_i_list = pred_prob[:, i]
                cnt_dicts[i]['tp'] += ((golden_y == i) & (pred_i_list >= threshold)).sum().item()
                cnt_dicts[i]['fp'] += ((golden_y != i) & (pred_i_list >= threshold)).sum().item()
                cnt_dicts[i]['fn'] += ((golden_y == i) & (pred_i_list < threshold)).sum().item()

        total_recall = 0.0
        total_precision = 0.0
        for k, v in cnt_dicts.items():
            if k == 0:
                continue
            tp, fn, fp = v['tp'], v['fn'], v['fp']
            total_recall += tp / (tp + fn) if tp + fn != 0 else 1.
            total_precision += tp / (tp + fp) if tp + fp != 0 else 1.

        # without NA
        recall = total_recall / (classes_num - 1)
        precision = total_precision / (classes_num - 1)

        return recall, precision

    with open(save_curve_dir, "w", encoding="UTF-8")as f:
        threshold = 0.0
        while threshold < 1.0 + interval:
            print("Current threshold is {0:.2f}".format(threshold))
            recall, precision = calculate_avg_pr_each_class_with_threshold(golden_list,
                                                                           pred_prob_list,
                                                                           class_num,
                                                                           threshold)
            threshold += interval
            f.write("{0:.5f}\t{1:.5f}\n".format(recall, precision))