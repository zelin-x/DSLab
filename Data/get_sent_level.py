from collections import defaultdict
import json
import re

DATA_PATH = r"bag_level/test.txt"
WRITE_PATH = r"sentence_level/test.json"

data_dicts = defaultdict(dict)
sample_idx = 0
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        lin = line.strip()
        if lin and len(lin.split('\t')) == 4:
            cur_dict = defaultdict(dict)
            head, tail, rel, text = lin.split('\t')
            h_str, h_type, h_idx = re.split('&|\.\.', head)
            t_str, t_type, t_idx = re.split('&|\.\.', tail)
            h_beg, h_end = h_idx.split(':')[0], h_idx.split(':')[1]
            t_beg, t_end = t_idx.split(':')[0], t_idx.split(':')[1]

            cur_dict['text'] = text

            cur_dict['head']['text'] = h_str
            cur_dict['head']['type'] = h_type
            cur_dict['head']['begin'] = int(h_beg)
            cur_dict['head']['end'] = int(h_end)

            cur_dict['tail']['text'] = t_str
            cur_dict['tail']['type'] = t_type
            cur_dict['tail']['begin'] = int(t_beg)
            cur_dict['tail']['end'] = int(t_end)

            cur_dict['relation'] = rel

            data_dicts[sample_idx] = cur_dict
            sample_idx += 1

with open(WRITE_PATH, 'w', encoding='utf-8')as f:
    json.dump(data_dicts, f, ensure_ascii=False)








