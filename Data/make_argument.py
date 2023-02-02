import json
import pkuseg
from tqdm import tqdm
from gensim import models
from gensim import corpora
from gensim.models import TfidfModel
from collections import defaultdict

train_path = r"sentence_level/train.json"
test_path = r"sentence_level/test.json"

with open(train_path, 'r', encoding='utf-8')as trf, open(test_path, 'r', encoding='utf-8')as tf:
    train_dicts = json.load(trf)
    test_dicts = json.load(tf)

all_texts = []
entity_list = []
for k, v in train_dicts.items():
    text = v['text']
    all_texts.append(text)
    head_str, head_type, head_beg, head_end = v['head'].values()
    tail_str, tail_type, tail_beg, tail_end = v['tail'].values()
    head_str, tail_str = text[head_beg:head_end + 1], text[tail_beg:tail_end + 1]
    entity_list.append(head_str)
    entity_list.append(tail_str)

for k, v in test_dicts.items():
    text = v['text']
    head_str, head_type, head_beg, head_end = v['head'].values()
    tail_str, tail_type, tail_beg, tail_end = v['tail'].values()
    head_str, tail_str = text[head_beg:head_end + 1], text[tail_beg:tail_end + 1]
    entity_list.append(head_str)
    entity_list.append(tail_str)

entity_list = list(set(entity_list))

with open('sentence_level/entities.txt', 'w', encoding='utf-8')as f:
    for _ in entity_list:
        f.write(_ + '\n')

seg = pkuseg.pkuseg(model_name="medicine", user_dict='sentence_level/entities.txt')
all_texts = [seg.cut(_) for _ in tqdm(all_texts)]

dictionary = corpora.Dictionary(all_texts)
corpus = [dictionary.doc2bow(text) for text in all_texts]
tf_idf_model = TfidfModel(corpus, normalize=True)
word_tf_tdf = list(tf_idf_model[corpus])

# assert len(dictionary.token2id) == len(word_tf_tdf)

id_mean_tf_idf = {}
for text_tf in word_tf_tdf:
    for _ in text_tf:
        idx, cur_tf_idf = _
        if idx not in id_mean_tf_idf:
            id_mean_tf_idf[idx] = [cur_tf_idf, 1]
        else:
            id_mean_tf_idf[idx][0] += cur_tf_idf
            id_mean_tf_idf[idx][1] += 1

for k, v in id_mean_tf_idf.items():
    id_mean_tf_idf[k] = v[0] / v[1]

word_tf_idf_dict = {}
for k, v in dictionary.token2id.items():
    word_tf_idf_dict[k] = id_mean_tf_idf[v]

upper_bound, lower_bound = max(word_tf_idf_dict.values()), min(word_tf_idf_dict.values())
gap = (upper_bound - lower_bound) / 15

bucket_reverse_index = defaultdict(dict)
for k, v in word_tf_idf_dict.items():
    try:
        bucket_reverse_index[(v - lower_bound) // gap][len(k)].append(k)
    except KeyError:
        bucket_reverse_index[(v - lower_bound) // gap][len(k)] = [k]

# for k, v in bucket_reverse_index.items():
#     print(k, v)

import random

for key, v in train_dicts.items():
    text = v['text']
    head_str, head_type, head_beg, head_end = v['head'].values()
    tail_str, tail_type, tail_beg, tail_end = v['tail'].values()
    head_str, tail_str = text[head_beg:head_end + 1], text[tail_beg:tail_end + 1]
    if head_beg < tail_beg:
        pre_text = text[0:head_beg]
        mid_text = text[head_end+1:tail_beg]
        last_text = text[tail_end+1:]
    else:
        pre_text = text[0:tail_beg]
        mid_text = text[tail_end + 1:head_beg]
        last_text = text[head_end + 1:]
    ent1 = head_str if head_beg < tail_beg else tail_str
    ent2 = head_str if head_beg > tail_beg else tail_str
    seg_text = seg.cut(pre_text) + [ent1] + seg.cut(mid_text) + [ent2] + seg.cut(last_text)
    # seg_text = seg.cut(text)
    aug_text = ""
    assert head_str in seg_text and tail_str in seg_text
    for sub_text in seg_text:
        if sub_text == head_str:
            head_beg, head_end = len(aug_text), len(aug_text) + len(sub_text) - 1
        elif sub_text == tail_str:
            tail_beg, tail_end = len(aug_text), len(aug_text) + len(sub_text) - 1
        elif sub_text in word_tf_idf_dict and (word_tf_idf_dict[sub_text] - lower_bound) // gap <= 2:
            if random.random() < 0.3:
                for k in [1.0, 2.0]:
                    if len(sub_text) in bucket_reverse_index[k]:
                        start, stop = 0, len(bucket_reverse_index[k][len(sub_text)])
                        sub_text = bucket_reverse_index[k][len(sub_text)][random.randrange(start, stop)]
                        break
        else:
            pass

        aug_text += sub_text
    # assert flag == 2
    assert aug_text[head_beg:head_end + 1] == head_str and aug_text[tail_beg:tail_end + 1] == tail_str
    train_dicts[key]['aug_text'] = aug_text
    train_dicts[key]['head']['aug_beg'] = head_beg
    train_dicts[key]['head']['aug_end'] = head_end
    train_dicts[key]['tail']['aug_beg'] = tail_beg
    train_dicts[key]['tail']['aug_end'] = tail_end

with open('sentence_level/aug_train.json', 'w', encoding='utf-8')as f:
    json.dump(train_dicts, f, ensure_ascii=False, indent=1)
