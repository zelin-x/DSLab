
label2id = {'NA': 0}
with open("train.txt", "r", encoding="UTF-8") as f:
    for line in f:
        lin = line.strip()
        if lin:
            lin = lin.split('\t')
            if len(lin) == 3 and lin[-1] not in label2id:
                label2id[lin[-1]] = len(label2id)


with open("label.txt", "w", encoding="UTF-8")as f:
    for k, v in label2id.items():
        f.write(k + ' ' + str(v) + '\n')
print('-')