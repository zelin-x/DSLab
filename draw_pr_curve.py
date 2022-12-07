import matplotlib.pyplot as plt

pcnn_one_pr_curve_path = r"Models/PCNN+ONE/result/model_2022_11_07_10_24_37.ckpt.txt"
seg_pr_curve_path = r"Models/SEG/result/model_2022_11_14_16_27_55.ckpt.txt"

x0s = []
y0s = []
with open(pcnn_one_pr_curve_path, "r", encoding="UTF-8")as f:
    for line in f:
        lin = line.strip()
        if lin:
            lin = lin.split("\t")
            x0s.append(float(lin[0]))
            y0s.append(float(lin[1]))
x0s.append(0)
y0s.append(1)

x1s = []
y1s = []
with open(seg_pr_curve_path, "r", encoding="UTF-8")as f:
    for line in f:
        lin = line.strip()
        if lin:
            lin = lin.split("\t")
            x1s.append(float(lin[0]))
            y1s.append(float(lin[1]))


plt.figure()
plt.plot(x0s[:], y0s[:], '-p', markersize=1.0, label="pcnn+one(baseline0)")
plt.plot(x1s[:], y1s[:], '-p', markersize=1.0, label="pcnn+ent-att+pooling(baseline1)")

plt.legend()
plt.show()
