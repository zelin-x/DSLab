import matplotlib.pyplot as plt

pr_curve_path = r"model_2022_11_07_10_24_37.ckpt.txt"

xs = []
ys = []
with open(pr_curve_path, "r", encoding="UTF-8")as f:
    for line in f:
        lin = line.strip()
        if lin:
            lin = lin.split("\t")
            xs.append(float(lin[0]))
            ys.append(float(lin[1]))
plt.figure()

plt.plot(xs[:], ys[:], '-p', markersize=1.0)
plt.show()
