import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from tabulate import tabulate

weights_names = [
    "None",
    "5:1",
    "10:1",
    "20:1",
    "1:5",
    "1:10",
    "1:20",
    'balanced'
]

datasets_names = [
    "cleveland-0_vs_4.csv",
    "dermatology-6.csv",
    "ecoli3.csv",
    "ecoli-0-6-7_vs_3-5.csv",
    "haberman.csv",
    "pima.csv",
    "wisconsin.csv",
    "flare-F.csv",
    "shuttle-c0-vs-c4.csv",
    "yeast3.csv",
    "stroke.csv",
]

scores = np.load("scores.npy")

t = PrettyTable([
    "Classifier",
    
    "DT None",
    "DT 5:1",
    "DT 10:1",
    "DT 20:1",
    "DT 1:5",
    "DT 1:10",
    "DT 1:20",
    "DT balanced",

])

x = PrettyTable([
    "Classifier",

    "RF None",
    "RF 5:1",
    "RF 10:1",
    "RF 20:1",
    "RF 1:5",
    "RF 1:10",
    "RF 1:20",
    "RF balanced",
])

z = PrettyTable([
    "Classifier",

    "SVC None",
    "SVC 5:1",
    "SVC 10:1",
    "SVC 20:1",
    "SVC 1:5",
    "SVC 1:10",
    "SVC 1:20",
    "SVC balanced",
])

for idx, dataset_name in enumerate(datasets_names):
    b = np.mean(scores[idx], axis=1)
    c = np.mean(b, axis=1, dtype=float)
    #
    # row = [dataset_name]
    # for i in c[:8]:
    #     row.append(round(i, 3))
    # t.add_row(row)
    #
    #
    # row = [dataset_name]
    # for i in c[8:16]:
    #     row.append(round(i, 3))
    # x.add_row(row)
    #
    # row = [dataset_name]
    # for i in c[16:]:
    #     row.append(round(i, 3))
    # z.add_row(row)
    #

    plt.plot(weights_names, c[:8], label="Decision Tree")

    plt.plot(weights_names, c[8:16], label="Random Forrest")

    plt.plot(weights_names, c[16:], label="SVC")
    plt.title(dataset_name)
    plt.legend()
    plt.savefig(f"../figs/{dataset_name}.jpg")
    plt.clf()

print(t.get_latex_string())
print(x.get_latex_string())
print(z.get_latex_string())
