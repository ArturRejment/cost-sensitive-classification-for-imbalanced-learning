import matplotlib.pyplot as plt
import numpy as np

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

for idx, dataset_name in enumerate(datasets_names):
    b = np.mean(scores[idx], axis=1)
    c = np.mean(b, axis=1)

    plt.plot(weights_names, c[:8])
    plt.title(dataset_name + " Decision Tree")
    plt.show()

    plt.plot(weights_names, c[8:16])
    plt.title(dataset_name + " Random Forrest")
    plt.show()

    plt.plot(weights_names, c[16:])
    plt.title(dataset_name + " SVC")
    plt.show()

    exit()

