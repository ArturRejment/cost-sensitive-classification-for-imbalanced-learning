import numpy as np
from scipy.stats import ttest_rel
from tabulate import tabulate



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

alfa = .05

scores = np.load("scores.npy")
scores = np.mean(scores, axis=-1)
scores = np.mean(scores, axis=-1)

t_statistic = np.zeros((24, 24))
p_value = np.zeros((24, 24))

for i in range(24):
    for j in range(24):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[:, i], scores[:, j])

headers = [
    "DT None",
    "DT 5:1",
    "DT 10:1",
    "DT 20:1",
    "DT 1:5",
    "DT 1:10",
    "DT 1:20",
    "DT balanced",
    "RF None",
    "RF 5:1",
    "RF 10:1",
    "RF 20:1",
    "RF 1:5",
    "RF 1:10",
    "RF 1:20",
    "RF balanced",
    "SVC None",
    "SVC 5:1",
    "SVC 10:1",
    "SVC 20:1",
    "SVC 1:5",
    "SVC 1:10",
    "SVC 1:20",
    "SVC balanced",
]
names_column = np.array([[i] for i in headers])
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")

advantage = np.zeros((24, 24))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
# print("\nAdvantage:\n", advantage_table)

significance = np.zeros((24, 24))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
# print("\nStatistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers, tablefmt="latex")
print("\nStatistically significantly better:\n", stat_better_table)
