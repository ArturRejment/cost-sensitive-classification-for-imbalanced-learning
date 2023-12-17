import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, fbeta_score

from mymetacost import MetaCost

df = pd.read_csv("../datasets/cleveland-0_vs_4.csv")

cost_matrixes = [
    np.array([[0, 5], [1, 0]]),
    # np.array([[0, 10], [1, 0]]),
    # np.array([[0, 20], [1, 0]]),
    # np.array([[0, 1], [5, 0]]),
    # np.array([[0, 1], [10, 0]]),
    # np.array([[0, 1], [20, 0]]),
]

datasets_init = [
    "../datasets/cleveland-0_vs_4.csv",
    "../datasets/dermatology-6.csv",
    "../datasets/ecoli3.csv",
    "../datasets/ecoli-0-6-7_vs_3-5.csv",
    "../datasets/haberman.csv",
    "../datasets/pima.csv",
    "../datasets/wisconsin.csv",
    "../datasets/flare-F.csv",
    "../datasets/shuttle-c0-vs-c4.csv",
    "../datasets/yeast3.csv",
    "../datasets/stroke.csv",
]

# PREPARE DATASETS
datasets = []

le = LabelEncoder()

for dataset in datasets_init:
    df = pd.read_csv(dataset)
    for header in df.columns:
        label = le.fit_transform(df[header])
        df.drop(header, axis=1, inplace=True)
        df[header] = label
    datasets.append(df)


clfs = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(probability=True)
]

rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1410)

scores = np.zeros(shape=(len(datasets), len(clfs) * len(cost_matrixes), rskf.get_n_splits()))

for dataset_idx, dataset in enumerate(datasets):
    data = dataset.values
    X = data[:, :-1]
    y = data[:, -1]

    for clf_idx, est in enumerate(clfs):

        for cost_matrix_idx, cost_matrix in enumerate(cost_matrixes):

            bac = []
            for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
                X_train = X[train]
                X_test = X[test]
                y_train = y[train]

                est_cloned = clone(est)
                clf = MetaCost(est_cloned, cost_matrix)
                clf = clf.fit_transform(X_train, y_train)
                y_pred = clf.predict(X_test)

                balanced_accuracy = balanced_accuracy_score(y[test], y_pred)
                scores[dataset_idx, clf_idx, fold_idx] = balanced_accuracy
                bac.append(balanced_accuracy)

            print("META COST ", np.mean(bac))
            print('\n')

np.save("meta_cost_scores", scores)
