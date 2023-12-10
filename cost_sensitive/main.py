import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.base import clone

NUM_CLASSIFIERS = 3
NUM_METRICS = 2


def prepare_clfs(weights):
    clfs = []
    for weight in weights:
        clfs.append(
            DecisionTreeClassifier(class_weight=weight),
        )
    for weight in weights:
        clfs.append(
            RandomForestClassifier(class_weight=weight),
        )
    for weight in weights:
        clfs.append(
            SVC(class_weight=weight),
        )
    return clfs


datasets_last_nominal = [
    "../datasets/cleveland-0_vs_4.csv",
    "../datasets/dermatology-6.csv",
    "../datasets/ecoli3.csv",
    "../datasets/ecoli-0-6-7_vs_3-5.csv",
    "../datasets/haberman.csv",
    "../datasets/pima.csv",
    "../datasets/wisconsin.csv",
]

datasets_all_nominal = [
    "../datasets/stroke.csv",
    "../datasets/creditcard.csv",
    "../datasets/heart_2022_no_nans.csv",
]

datasets_multiclass = [
    "../datasets/dermatology.csv"
]

# PREPARE DATASETS
datasets = {}

binary_weights = [
    None,
    {0: 20, 1: 1},
    'balanced'
]

multiclass_weights = [
    None,
    {1: 10, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
    'balanced'
]

le = LabelEncoder()
for dataset_name in datasets_last_nominal:
    df = pd.read_csv(dataset_name)
    header = df.columns[-1]
    label = le.fit_transform(df[header])
    df.drop(header, axis=1, inplace=True)
    df[header] = label
    datasets[dataset_name] = {
        'data': df.values,
        'weights': binary_weights
    }

for dataset_name in datasets_all_nominal:
    df = pd.read_csv(dataset_name)
    for header in df.columns:
        label = le.fit_transform(df[header])
        df.drop(header, axis=1, inplace=True)
        df[header] = label
    datasets[dataset_name] = {
        'data': df.values,
        'weights': binary_weights
    }

for dataset_name in datasets_multiclass:
    df = pd.read_csv(dataset_name)
    datasets[dataset_name] = {
        'data': df.values,
        'weights': multiclass_weights
    }

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1410)

scores = np.zeros(shape=(len(datasets), NUM_CLASSIFIERS*3, rskf.get_n_splits(), NUM_METRICS))

for dataset_idx, (name, item) in enumerate(datasets.items()):
    print("*" * 40)
    print(name, '\n')

    data = item["data"]
    weights = item["weights"]

    X = data[:, :-1]
    y = data[:, -1]

    clfs = prepare_clfs(weights)
    for est_idx, est in enumerate(clfs):
        print("-" * 30)
        print(est)

        roc = []
        fbeta = []
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            X_train = X[train]
            X_test = X[test]
            y_train = y[train]

            clf = clone(est)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            roc_auc_sc = roc_auc_score(y[test], y_pred)
            fbeta_sc = fbeta_score(y[test], y_pred, beta=0.5, average="weighted")

            scores[dataset_idx, est_idx, fold_idx, 0] = roc_auc_sc
            scores[dataset_idx, est_idx, fold_idx, 1] = fbeta_sc

            roc.append(roc_auc_sc)
            fbeta.append(fbeta_sc)

        print(f"ROC: {np.mean(roc)}")
        print(f"F-BETA: {np.mean(fbeta)}")

np.save("scores", scores)
