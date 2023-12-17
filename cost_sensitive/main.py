import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone

NUM_CLASSIFIERS = 3
NUM_METRICS = 1

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
            SVC(class_weight=weight, probability=True),
        )
    return clfs


weights_binary = [
    None,
    {0: 5, 1: 1},
    {0: 10, 1: 1},
    {0: 20, 1: 1},
    {0: 1, 1: 5},
    {0: 1, 1: 10},
    {0: 1, 1: 20},
    'balanced'
]

datasets_init = [
    {'dataset': "../datasets/cleveland-0_vs_4.csv", 'weights': weights_binary},
    {'dataset': "../datasets/dermatology-6.csv", 'weights': weights_binary},
    {'dataset': "../datasets/ecoli3.csv", 'weights': weights_binary},
    {'dataset': "../datasets/ecoli-0-6-7_vs_3-5.csv", 'weights': weights_binary},
    {'dataset': "../datasets/haberman.csv", 'weights': weights_binary},
    {'dataset': "../datasets/pima.csv", 'weights': weights_binary},
    {'dataset': "../datasets/wisconsin.csv", 'weights': weights_binary},
    {'dataset': "../datasets/flare-F.csv", 'weights': weights_binary},
    {'dataset': "../datasets/shuttle-c0-vs-c4.csv", 'weights': weights_binary},
    {'dataset': "../datasets/yeast3.csv", 'weights': weights_binary},
    {'dataset': "../datasets/stroke.csv", 'weights': weights_binary},
]

# PREPARE DATASETS
datasets = {}

le = LabelEncoder()

for dataset in datasets_init:
    dataset_name = dataset['dataset']
    dataset_weights = dataset['weights']

    df = pd.read_csv(dataset_name)
    for header in df.columns:
        label = le.fit_transform(df[header])
        df.drop(header, axis=1, inplace=True)
        df[header] = label
    datasets[dataset_name] = {
        'data': df.values,
        'weights': dataset_weights
    }

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1410)

scores = np.zeros(shape=(len(datasets), NUM_CLASSIFIERS * len(weights_binary), rskf.get_n_splits(), NUM_METRICS))

for dataset_idx, (name, item) in enumerate(datasets.items()):
    print("=" * 40)
    print(name.split('/')[-1], '\n')

    data = item["data"]
    weights = item["weights"]

    X = data[:, :-1]
    y = data[:, -1]

    clfs = prepare_clfs(weights)
    for est_idx, est in enumerate(clfs):
        print("-" * 30)
        print(est)

        bac = []
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            X_train = X[train]
            X_test = X[test]
            y_train = y[train]

            clf = clone(est)
            clf.fit(X_train, y_train)

            y_pred_proba = clf.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:
                y_pred_proba = y_pred_proba[:, 1]

            y_pred = clf.predict(X_test)

            balanced_accuracy = balanced_accuracy_score(y[test], y_pred)

            scores[dataset_idx, est_idx, fold_idx, 0] = balanced_accuracy

            bac.append(balanced_accuracy)

        print(f"BAC: {np.mean(bac)}")

# np.save("scores", scores)
