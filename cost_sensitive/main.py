import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score


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

datasets_multi_class = [
    "../datasets/dermatology.csv"
]

# PREPARE DATASETS
datasets = {}

for dataset_name in datasets_last_nominal:
    df = pd.read_csv(dataset_name)
    header = df.columns[-1]
    le = LabelEncoder()
    label = le.fit_transform(df[header])
    df.drop(header, axis=1, inplace=True)
    df[header] = label
    data = df.values
    datasets[dataset_name] = data

le = LabelEncoder()
for dataset_name in datasets_all_nominal:
    df = pd.read_csv(dataset_name)
    for header in df.columns:
        label = le.fit_transform(df[header])
        df.drop(header, axis=1, inplace=True)
        df[header] = label
    datasets[dataset_name] = df.values

for dataset_name in datasets_multi_class:
    df = pd.read_csv(dataset_name)
    datasets[dataset_name] = df.values

weights = [
    None,
    {0: 20, 1: 1},
    'balanced',
]


# PREPARE CLFs
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


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1410)

for name, data in datasets.items():

    X = data[:, :-1]
    y = data[:, -1]

    for clf in clfs:
        print(clf)

        roc = []
        fbeta = []
        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            X_train = X[train]
            X_test = X[test]
            y_train = y[train]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # roc_auc = roc_auc_score(y[test], y_pred)
            fbeta_sc = fbeta_score(y[test], y_pred, beta=0.5, average="micro")

            # roc.append(roc_auc)
            fbeta.append(fbeta_sc)

        # print(f'ROC AUC: {np.mean(roc)}')
        print(f'FBeta: {np.mean(fbeta)}')

        print("-" *40)
