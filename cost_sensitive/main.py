import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from metacost import CustomMetaCostClassifier
from mymetacost import MetaCost


# df = pd.read_csv("diabetes.csv")
# headers = [
#     "race", "gender", "age", "addmision_source", "medical_speciality", "primary_diagnosis",
#     "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed", "medicare",
#     "medicaid", "had_emergency", "had_inpatient_days", "had_outpatient_days", "readmitted"
# ]
#
# le = LabelEncoder()
# for header in headers:
#     label = le.fit_transform(df[header])
#     df.drop(header, axis=1, inplace=True)
#     df[header] = label


# data = df.values
# X = data[:, :-1]
# y = data[:, -1]
# #


weights = [
    None,
    {0: 5, 1: 1},
    {0: 20, 1: 1},
    {0: 50, 1: 1},
    {0: 100, 1: 1},
    {0: 1, 1: 5},
    {0: 1, 1: 20},
    {0: 1, 1: 50},
    {0: 1, 1: 100},
    'balanced',
]

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1410)

clfs = []

for weight in weights:
    clfs.append(
        RandomForestClassifier(class_weight=weight),
    )

for weight in weights:
    clfs.append(
        DecisionTreeClassifier(class_weight=weight),
    )

for weight in weights:
    clfs.append(
        SVC(class_weight=weight),
    )

# for weight in costs:
#     clfs.append(
#         CustomMetaCostClassifier(df, DecisionTreeClassifier(), weight),
#     )


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1410)

for clf in clfs:
    print(clf)

    roc = []
    bac = []
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        bac_score = balanced_accuracy_score(y[test], y_pred)
        roc_auc = roc_auc_score(y[test], y_pred)

        roc.append(roc_auc)
        bac.append(bac_score)

    print(f'ROC AUC: {np.mean(roc)}')
    print(f'BAC: {np.mean(bac)}')

    print("-" *40)
