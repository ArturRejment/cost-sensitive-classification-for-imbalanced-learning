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

le = LabelEncoder()
header = df.columns[-1]
label = le.fit_transform(df[header])
df.drop(header, axis=1, inplace=True)
df[header] = label
data = df.values

cost_matrix = np.array([[0, 10], [1, 0]])

clfs = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(probability=True)
]

X = data[:, :-1]
y = data[:, -1]

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1410)

for est in clfs:
    print(est)

    bac = []
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]

        est_cloned = clone(est)
        clf = MetaCost(est_cloned, cost_matrix)
        clf = clf.fit_transform(X_train, y_train)
        y_pred = clf.predict(X_test)

        bac.append(balanced_accuracy_score(y[test], y_pred))

    print("META COST ", np.mean(bac))
    print('\n')
