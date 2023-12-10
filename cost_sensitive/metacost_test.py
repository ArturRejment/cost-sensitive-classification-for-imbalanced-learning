import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, fbeta_score

from mymetacost import MetaCost

df = pd.read_csv("../datasets/cleveland-0_vs_4.csv")

le = LabelEncoder()
header = df.columns[-1]
label = le.fit_transform(df[header])
df.drop(header, axis=1, inplace=True)
df[header] = label
data = df.values

cost_matrix = np.array([[0, 1000], [1, 0]])

X = data[:, :-1]
y = data[:, -1]

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1410)

for i in range(10):
    bac = []
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]

        clf = MetaCost().fit(
            X=X_train,
            y=y_train,
            L=KNeighborsClassifier(),
            C=cost_matrix
        )

        y_pred = clf.predict(X_test)

        bac.append(fbeta_score(y[test], y_pred, beta=2))

    print(np.mean(bac))
