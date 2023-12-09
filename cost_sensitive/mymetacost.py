import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class MetaCost(BaseEstimator, ClassifierMixin):

    def fit(self, S: pd.DataFrame, L: BaseEstimator, C, m=5, n=1, p=True, q=True):
        n = len(S) * n
        S_ = {}
        M = []

        for i in range(m):
            S_[i] = S.sample(n=n, replace=True)
            X = S_[i][:-1].values
            y = S_[i].iloc[:, -1].values[:-1]

            model = clone(L)
            M.append(model.fit(X, y))

        label = []
        num_classes = len(np.unique(S.iloc[:, -1]))
        S_array = S[:-1].values
        for x in range(len(S)):
            if q:
                M_ = M
            else:
                idxs = [k for k,v in S_.items() if x not in v.index]
                M_ = list(np.array(M)[idxs])

            if p:
                P_j = [model.predict_proba(S_array[[i]]) for model in M_]
            else:
                P_j = []
                vector = [0] * num_classes
                for model in M_:
                    vector[model.predict(S_array[[x]])] = 1
                    P_j.append(vector)

            P = np.array(np.mean(P_j, 0)).T

            label.append(np.argmin(C.dot(P)))

        X_train = S[:-1].values
        y_train = np.array(label)[:-1]
        new_model = clone(L)
        new_model.fit(X_train, y_train)
        return new_model
