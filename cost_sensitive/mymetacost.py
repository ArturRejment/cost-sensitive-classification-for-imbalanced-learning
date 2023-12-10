import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class MetaCost(BaseEstimator, ClassifierMixin):

    def fit(self, X, y , L: BaseEstimator, C, m=5, n=1, p=True, q=True):

        # Create dataframe from X and y
        columns = [f"Feature_{i + 1}" for i in range(X.shape[1])] + ['Target']
        S = pd.DataFrame(data=X, columns=columns[:-1])
        S['Target'] = y

        new_index = list(range(len(S)))
        S.index = new_index
        n = len(S) * n
        S_ = {}
        M = []

        for i in range(m):
            S_[i] = S.sample(n=n, replace=True)
            X = S_[i].iloc[:, :-1].values
            y = S_[i].iloc[:, -1].values

            model = clone(L)
            M.append(model.fit(X, y))

        label = []
        num_classes = len(np.unique(S.iloc[:, -1]))
        S_array = S.iloc[:, :-1].values
        for x in range(len(S)):
            if q:
                M_ = M
            else:
                idxs = [k for k,v in S_.items() if x not in v.index]
                M_ = list(np.array(M)[idxs])

            if p:
                P_j = [model.predict_proba(S_array[[x]]) for model in M_]
            else:
                P_j = []
                vector = [0] * num_classes
                for model in M_:
                    vector[model.predict(S_array[[x]])] = 1
                    P_j.append(vector)

            P = np.array(np.mean(P_j, 0)).T

            label.append(np.argmin(C.dot(P)))

        X_train = S.iloc[:, :-1].values
        y_train = np.array(label)
        new_model = clone(L)
        new_model.fit(X_train, y_train)
        return new_model
