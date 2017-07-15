from C45 import C45, mean

from collections import Counter

class MajorityClassifier:
    def __init__(self, models):
        self.models = models

    def predict_single(self, x):
        ys = [model.predict_single(x) for model in self.models]
        return Counter(ys).most_common()[0][0]

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def score(self, X, y):
        """
        Returns the accuracy for predicting the given dataset X
        """

        correct = sum(self.predict(X) == y)
        return float(correct) / len(y)

import random

class RandomForest(MajorityClassifier):
    def __init__(self, num_trees, continuous, max_depth=float("inf")):
        self.models = [C45(continuous=continuous, max_depth=max_depth) for _ in range(num_trees)]

    def fit(self, X, y, k=0.1):
        num_train = int(len(X) * k)

        for model in self.models:
            sub = [random.choice(range(len(X))) for _ in range(num_train)]
            X_sub = X[sub]
            y_sub = y[sub]

            model.fit(X_sub, y_sub)

    def prune(self, X_val, y_val):
        for model in self.models:
            model.prune(X_val, y_val)

class FeatureRandomForest(RandomForest):
    def fit(self, X, y, num_features, p=None):
        self.num_total_features = X.shape[1]
        self.features = {}

        all_features = range(self.num_total_features)

        for model in self.models:
            self.features[model] = set(np.random.choice(all_features, size=num_features, p=p, replace=False))

            X_cut = self._cut_data(model, X)

            model.fit(X_cut, y)

    def _cut_data(self, model, X):
        features = self.features[model]
        cut_features = [feature for feature in range(self.num_total_features) if feature not in features]

        X_cut = X.copy()
        X_cut[:, cut_features] = 0
        return X_cut

    def prune(self, X_val, y_val):
        for model in self.models:
            X_cut = self._cut_data(model, X_val)
            model.prune(X_cut, y_val)
