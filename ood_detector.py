import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import LedoitWolf
from scipy.stats import chi2

class MahalanobisOODDetector:
    def __init__(self, contamination=0.05):
        self.cov = LedoitWolf()
        self.contamination = contamination
        self.threshold = None

    def fit(self, X_train):
        self.cov.fit(X_train)
        df = X_train.shape[1]
        self.threshold = chi2.ppf(1 - self.contamination, df)
        return self

    def score_samples(self, X):
        return self.cov.mahalanobis(X)

    def predict(self, X):
        distances = self.score_samples(X)
        return distances > self.threshold


class IsolationForestOODDetector:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, X_train):
        self.model.fit(X_train)
        return self

    def score_samples(self, X):
        return self.model.score_samples(X)  # higher = more normal

    def predict(self, X):
        return self.model.predict(X) == -1  # -1 = anomaly


class HybridOODDetector:
    def __init__(self, contamination=0.05):
        self.mahalanobis = MahalanobisOODDetector(contamination)
        self.isolation = IsolationForestOODDetector(contamination)

    def fit(self, X_train):
        self.mahalanobis.fit(X_train)
        self.isolation.fit(X_train)
        return self

    def get_scores(self, X):
        return {
            'mahalanobis': self.mahalanobis.score_samples(X),
            'isolation': self.isolation.score_samples(X)
        }

    def predict(self, X):
        m_flags = self.mahalanobis.predict(X)
        i_flags = self.isolation.predict(X)
        return np.logical_or(m_flags, i_flags)
