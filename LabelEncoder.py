from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class LabelEncoder(BaseEstimator, TransformerMixin):

    def index_default_(self):
        return -1

    def index_default_check_(self, index):
        return index != self.index_default_()

    def class_default_(self):
        return ""

    def class_default_check_(self, index):
        return index != self.class_default_()

    def fit(self, y):
        self.classes_ = pd.Series(y).unique()
        self.index_ = range(self.classes_.size)
        self.encoder = defaultdict(self.index_default_, zip(self.classes_, self.index_))
        self.decoder = defaultdict(self.class_default_, zip(self.index_, self.classes_))
        return self

    def encode(self, y):
        return self.encoder[y]

    def decode(self, y):
        return self.decoder[y]

    def transform(self, y, drop=True):
        encoded = pd.Series(y).apply(self.encode)
        if drop:
            return pd.Series(list(filter(self.index_default_check_, encoded)))
        else:
            return encoded

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y, drop=False)

    def inverse_transform(self, y, drop=True):
        decoded = pd.Series(y).apply(self.decode)
        if drop:
            return pd.Series(list(filter(self.class_default_check_, decoded)))
        else:
            return decoded
