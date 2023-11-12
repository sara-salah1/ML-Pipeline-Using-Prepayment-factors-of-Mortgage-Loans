from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.label_encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            le = LabelEncoder()
            le.fit(X[column])
            self.label_encoders[column] = le
        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            if column in self.label_encoders:
                X_copy[column] = X_copy[column].map(lambda s: s if s in self.label_encoders[column].classes_ else 'unseen')
                X_copy[column] = self.label_encoders[column].transform(X_copy[column])
        return X_copy

class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, Y=None):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X):

        df_OH = pd.DataFrame(self.encoder.transform(X[self.columns])).astype('int64')
        df_OH.columns = self.encoder.get_feature_names_out(self.columns)
        df_OH.index = X.index
        X = pd.concat([X, df_OH], axis=1)
        return X

class RemainingFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        

        X['FirstPaymentDate'] = pd.to_datetime(X['FirstPaymentDate'], format="%Y%m")
        X['MaturityDate'] = pd.to_datetime(X['MaturityDate'], format="%Y%m")
        X['InvestmentPeriod'] = (X['MaturityDate'] - X['FirstPaymentDate']).dt.days
        X = X.drop(['FirstPaymentDate', 'MaturityDate', 'OrigLoanTerm', 'PostalCode', 'PPM', 'LoanSeqNum', 'ProductType', 'Occupancy',
                    'FirstTimeHomebuyer', 'Channel', 'PropertyType', 'LoanPurpose', 'ServicerName', 'SellerName', 'PropertyState', 'MSA'], axis=1)
        return X
    

# Create a data processing pipeline
feature_engineering_pipeline = Pipeline([
    ('label_encoding', LabelEncoderTransformer(columns=['NumBorrowers', 'credit_bins', 'LTV_bins', 'MonthsInRepayment_bins'])),
    ('one_hot_encoding', OneHotEncoderTransformer(columns=['ServicerName', 'SellerName', 'PropertyState', 'MSA', 'FirstTimeHomebuyer', 'Channel', 'PropertyType', 'LoanPurpose'])),
    ('remaining_features', RemainingFeaturesTransformer())
])

# Apply the pipeline to your DataFrame
# df_final = feature_engineering_pipeline.fit_transform(df_final)
