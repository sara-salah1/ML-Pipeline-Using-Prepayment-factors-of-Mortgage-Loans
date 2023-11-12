from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class ReplaceXWithNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns = X.select_dtypes(include=['object']).columns
        for column in columns:
            X.loc[X[column].str.strip() == 'X', column] = np.nan
        return X

class ImputeMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mode_values = None

    def fit(self, X, y=None):
        
        self.mode_values = {
            'SellerName': X['SellerName'].mode()[0],
            'PropertyType': X['PropertyType'].mode()[0],
            'MSA': X['MSA'].mode()[0],
            'PostalCode': X['PostalCode'].mode()[0],
            'PPM': X['PPM'].mode()[0],
            'NumBorrowers': X['NumBorrowers'].mode()[0]
        }
        return self

    def transform(self, X):
        X['FirstTimeHomebuyer'].fillna('Y', inplace=True)
        if self.mode_values is not None:
            X['SellerName'].fillna(self.mode_values['SellerName'], inplace=True)
            X['PropertyType'].fillna(self.mode_values['PropertyType'], inplace=True)
            X['MSA'].fillna(self.mode_values['MSA'], inplace=True)
            X['PostalCode'].fillna(self.mode_values['PostalCode'], inplace=True)
            X['PPM'].fillna(self.mode_values['PPM'], inplace=True)
            X['NumBorrowers'].fillna(self.mode_values['NumBorrowers'], inplace=True)
        return X

class CreditTransformation(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if 'CreditScore' in X.columns:
            X['credit_bins'] = pd.cut(X['CreditScore'], bins=[-np.inf, 650, 700, 750, np.inf],
                                     labels=['Poor', 'Fair', 'Good', 'Excellent'], right=False)
            X.drop(columns=['CreditScore'], axis=1, inplace=True)
            return X

class LTVTransformation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['LTV_bins'] = pd.cut(X['LTV'], bins=[-np.inf, 25, 50, np.inf],
                             labels=['Low', 'Medium', 'High'])
        X.drop(columns=['LTV'], axis=1, inplace=True)
        return X

class MonthsInRepaymentTransformation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['MonthsInRepayment_bins'] = pd.cut(X['MonthsInRepayment'], bins=[0, 4, 8, 12, 16, np.inf],
                                            labels=['0 - 4 yrs', '4 - 8 yrs', '8 - 12 yrs', '12 - 16 yrs', '16 - 20 yrs'])
        X.drop(columns=['MonthsInRepayment'], axis=1, inplace=True)
        return X

# Create a data processing pipeline
data_processing_pipeline = Pipeline([
    ('replace_x_with_nan', ReplaceXWithNaN()),
    ('impute_missing_values', ImputeMissingValues()),
    ('credit_transformation', CreditTransformation()),
    ('ltv_transformation', LTVTransformation()),
    ('months_in_repayment_transformation', MonthsInRepaymentTransformation())
])

    # Apply the pipeline to your DataFrame
# df_initial = data_processing_pipeline.fit_transform(df_initial)