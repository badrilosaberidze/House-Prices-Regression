## This class takes care of missing values, it drops columns if missing values are more than specific threshold, and 
## Takes Care of numerical and categorical column's missing values as provided strategy. can use median or mean for numericals
## and mode for categorical column values.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, num_strategy='median', cat_strategy='mode'):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.num_impute_values = {}
        self.cat_impute_values = {}
    
    def fit(self, X, y=None):
        for col in X.columns:
            if X[col].dtype == 'object':
                self.cat_impute_values[col] = X[col].mode()[0]
            else:
                if self.num_strategy == 'median':
                    self.num_impute_values[col] = X[col].median()
                elif self.num_strategy == 'mean':
                    self.num_impute_values[col] = X[col].mean()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col, val in self.cat_impute_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        
        for col, val in self.num_impute_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
                
        return X


## This class standardize or scale numerical features so that they will be on a similar scale 

class NumFeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols):
        self.num_cols = num_cols
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.num_cols])
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        return X

class CatFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, strategy='onehot'):
        self.cat_cols = cat_cols
        self.strategy = strategy
        self.encoder = None

    def fit(self, X, y=None):
        self.existing_cat_cols_ = [col for col in self.cat_cols if col in X.columns]
        if self.strategy == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(X[self.existing_cat_cols_])
        # If you implement other strategies, add them here.
        return self

    def transform(self, X):
        X = X.copy()
        cols_to_encode = [col for col in self.cat_cols if col in X.columns]
        if self.strategy == 'onehot' and cols_to_encode:
            encoded_array = self.encoder.transform(X[cols_to_encode])
            encoded_df = pd.DataFrame(
                encoded_array, 
                columns=self.encoder.get_feature_names_out(cols_to_encode),
                index=X.index
            )
            X = X.drop(columns=cols_to_encode)
            X = pd.concat([X, encoded_df], axis=1)
        return X

###################################################################

class CorrelationFeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.features_to_drop_ = []

    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        high_corr_pairs = [
            (column, row)
            for column in upper.columns
            for row in upper.index
            if upper.loc[row, column] > self.threshold
        ]
        
        features_to_drop = set()
        for feat1, feat2 in high_corr_pairs:
            # Compare correlation with y to keep the stronger feature
            if y is not None:
                corr1 = abs(np.corrcoef(X[feat1], y)[0, 1])
                corr2 = abs(np.corrcoef(X[feat2], y)[0, 1])
                if corr1 < corr2:
                    features_to_drop.add(feat1)
                else:
                    features_to_drop.add(feat2)
            else:
                # If no y provided, arbitrarily drop 
                features_to_drop.add(feat2)
        
        self.features_to_drop_ = list(features_to_drop)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        return X_transformed.drop(columns=self.features_to_drop_, errors='ignore')
    
# --- Custom WOE + One-Hot Preprocessor ---
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, woe_columns, one_hot_columns, nan_drop_threshold=0.8):
        self.woe_columns = woe_columns
        self.one_hot_columns = one_hot_columns
        self.nan_drop_threshold = nan_drop_threshold
        self.features_to_drop = [
            "Id", "MiscVal", "MiscFeature", "PoolQC", "PoolArea", "Fence", "Alley",
            "Utilities", "RoofMatl", "Street", "Condition2", "3SsnPorch", 
            "LowQualFinSF", "MoSold", "YrSold", "GarageYrBlt"
        ]

    def fit(self, X, y):
        missing_percent = X.isna().mean()
        self.features_to_drop += missing_percent[missing_percent > self.nan_drop_threshold].index.tolist()

        self.woe_columns = [col for col in self.woe_columns if col not in self.features_to_drop]
        self.one_hot_columns = [col for col in self.one_hot_columns if col not in self.features_to_drop]

        self.woe_columns_fill_na = X[self.woe_columns].mode().T[0].to_dict()

        df_woe = X.copy()
        df_woe['target'] = y

        self.woe_mappings = {}
        self.iv_values = {}

        for col in self.woe_columns:
            groups = df_woe.groupby([col])['target'].agg(['count', 'sum'])
            groups.columns = ['n_obs', 'n_pos']
            groups['n_neg'] = groups['n_obs'] - groups['n_pos']
            groups['prop_pos'] = groups['n_pos'] / groups['n_pos'].sum()
            groups['prop_neg'] = groups['n_neg'] / groups['n_neg'].sum()
            groups['woe'] = np.log(groups['prop_pos'] / groups['prop_neg'])
            groups['iv'] = (groups['prop_pos'] - groups['prop_neg']) * groups['woe']
            groups.replace([np.inf, -np.inf], 0, inplace=True)
            groups.fillna(0, inplace=True)

            self.woe_mappings[col] = groups['woe'].to_dict()
            self.iv_values[col] = groups['iv'].sum()

        self.one_hot_encoder = OneHotEncoder(
            handle_unknown='ignore',
            drop='first',
            sparse_output=False,
            dtype=int
        )
        self.one_hot_encoder.fit(X[self.one_hot_columns])

        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.drop(columns=self.features_to_drop, errors='ignore')

        for col in self.woe_columns:
            X_transformed[f'{col}_woe'] = X_transformed[col].map(self.woe_mappings[col])
            X_transformed.drop(columns=col, inplace=True)

        # One-hot encode
        ohe_array = self.one_hot_encoder.transform(X_transformed[self.one_hot_columns])
        ohe_columns = self.one_hot_encoder.get_feature_names_out(self.one_hot_columns)
        df_ohe = pd.DataFrame(ohe_array, columns=ohe_columns, index=X_transformed.index)

        # Drop original one-hot columns and concatenate encoded ones
        X_transformed.drop(columns=self.one_hot_columns, inplace=True)
        X_transformed = pd.concat([X_transformed, df_ohe], axis=1)

        na_cols = X_transformed.columns[X_transformed.isna().any()].tolist()
        for col in na_cols:
            if col.endswith("_woe"):
                name = col[:-4]
                X_transformed[col] = X_transformed[col].fillna(
                    self.woe_mappings[name][self.woe_columns_fill_na[name]])
        return X_transformed
