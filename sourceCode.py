# %%
import pandas as pd
import numpy as np

import os

# %% [markdown]
# # Load models

# %%
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed

set_random_seed(42)


def data_splitting(df: pd.DataFrame, seed: int,
                   train_frac=0.7, valid_frac=0.3):
    """
    Split the fraud (labeled) dataset into training and validation subsets.
    """
    main_labeled = df[df["labeled"] == 1]
    main_unlabeled = df[df["labeled"] == 0] 
    np.random.seed(seed)
    n = len(main_labeled)
    train_indices = np.random.choice(main_labeled.index, size=int(train_frac * n), replace=False)
    remaining = list(set(main_labeled.index) - set(train_indices))
    valid_indices = np.random.choice(remaining, size=int(valid_frac * n), replace=False)
    fraud_train = main_labeled.loc[train_indices].copy()
    fraud_val = main_labeled.loc[valid_indices].copy()
    main_train_full = pd.concat([fraud_train, main_unlabeled], ignore_index=True)
    return fraud_val, main_train_full


    
def unsupervised_model(df_train: pd.DataFrame, method: str = "svm") -> pd.DataFrame:
    train_fraud = df_train[df_train["labeled"] == 1]
    X_train = train_fraud.drop(columns=["owner_tin", "tax_year", "qtr", "dummy", "labeled", "fraud"], errors='ignore')
    df_pred = df_train.drop(columns=["owner_tin", "tax_year", "qtr", "dummy", "labeled", "fraud"], errors='ignore')
    if method == "svm":
        model = OneClassSVM(gamma='auto').fit(X_train)
        preds = model.predict(df_pred)
        df_pred = df_pred.copy()
        df_pred['fraud_pred'] = np.where(preds == -1, "A", "B")
    elif method == "iso":
        model = IsolationForest(n_estimators=200, random_state=42).fit(X_train)
        scores = model.decision_function(df_pred)
        thresh = np.median(scores)
        df_pred = df_pred.copy()
        df_pred["fraud_pred"] = np.where(scores < thresh, "A", "B")
    elif method == "kmeans":
        # Fit KMeans on the training data
        model = KMeans(n_clusters=2, random_state=42).fit(X_train)
        # Determine which cluster is the minority (assumed fraudulent)
        train_clusters = model.labels_
        unique, counts = np.unique(train_clusters, return_counts=True)
        freq = dict(zip(unique, counts))
        fraud_cluster = min(freq, key=freq.get)  # Cluster with fewer points
        # Predict clusters for the entire dataset
        preds = model.predict(df_pred)
        df_pred = df_pred.copy()
        df_pred["fraud_pred"] = np.where(preds == fraud_cluster, "A", "B")
    else:
        raise ValueError("Unknown method")
    return df_pred

def pca_transform(df: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """
    Apply PCA to reduce feature dimensions.
    
    Parameters:
    - df: DataFrame to transform (should not include non-numeric or target columns).
    - n_components: Number of PCA components to keep.

    Returns:
    - Transformed DataFrame with reduced dimensions.
    """
    # Drop non-numeric and target columns safely
    non_feature_cols = ['owner_tin', 'tax_year', 'qtr', 'dummy', 'labeled', 'fraud', 'fraud_pred']
    feature_df = df.drop(columns=[col for col in non_feature_cols if col in df.columns])

    # Standardize before PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(scaled_features)

    # Construct a DataFrame with PCA components
    pca_df = pd.DataFrame(pca_features, columns=[f'pca_{i+1}' for i in range(n_components)])
    
    # Preserve target or ID columns if needed
    retained_cols = df[[col for col in non_feature_cols if col in df.columns]].reset_index(drop=True)
    result = pd.concat([pca_df, retained_cols], axis=1)

    return result

def supervised_model(data: pd.DataFrame, method: str = "multi"):
    # Separate features and target
    X = data.drop(columns=["fraud"], errors='ignore')
    y = data["fraud"]
    
    if method == "rf":
        # Random Forest classifier
        model = RandomForestClassifier(n_estimators=500, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)

    elif method == "multi":
        model = LogisticRegression()
        model.fit(X,y)
        preds = model.predict(X)
    
    elif method == "ann":
        # ANN using Keras from TensorFlow
        
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=X.shape[1]))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        # Output layer: one neuron with sigmoid for binary classification
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        # Train the model (you can adjust epochs and batch_size as needed)
        model.fit(X, y, epochs=100, batch_size=32, verbose=0)
        # Predict probabilities, then threshold at 0.5 for binary output
        preds = (model.predict(X) > 0.5).astype(int).flatten()
        
    elif method == "xgb":
        # XGBoost classifier
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=42, n_estimators=500)
        model.fit(X, y)
        preds = model.predict(X)
        
    else:
        raise ValueError("Unknown method")
    
    return model, preds


def validate_model(unsup_fraud: pd.DataFrame, unlabeled: pd.DataFrame, fraud_val: pd.DataFrame):
    results = {}
    for u_method in ["svm", "iso"]:
        for s_method in ["multi", "rf"]:
            combined = unsupervised_model(unsup_fraud, unlabeled, method=u_method)
            model = supervised_model(combined, method=s_method)
            X_val = fraud_val.drop(columns=['Label', 'TIN'], errors='ignore')
            preds = model.predict(X_val)
            perc_kf = np.mean(preds == "Known Fraud")
            perc_lf = np.mean(preds == "Likely Fraud")
            perc_ll = np.mean(preds == "Likely Legitimate")
            results[f"{u_method}_{s_method}"] = (perc_kf + perc_lf, perc_kf, perc_ll)
    return results



# %% [markdown]
# # Load preprocessing code

# %%
from mlxtend.frequent_patterns import apriori, association_rules

import pandas as pd
import numpy as np
import re

from typing import Union, List

def parse_qtr(x):
    # if it’s already an int (or float that’s equivalent to an int), just return it
    if isinstance(x, int):
        return x
    # otherwise coerce to string and pull out the first number you see
    s = str(x)
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

class SLS:
    def __init__(self, df: Union[pd.DataFrame, List[pd.DataFrame]]):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
        """
        self.df = df

        self._clean_df = None
        self._agg_df = None
        self._txn_df = None

        self._clean_columns = ["owner_tin", "tax_year", "qtr", "sls_taxable_sales"]
        self._txn_columns = ["owner_tin", "pur_tin", "tax_year", "qtr", "sls_taxable_sales"]

    @property
    def clean_df(self):
        if self._clean_df is None:
            self._clean_df = self.clean()
        return self._clean_df
    
    @clean_df.setter
    def clean_df(self, value):
        if value is not isinstance(value, pd.DataFrame):
            raise ValueError("clean_df must be a pd.DataFrame or None")
        self._clean_df = value

    @property
    def agg_df(self):
        if self._agg_df is None:
            self._agg_df = self.agg()
        return self._agg_df
    
    @agg_df.setter
    def agg_df(self, value):
        if value is not isinstance(value, pd.DataFrame):
            raise ValueError("agg_df must be pd.DataFrame or None")
        
        self._agg_df = value

    @property
    def txn_df(self):
        if self._txn_df is None:
            self._txn_df = self.txn()
        return self._txn_df
    
    @txn_df.setter
    def txn_df(self, value):
        if value is not isinstance(value, pd.DataFrame):
            raise ValueError("txn_df must be pd.DataFrame or None")
        
        self._txn_df = value  

    def clean(self):
        if isinstance(self.df, pd.DataFrame):
            df_sls: pd.DataFrame = self.df[self._txn_columns]

        elif isinstance(self.df, List) and all(isinstance(df, pd.DataFrame) for df in self.df):
            df_sls = pd.concat([df[self._txn_columns] for df in self.df])

        else:
            raise TypeError("Expected 'df' to be a DataFrame or a list of DataFrames.")

        df_sls["qtr"] = df_sls["qtr"].apply(parse_qtr)
        df_sls["sls_taxable_sales"] = pd.to_numeric(df_sls["sls_taxable_sales"], errors="coerce")
        return df_sls
    
    def agg(self):
        # Group by owner_tin, tax_year, and qtr, then sum sls_taxable_sales
        agg_df = self.clean_df[self._clean_columns].groupby(['owner_tin', 'tax_year', 'qtr']).agg(
            total_sales=('sls_taxable_sales', np.nansum)
        ).reset_index()
        # print
        # Create the "dummy" column by concatenating the group keys
        agg_df['dummy'] = agg_df[['owner_tin', 'tax_year', 'qtr']].astype(str).agg('-'.join, axis=1)
        return agg_df
    
    def txn(self):
        agg_df = self.clean_df[["owner_tin", "pur_tin", "sls_taxable_sales"]]
        agg_df = agg_df.groupby(["owner_tin", "pur_tin"]).agg(
            count=('sls_taxable_sales', "count"),
            total=("sls_taxable_sales", np.nansum)
        ).reset_index()
        return agg_df

    
class SLP:
    def __init__(self, df: Union[pd.DataFrame, List[pd.DataFrame]]):
        
        self.df = df

        self._clean_df = None
        self._agg_df = None
        self._txn_df = None

        self._clean_columns = ["owner_tin", "tax_year", "qtr", "gross_taxable_purchases"]
        self._txn_columns = ["owner_tin", "sel_tin", "tax_year", "qtr", "gross_taxable_purchases"]

    @property
    def clean_df(self):
        if self._clean_df is None:
            self._clean_df = self.clean()
        return self._clean_df
    
    @clean_df.setter
    def clean_df(self, value):
        if value is not isinstance(value, pd.DataFrame):
            raise ValueError("clean_df must be a pd.DataFrame or None")
        self._clean_df = value

    @property
    def agg_df(self):
        if self._agg_df is None:
            self._agg_df = self.agg()
        return self._agg_df
    
    @agg_df.setter
    def agg_df(self, value):
        if value is not isinstance(value, pd.DataFrame):
            raise ValueError("agg_df must be pd.DataFrame or None")
        self._agg_df = value

    @property
    def txn_df(self):
        if self._txn_df is None:
            self._txn_df = self.txn()
        return self._txn_df
    
    @txn_df.setter
    def txn_df(self, value):
        if value is not isinstance(value, pd.DataFrame):
            raise ValueError("txn_df must be pd.DataFrame or None")
        self._txn_df = value
    
    def clean(self):
        if isinstance(self.df, pd.DataFrame):
            df_slp: pd.DataFrame = self.df[self._txn_columns]
        elif isinstance(self.df, List) and all(isinstance(df, pd.DataFrame) for df in self.df):
            df_slp: pd.DataFrame = pd.concat([df[self._txn_columns] for df in self.df])

        df_slp["qtr"] = df_slp["qtr"].apply(parse_qtr)
        df_slp["gross_taxable_purchases"] = pd.to_numeric(df_slp["gross_taxable_purchases"], errors="coerce")
        return df_slp
    
    def agg(self):
        agg_df = self.clean_df[self._clean_columns].groupby(['owner_tin', 'tax_year', 'qtr'], as_index=False).agg(
            total_purch=('gross_taxable_purchases', np.nansum)
        ).reset_index(drop=True)
        
        # Create the "dummy" column    by concatenating the group keys
        agg_df['dummy'] = agg_df[['owner_tin', 'tax_year', 'qtr']].astype(str).agg('-'.join, axis=1)
        
        return agg_df
    
    def txn(self):
        agg_df: pd.DataFrame = self.clean_df[["owner_tin", "sel_tin", "gross_taxable_purchases"]]
        agg_df = agg_df.groupby(["owner_tin", "sel_tin"]).agg(
            count=("gross_taxable_purchases", "count"),
            total=("gross_taxable_purchases", np.nansum),
        ).reset_index()
        return agg_df


class VAT:
    def __init__(self, df: Union[pd.DataFrame, List[pd.DataFrame]]):
        self.df = df
        self._clean_df = None
        self._agg_df = None

    @property
    def clean_df(self):
        if self._clean_df is None:
            self._clean_df = self.clean()
        return self._clean_df
    
    @clean_df.setter
    def clean_df(self, value):
        if value is not isinstance(value, pd.DataFrame):
            raise ValueError("clean_df must be a pd.DataFrame or None")
        self._clean_df = value

    def clean(self):
        if isinstance(self.df, pd.DataFrame):
            df_vat: pd.DataFrame = self.df[["DATE_FILED", "TIN", "YEAR", "QTR", "NET_PAYBLE", "AMENDED_YN", "PENALTIES"]]
        
        elif isinstance(self.df, List) and all(isinstance(df, pd.DataFrame) for df in self.df):
            df_vat: pd.DataFrame = pd.concat([df[["DATE_FILED", "TIN", "YEAR", "QTR", "NET_PAYBLE", "AMENDED_YN", "PENALTIES"]] for df in self.df])
        
        else:
            raise TypeError("Expected 'df' to be a DataFrame or a list of DataFrames.")

        df_vat.rename(
            columns={
                "TIN": "owner_tin",
                "YEAR": "tax_year",
                "QTR": "qtr",
                "NET_PAYBLE": "net_payable",
                "AMENDED_YN": "amended_yn",
                "PENALTIES": "penalties",
            },
            inplace=True
            )
        df_vat["qtr"] = df_vat["qtr"].apply(parse_qtr)
        df_vat['amended_yn'] = df_vat['amended_yn'].map({"Y": 1, "N": 0})
        df_vat['dummy'] = df_vat[['owner_tin', 'tax_year', 'qtr']].astype(str).agg('-'.join, axis=1)
        return df_vat
    
class Industry:
    def __init__(self, df: Union[pd.DataFrame, List[pd.DataFrame]]):
        self.df = df

        self._columns = ["TIN", "INDUSTRY_GROUP_ID", "INDUSTRY_GROUP_DESC"]
        self._clean_df = None
        self._agg_df = None

    @property
    def clean_df(self):
        if self._clean_df is None:
            self._clean_df = self.clean()
        return self._clean_df
    
    @clean_df.setter
    def clean_df(self, value):
        if value is not isinstance(value, pd.DataFrame):
            raise ValueError("clean_df must be a pd.DataFrame or None")
        self._clean_df = value

    def clean(self):
        if isinstance(self.df, pd.DataFrame):
            df_ind: pd.DataFrame = self.df[self._columns]
        elif isinstance(self.df, List) and all(isinstance(df, pd.DataFrame) for df in self.df):
            df_ind: pd.DataFrame = pd.concat([df[self._columns] for df in self.df])

        df_ind["INDUSTRY_GROUP_ID"] = pd.to_numeric(df_ind["INDUSTRY_GROUP_ID"], errors="coerce")

        df_ind.fillna({
            'INDUSTRY_GROUP_ID': 0,
            'INDUSTRY_GROUP_DESC': 'UNKNOWN'
        }, inplace=True)

        df_ind["INDUSTRY_GROUP_ID"] = df_ind["INDUSTRY_GROUP_ID"].astype(int)
        
        df_ind.rename(
            columns={
                "TIN": "owner_tin",
                "INDUSTRY_GROUP_DESC": "industry_group_desc",
                "INDUSTRY_GROUP_ID": "industry_group_id"
            }, inplace=True
        )

        return df_ind
    
class Aggregate:
    def __init__(self, sls: SLS, slp: SLP, vat: VAT, industry: Industry):
        self.sls = sls
        self.slp = slp
        self.vat = vat
        self.ind = industry

        self._merge_df = None
        self._cross_df = None
        self._complete_df = None
        self._median_tin_df = None
        self._median_ind_df = None
        self._interpolated_df = None

    @property
    def merge_df(self):
        if self._merge_df is None:
            self._merge_df = self.merge()
        return self._merge_df
        
    @merge_df.setter
    def merge_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("merge_df must be a pd.DataFrame or None")
        self._merge_df = value

    @property
    def cross_df(self):
        if self._cross_df is None:
            self._cross_df = self.cross_ratios()
        return self._cross_df
        
    @cross_df.setter
    def cross_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("cross_df must be a pd.DataFrame or None")
        self._cross_df = value

    @property
    def complete_df(self):
        if self._complete_df is None:
            self._complete_df = self.complete()
        return self._complete_df
        
    @complete_df.setter
    def complete_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("complete_df must be a pd.DataFrame or None")
        self._complete_df = value

    @property
    def median_ind_df(self):
        if self._median_ind_df is None:
            self._median_ind_df = self.median_ind()
        return self._median_ind_df
        
    @median_ind_df.setter
    def median_ind_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("median_ind_df must be a pd.DataFrame or None")
        self._median_ind_df = value
    
    @property
    def median_tin_df(self):
        if self._median_tin_df is None:
            self._median_tin_df = self.median_tin()
        return self._median_tin_df
        
    @median_tin_df.setter
    def median_tin_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("median_tin_df must be a pd.DataFrame or None")
        self._median_tin_df = value

    @property
    def interpolated_df(self):
        if self._interpolated_df is None:
            self._interpolated_df = self.interpolate()
        return self._interpolated_df
        
    @interpolated_df.setter
    def interpolated_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("interpolated_df must be a pd.DataFrame or None")
        self._interpolated_df = value


    def merge(self):
        # Merging sls and slp on 'dummy'
        agg_df = pd.merge(self.sls.agg_df, self.slp.agg_df[["total_purch", "dummy"]], how="left", on="dummy")
        # Merging with vat data on 'dummy'
        agg_df = pd.merge(agg_df, self.vat.clean_df[["net_payable", "amended_yn", "penalties", "dummy"]], how="left", on="dummy")
        # Merging with industry data on 'owner_tin'
        return agg_df

    def cross_ratios(self):
        cross_df = self.merge_df.copy()
        cross_df["sales_purch"] = cross_df["total_sales"] / cross_df["total_purch"]
        cross_df["purch_sales"] = 1 / cross_df["sales_purch"]
        cross_df["sales_vat"] = cross_df["total_sales"] / cross_df["net_payable"]
        cross_df["vat_sales"] = 1 / cross_df["sales_vat"]
        cross_df["purch_vat"] = cross_df["total_purch"] / cross_df["net_payable"]
        cross_df["vat_purch"] = 1 / cross_df["purch_vat"]
        return cross_df

    def complete(self):
        cross_df = self.cross_df.copy()
        tin_lst = cross_df["owner_tin"].unique()
        min_year = cross_df["tax_year"].min()
        max_year = cross_df["tax_year"].max()
        
        # Define the ranges for years and quarters
        years = range(min_year, max_year + 1)
        qtrs = range(1, 5)
        
        # Create a MultiIndex of all permutations and convert to a DataFrame
        perm_index = pd.MultiIndex.from_product(
            [tin_lst, years, qtrs],
            names=["owner_tin", "tax_year", "qtr"]
        )
        perm_df = perm_index.to_frame(index=False)
        # Create a dummy key by joining owner_tin, tax_year, and qtr as strings
        perm_df["dummy"] = perm_df[['owner_tin', 'tax_year', 'qtr']].astype(str).agg('-'.join, axis=1)
        
        # Merge with cross_df on 'dummy', keeping one copy of duplicate columns
        perm_df = pd.merge(perm_df, self.cross_df, how="left", on="dummy", suffixes=("", "_dup"))
        perm_df = perm_df.loc[:, ~perm_df.columns.str.endswith('_dup')]

        perm_df = perm_df.drop_duplicates(subset="dummy", keep="first").reset_index(drop=True)

        perm_df["missing_total_sales"] = perm_df["total_sales"].isnull().astype(int)
        perm_df["missing_total_purch"] = perm_df["total_purch"].isnull().astype(int)
        perm_df["missing_net_payable"] = perm_df["net_payable"].isnull().astype(int)

        perm_df["total_missing"] = perm_df["missing_total_sales"] + perm_df["missing_total_purch"] + perm_df["missing_net_payable"]

        perm_df = pd.merge(perm_df, self.ind.clean_df, how="left", on="owner_tin")

        perm_df.fillna({
            'industry_group_id': 0,
            'industry_group_desc': 'UNKNOWN'
        }, inplace=True)

        perm_df = perm_df.drop_duplicates(subset="dummy", keep="first").reset_index(drop=True)
        
        return perm_df
    
    def median_ind(self):
        df = self.complete_df.copy()
        
        median_ind = df[["industry_group_id", "sales_purch", "purch_sales", "sales_vat", "vat_sales", "purch_vat", "vat_purch"]].groupby("industry_group_id").median().reset_index()
        return median_ind 
    
    def median_tin(self):
        df = self.complete_df.copy()
        df_ind = self.median_ind_df.copy()

        # Compute owner-level medians for key ratio columns.
        median_tin = df[["owner_tin", "industry_group_id", "sales_purch", "purch_sales",
                        "sales_vat", "vat_sales", "purch_vat", "vat_purch"]].groupby("owner_tin").median().reset_index()

        # Merge with industry median dataframe on 'industry_group_id'.
        median_tin = pd.merge(median_tin, df_ind, how="left", on="industry_group_id", suffixes=("", "_ind"))

        # Impute missing owner-level medians with corresponding industry medians.
        for col in ["sales_purch", "purch_sales", "sales_vat", "vat_sales", "purch_vat", "vat_purch"]:
            median_tin[col] = median_tin[col].fillna(median_tin[f"{col}_ind"])

        # Optionally, drop the industry median columns after imputation.
        ind_cols = [f"{col}_ind" for col in ["sales_purch", "purch_sales", "sales_vat", "vat_sales", "purch_vat", "vat_purch"]]
        median_tin.drop(columns=ind_cols, inplace=True)

        return median_tin

    def interpolate(self):
        df = self.complete_df.copy()
        median_tin_df = self.median_tin_df.copy()

        mask_sales = df["total_sales"].isna()
        mask_purch = df["total_purch"].isna()
        mask_vat = df["net_payable"].isna()

        df = pd.merge(df, median_tin_df, how="left", on="owner_tin", suffixes=("","_median"))  # this already has the median ratios, redo case 2 and extend to when sales is missing and purch is missint
        
       # --- Case 2: Two Missing ---
        # Case 2A: total_sales and total_purch missing, net_payable present.
        mask_case2a = (df["total_missing"] == 2) & mask_sales & mask_purch & (~mask_vat)
        df.loc[mask_case2a, "total_sales"] = df.loc[mask_case2a, "net_payable"] * df.loc[mask_case2a, "sales_vat_median"]
        df.loc[mask_case2a, "total_purch"] = df.loc[mask_case2a, "net_payable"] * df.loc[mask_case2a, "purch_vat_median"]
        
        # Case 2B: total_sales and net_payable missing, total_purch present.
        mask_case2b = (df["total_missing"] == 2) & mask_sales & mask_vat & (~mask_purch)
        # Derive net_payable from total_purch using the median purch_vat ratio:
        df.loc[mask_case2b, "net_payable"] = df.loc[mask_case2b, "total_purch"] / df.loc[mask_case2b, "purch_vat_median"]
        # Then impute total_sales:
        df.loc[mask_case2b, "total_sales"] = df.loc[mask_case2b, "net_payable"] * df.loc[mask_case2b, "sales_vat_median"]
        
        # Case 2C: total_purch and net_payable missing, total_sales present.
        mask_case2c = (df["total_missing"] == 2) & mask_purch & mask_vat & (~mask_sales)
        # Derive net_payable from total_sales using the median sales_vat ratio:
        df.loc[mask_case2c, "net_payable"] = df.loc[mask_case2c, "total_sales"] / df.loc[mask_case2c, "sales_vat_median"]
        # Then impute total_purch:
        df.loc[mask_case2c, "total_purch"] = df.loc[mask_case2c, "net_payable"] * df.loc[mask_case2c, "purch_vat_median"]
        
        # --- Case 3: 1 missing ---
        # If vat is missing: net_payable = (total_sales - total_purch) * 0.12
        mask_case3_vat = (df["total_missing"] == 1) & mask_vat & (~mask_sales) & (~mask_purch)
        df.loc[mask_case3_vat, "net_payable"] = (df.loc[mask_case3_vat, "total_sales"] - df.loc[mask_case3_vat, "total_purch"]) * 0.12

        # If sales is missing: total_sales = (net_payable / 0.12) + total_purch
        mask_case3_sales = (df["total_missing"] == 1) & mask_sales & (~mask_vat) & (~mask_purch)
        df.loc[mask_case3_sales, "total_sales"] = (df.loc[mask_case3_sales, "net_payable"] / 0.12) + df.loc[mask_case3_sales, "total_purch"]

        # If purch is missing: total_purch = total_sales - (net_payable / 0.12)
        mask_case3_purch = (df["total_missing"] == 1) & mask_purch & (~mask_vat) & (~mask_sales)
        df.loc[mask_case3_purch, "total_purch"] = df.loc[mask_case3_purch, "total_sales"] - (df.loc[mask_case3_purch, "net_payable"] / 0.12)
        
        # --- Case 1: 3 missing ---
        # For rows with all three values missing, we forward fill later.
        # Sort by time and fill per company.
        df = df.sort_values(by=["owner_tin", "tax_year", "qtr"])
        df[["total_sales", "total_purch", "net_payable"]] = (
            df.groupby("owner_tin")[["total_sales", "total_purch", "net_payable"]].ffill()
        )
        
        # Cleanup: drop helper columns
        df = df.drop(columns=["total_missing"], errors="ignore")

        df.dropna(subset=["total_sales", "total_purch", "net_payable"], how="all", inplace=True)
        
        df["amended_yn"] = df["amended_yn"].fillna(0)
        df["penalties"] = df["penalties"].fillna(0)

        df["sales_purch"] = df["total_sales"] / df["total_purch"]
        df["purch_sales"] = 1 / df["sales_purch"]
        df["sales_vat"] = df["total_sales"] / df["net_payable"]
        df["vat_sales"] = 1 / df["sales_vat"]
        df["purch_vat"] = df["total_purch"] / df["net_payable"]
        df["vat_purch"] = 1 / df["purch_vat"]
        
        return df
    
    # ===============================================================
# ZScore Calculation using Median and MAD
# ===============================================================
class ZScore:
    @staticmethod
    def compute_zscore(df: pd.DataFrame, value_col: str, group_col: str = "owner_tin") -> pd.DataFrame:
        """
        Computes two z-score features for the specified value column:
         - Global (industry-level) z-score: (x - median) / MAD (computed over the entire dataset)
         - Personal (per-owner) z-score: (x - owner_median) / owner_MAD
        
        Assumes that the input column contains numeric values.
        """
        df = df.copy()
        # Global z-score: median and MAD over the entire dataset (ignoring NaNs)
        global_med = df[value_col].median()
        global_mad = np.median(np.abs(df[value_col] - global_med))
        ind_zscore_col = f"{value_col}_ind_zscore"
        df[ind_zscore_col] = (df[value_col] - global_med) / global_mad
        
        # Personal z-score: compute median and MAD per owner (grouped by group_col)
        personal_med = df.groupby(group_col)[value_col].median().rename(f"med_{value_col}")
        df = df.merge(personal_med, on=group_col, how="left")
        # Compute absolute deviation from the owner median
        df[f"abs_diff_{value_col}"] = np.abs(df[value_col] - df[f"med_{value_col}"])
        personal_mad = df.groupby(group_col)[f"abs_diff_{value_col}"].median().rename(f"mad_{value_col}")
        df = df.merge(personal_mad, on=group_col, how="left")
        per_zscore_col = f"{value_col}_per_zscore"
        df[per_zscore_col] = (df[value_col] - df[f"med_{value_col}"]) / df[f"mad_{value_col}"]
        
        # Optionally drop the intermediate columns (if not needed for further analysis)
        df.drop(columns=[f"med_{value_col}", f"abs_diff_{value_col}", f"mad_{value_col}"], inplace=True)
        
        return df

    @staticmethod
    def add_zscores(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds z-score features for total_sales, total_purch, and net_payable.
        """
        for col in ["total_sales", "total_purch", "net_payable"]:
            df = ZScore.compute_zscore(df, col)
        return df


# ===============================================================
# Benford Analysis
# ===============================================================
class Benford:
    @staticmethod
    def analyze_column(series: pd.Series) -> str:
        """
        Performs Benford analysis on a numeric series by:
          1. Dropping NaNs and converting values to strings.
          2. Extracting the first non-zero digit.
          3. Computing the relative frequency of digit "1".
          
        Returns:
          - "Close conformity" if frequency of "1" > 25%
          - "Acceptable conformity" if frequency > 15%
          - "Nonconformity" otherwise.
        """
        if series.empty:
            return "no file"
        # Remove NaNs and convert values to string
        series = series.dropna().astype(str)
        # Extract the first non-zero digit (ignoring any leading zeros and decimal points)
        leading_digits = series.apply(lambda x: x.lstrip("0.")[0] if x.lstrip("0.") else None)
        counts = leading_digits.value_counts(normalize=True)
        if counts.get('1', 0) > 0.25:
            return "Close conformity"
        elif counts.get('1', 0) > 0.15:
            return "Acceptable conformity"
        else:
            return "Nonconformity"

    @staticmethod
    def score(conformity: str) -> int:
        """
        Maps a Benford conformity string to a numeric score.
        """
        mapping = {
            "Close conformity": 2,
            "Acceptable conformity": 1,
            "Nonconformity": -1,
            "no file": 0
        }
        return mapping.get(conformity, np.nan)
    
    @staticmethod
    def add_benford_scores(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Benford analysis scores for total_sales and total_purch on a per-owner basis.
        Returns a DataFrame with owner-level Benford features.
        """
        df = df.copy()
        owners = df['owner_tin'].unique()
        results = []
        for tin in owners:
            sub = df[df['owner_tin'] == tin]
            sales_conf = Benford.analyze_column(sub["total_sales"])
            purch_conf = Benford.analyze_column(sub["total_purch"])
            results.append({
                "owner_tin": tin,
                "benford_sales_conformity": sales_conf,
                "benford_sales_score": Benford.score(sales_conf),
                "benford_purch_conformity": purch_conf,
                "benford_purch_score": Benford.score(purch_conf)
            })
        ben_df = pd.DataFrame(results)
        return ben_df


# ===============================================================
# Main Function: Create Feature Set for Fraud Analysis
# ===============================================================
def create_independent_vars(input_df: pd.DataFrame,
                            tax_year: Union[int, None] = None,
                            tax_qtr: Union[int, None] = None) -> pd.DataFrame:
    """
    Reads the fraud analysis dataset (with columns such as:
      owner_tin, tax_year, qtr, total_sales, total_purch, net_payable, 
      amended_yn, penalties, industry_group_id, industry_group_desc, etc.)
    and adds new feature columns based on:
      - Z-scores (industry-level and per-owner) for total_sales, total_purch, net_payable.
      - Benford analysis scores for total_sales and total_purch.
    
    Optionally, the dataset can be filtered to a specific tax_year and tax_qtr.
    """
    df = input_df.copy()
    
    # Optionally filter by tax_year and quarter if provided
    if tax_year is not None and tax_qtr is not None:
        df = df[(df['tax_year'] == str(tax_year)) & (df['qtr'] == tax_qtr)].copy()
    
    # Add ZScore features
    df = ZScore.add_zscores(df)
    
    # Compute Benford scores on a per-owner basis and merge back in
    ben_df = Benford.add_benford_scores(df)
    df = df.merge(ben_df, on="owner_tin", how="left")
    
    return df

# %% [markdown]
# # Further preprocessing, feature extraction, and assigning labels

# %%
df_1 = pd.read_csv("processed_data/initial_feats.csv", index_col=0)
df_2 = pd.read_csv("processed_data/tard_feats.csv", index_col=0)
 
known_legit = df_2["owner_tin"].unique().tolist()

df = pd.concat([df_1, df_2])
df.drop(columns=["total_sales", "total_purch", "net_payable", "penalties", "missing_total_sales", "missing_total_purch", "missing_net_payable", "industry_group_id_median", "purch_sales_median", "sales_purch_median", "sales_vat_median", "vat_sales_median", "purch_vat_median", "vat_purch_median", "total_purch_ind_zscore", "net_payable_ind_zscore", "benford_sales_conformity","benford_purch_conformity", "industry_group_id", "industry_group_desc"], inplace=True)
df["owner_tin"] = df["owner_tin"].astype(int)

known_fraud = pd.read_csv("raw_data/Known_Fraud_TINs.csv")
known_fraud = list(known_fraud["TIN"])
known_label = known_legit + known_fraud

df["labeled"] = df["owner_tin"].isin(known_label).astype(int)
df
df["fraud"] = np.nan
df.loc[(df["owner_tin"].isin(known_fraud), "fraud")] = 1
df.loc[(df["owner_tin"].isin(known_legit), "fraud")] = 0

# %% [markdown]
# # Train-test split

# %%
fraud_val, main_train_full = data_splitting(df, 0)
fraud_val.drop(columns=["total_sales", "total_purch", "net_payable", "penalties", "missing_total_sales", "missing_total_purch", "missing_net_payable", "industry_group_id_median", "purch_sales_median", "sales_purch_median", "sales_vat_median", "vat_sales_median", "purch_vat_median", "vat_purch_median", "total_purch_ind_zscore", "net_payable_ind_zscore", "benford_sales_conformity","benford_purch_conformity", "industry_group_id", "industry_group_desc"], errors="ignore", inplace=True)
val_labels = fraud_val["fraud"]
fraud_val.drop(columns=["owner_tin", "tax_year", "qtr", "dummy", "labeled", "fraud"], errors='ignore', inplace=True)

exclude = "fraud"
fill_dict = {col: 0 for col in fraud_val.columns if col != exclude}
fraud_val = fraud_val.fillna(value=fill_dict)

fraud_val.replace([np.inf], 999, inplace=True)
fraud_val.replace([-np.inf], -999, inplace=True)
exclude = "fraud"
fill_dict = {col: 0 for col in main_train_full.columns if col != exclude}
main_train_full = main_train_full.fillna(value=fill_dict)

main_train_full.replace([np.inf], 999, inplace=True)
main_train_full.replace([-np.inf], -999, inplace=True)

# %% [markdown]
# # Pseudolabelling, supervised learning

# %%
model = unsupervised_model(main_train_full, method="iso")

supervised_df = model.copy()
supervised_df["fraud_pred"] = supervised_df["fraud_pred"].map({"B": 1, "A": 0})
mask_labeled = main_train_full["labeled"] == 1
supervised_df["fraud"] = np.where(
    main_train_full["labeled"] == 1,
    main_train_full["fraud"],        # keep original labels
    supervised_df["fraud_pred"]    # otherwise use the model’s prediction
)

supervised_df.drop(columns=["fraud_pred"], inplace=True)
pseudolabels = pd.concat([main_train_full["fraud"], model["fraud_pred"]], axis=1)
confusion_matrix = pd.crosstab(pseudolabels["fraud"], pseudolabels["fraud_pred"])
confusion_matrix
pd.crosstab(pseudolabels["fraud_pred"], pseudolabels["fraud_pred"])
supervised_df
# model_s, pred_s = supervised_model(supervised_df, "xgb")

model_s_lst = []

for i in ["multi", "rf", "ann", "xgb"]:
    model_s, pred_s = supervised_model(supervised_df, i)
    a = [model_s, pred_s]
    model_s_lst.append(a)

ensemble = VotingClassifier(
    estimators=[
        ("multi", model_s_lst[0][0]),
        ("rf", model_s_lst[1][0]),
        # ("ann", model_s_lst[2][0]),
        ("xgb", model_s_lst[3][0]),
    ],
    voting="soft"
)

ensemble.fit(
    supervised_df.drop(columns=["fraud"], errors='ignore'),
    supervised_df["fraud"]
)
supervised_df.drop(columns=["fraud"])
pred_s = model_s_lst[3][0].predict(supervised_df.drop(columns=["fraud"]))
label_s = pd.concat([supervised_df["fraud"], pd.Series(pred_s, name="pred")], axis=1)

confusion_matrix_s = pd.crosstab(label_s["fraud"], label_s["pred"])
confusion_matrix_s
val_pred = model_s_lst[3][0].predict(fraud_val)
val_pred = np.where(val_pred > 0.5, 1, 0).flatten()
val_pred
val_labels.reset_index(drop=True,inplace=True)
val_pred_df = pd.concat([val_labels, pd.Series(val_pred, name="pred")], axis=1)

conf_matrix_s_val = pd.crosstab(val_pred_df["fraud"], val_pred_df["pred"])
pd.Series(val_pred, name="pred")

# %%
unl = supervised_df[main_train_full["labeled"] == 0]

# %%
unl["owner_tin"] = main_train_full[main_train_full["labeled"] == 0]["owner_tin"]
unl

# %%
unl_company = unl.groupby("owner_tin")["fraud"].mean().reset_index()
unl_company["fraud"] = np.where(unl_company["fraud"] > 0.5, 1, 0)
unl_company

# %%
association = pd.read_csv("processed_data/association_rules.csv", index_col=0)
association = association[(association["association"] == True) & (association["fraud"] == False)].reset_index(drop=True)
association

# %%
association_matrix = pd.merge(association, unl_company, on="owner_tin", how="outer")
association_matrix

# %%
association_matrix["fraud_y"] = association_matrix["fraud_y"].fillna("<N/A>")
association_matrix["association"] = association_matrix["association"].fillna("<N/A>")

# %%
pd.crosstab(association_matrix["association"], association_matrix["fraud_y"], )

# %% [markdown]
# # Load and preprocess out-of-sample (labeled) data

# %%
fraud_list = pd.read_excel("raw_data/Ateneo Batch 2 Request/Master List_Data Request for ADMU.xlsx")
fraud_list.rename(columns={"TIN": "owner_tin"}, inplace=True)
fraud_list["fraud"] = fraud_list.apply(lambda row: 1 if row["REMARKS"] == "Fraudulent" else 0, axis=1)
fraud_list.drop(columns=["NO.", "REMARKS"], inplace=True)

sales_lst = [
    "Ateneo Batch 2 Request/OCIR Data Request (30 Fraudulent TPs)/sls_1.xlsx"   
]
purchases_lst = [
    "Ateneo Batch 2 Request/OCIR Data Request (30 Fraudulent TPs)/slp_1.xlsx"
]
vat_lst = [
    "Ateneo Batch 2 Request/OCIR Data Request (30 Fraudulent TPs)/vat_1_2021.xlsx",
    "Ateneo Batch 2 Request/OCIR Data Request (30 Fraudulent TPs)/vat_1_2022.xlsx",
    "Ateneo Batch 2 Request/OCIR Data Request (30 Fraudulent TPs)/vat_1_2023.xlsx",
    "Ateneo Batch 2 Request/OCIR Data Request (30 Fraudulent TPs)/vat_2_2023.xlsx",
    "Ateneo Batch 2 Request/OCIR Data Request (30 Fraudulent TPs)/vat_1_2024.xlsx"  
]
ind_lst = [
    "Ateneo Batch 2 Request/OCIR Data Request (30 Fraudulent TPs)/Registration Details - OCIR Data Request (30 Taxpayers.xlsx"
]

sls_df = [pd.read_excel(f"raw_data/{sale}") for sale in sales_lst]
slp_df = [pd.read_excel(f"raw_data/{purch}") for purch in purchases_lst]
vat_df = [pd.read_excel(f"raw_data/{vat}") for vat in vat_lst]
ind_df = [pd.read_excel(f"raw_data/{ind}") for ind in ind_lst]
# sls_df = pd.read_excel("raw_data/Summary List of Sales.xls")

# %%
sls = SLS(sls_df)
slp = SLP(slp_df)
vat = VAT(vat_df)
ind = Industry(ind_df)
agg = Aggregate(sls=sls, slp=slp, vat=vat, industry=ind)

features_df = create_independent_vars(agg.interpolated_df.dropna())
features_df.drop(columns=["total_sales", "total_purch", "net_payable", "penalties", "missing_total_sales", "missing_total_purch", "missing_net_payable", "industry_group_id_median", "purch_sales_median", "sales_purch_median", "sales_vat_median", "vat_sales_median", "purch_vat_median", "vat_purch_median", "total_purch_ind_zscore", "net_payable_ind_zscore", "benford_sales_conformity","benford_purch_conformity", "industry_group_id", "industry_group_desc"], inplace=True)
features_df = pd.merge(features_df, fraud_list[["fraud", "owner_tin"]], on="owner_tin")

features_df_dropped = features_df.drop(columns=["owner_tin", "tax_year", "qtr", "dummy", "labeled", "fraud"], errors='ignore')

features_df_dropped.replace([np.nan, np.inf], 999, inplace=True)
features_df_dropped.replace([-np.nan, -np.inf], -999, inplace=True) 

oos_pred = [model_s_lst[i][0].predict(features_df_dropped) for i in range(4)]
oos_pred = pd.DataFrame(oos_pred).transpose()

oos_pred[2] = np.where(oos_pred[2]> 0.5, 1, 0)
oos_pred["fraud_final"] = oos_pred.mean(axis=1)
oos_pred["fraud_final"] = np.where(oos_pred["fraud_final"]>= 0.5, 1, 0)

conf_matrix_s = pd.concat([features_df["fraud"], oos_pred["fraud_final"]],axis=1)
conf_matrix_s.rename(columns={0: "fraud_pred"}, inplace=True)
pd.crosstab(conf_matrix_s["fraud"], conf_matrix_s["fraud_final"])
conf_matrix_s["owner_tin"] = features_df["owner_tin"]

company = conf_matrix_s.groupby("owner_tin").mean().reset_index()
company["fraud_pred"] = np.where(company["fraud_final"] > 0.4, 1, 0)

# %% [markdown]
# # Taxpayer-level classification performance (out-of-sample)

# %%
company

# %% [markdown]
# ## Confusion matrix

# %%
conf_matrix_oos = pd.crosstab(company["fraud"], company["fraud_pred"])
conf_matrix_oos

# %%
acc_oos = conf_matrix_oos[1].values[0] / conf_matrix_oos.sum(axis=1).values[0]
f"Out-of-sample accuracy: {acc_oos:.1%}"

# %%
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

reg_data = pd.read_excel("raw_data/TRS_REG.xls")
efps = pd.read_excel("raw_data/VAT_RETURNS_EFPS.xls")
tard_sls = pd.read_excel("raw_data/TARD/TARD SLS.xlsx")
tard_slp = pd.read_excel("raw_data/TARD/TARD SLP.xlsx")
tard_vat = pd.read_excel("raw_data/TARD/TARD VAT.xlsx")


tard_sls = SLS(tard_sls)
tard_slp = SLP(tard_slp)
tard_vat = VAT(tard_vat)

hello = pd.read_csv("hello_3.csv")
hello = pd.read_excel("raw_data/TARD/TARD VAT.xlsx")
hello["INDUSTRY_GROUP_ID"] = np.floor(hello["PSIC"]/10)
hello["INDUSTRY_GROUP_DESC"] = ""
hello = hello[["TIN", "INDUSTRY_GROUP_DESC", "INDUSTRY_GROUP_ID"]].drop_duplicates().reset_index()
tard_ind = Industry(hello)

tard_agg = Aggregate(
    sls=tard_sls,
    slp=tard_slp,
    vat=tard_vat,
    industry=tard_ind
    )

hello_tard = create_independent_vars(tard_agg.interpolated_df)
hello_tard.to_csv("processed_data/tard_feats.csv")

known_fraud = pd.read_csv("raw_data/Known_Fraud_TINs.csv")
known_fraud = set(known_fraud["TIN"])

sls_txn = pd.concat([sls.txn_df, tard_sls.txn_df])
sls_txn["pur_tin"] = pd.to_numeric(sls_txn["pur_tin"], errors="coerce")
sls_txn["owner_tin"] = pd.to_numeric(sls_txn["owner_tin"], errors="coerce")

slp_txn = pd.concat([slp.txn_df, tard_slp.txn_df])
slp_txn["sel_tin"] = pd.to_numeric(slp_txn["sel_tin"], errors="coerce")
slp_txn["owner_tin"] = pd.to_numeric(slp_txn["owner_tin"], errors="coerce")
slsp_txn = pd.concat([slp_txn.rename(columns={"sel_tin": "counterparty_tin"}), sls_txn.rename(columns={"pur_tin": "counterparty_tin"})])
slsp_txn = slsp_txn[slsp_txn["owner_tin"].isin(known_fraud) | slsp_txn["counterparty_tin"].isin(known_fraud)]
slsp_txn.dropna(inplace=True)

slsp_txn["counterparty_tin"] = slsp_txn["counterparty_tin"].apply(lambda x: int(x))
slp_cross = pd.crosstab(slsp_txn["owner_tin"], slsp_txn["counterparty_tin"], slsp_txn["total"], aggfunc=np.nansum)
slp_cross.fillna(0, inplace=True)
slsp_txn = slsp_txn.reset_index(drop=True)
slp_x = slsp_txn.loc[slsp_txn.index.repeat(slsp_txn['count'])].reset_index(drop=True)

slp_x = pd.read_csv("slp_x.csv")
slsp_x = slp_x[['owner_tin',  'counterparty_tin']].values.tolist()


te = TransactionEncoder()
te_ary = te.fit(slsp_x).transform(slsp_x)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)

# %%
min_support = 0.10
frequent_itemsets = apriori(slp_cross, min_support=min_support, use_colnames=True)
min_confidence = 0.15
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules.sort_values(by="lift", ascending=False).reset_index(drop=True)
hello_set = frozenset().union(*rules["antecedents"], *rules["consequents"])
hello_set
rules.apply(lambda row: row["consequents"], axis=1)
rules["association"] = rules.apply(lambda row: row["consequents"].union(row["antecedents"]), axis=1)
rules
rules['has_id'] = rules.apply(
    lambda row: 0 if (row['association'].isdisjoint(set(known_fraud)))
               else 1,
    axis=1
)

rules.sort_values("lift", ascending=False)
assoc_dict = dict()

for tin in hello_set:
    assoc_dict[tin] = rules.apply(lambda row: True if (tin in row["association"]) and (row["has_id"] == 1) else False, axis=1).any()

association = pd.DataFrame({"owner_tin": assoc_dict.keys(), "association": assoc_dict.values()})
association["fraud"] = association["owner_tin"].isin(known_fraud)
association
association[(association["association"] == True) & (association["fraud"] == False)]
association.to_csv("processed_data/association_rules.csv")


