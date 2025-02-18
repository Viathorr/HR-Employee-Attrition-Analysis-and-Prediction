import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin 
from scipy.stats import mstats


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, log_transform: bool = False, winsorize_transform: bool = False, winsor_limits: tuple = (0.05, 0.05)):
    """
    Custom Feature Engineering Transformer.

    Args:
        log_transform (bool): Apply log transformation.
        winsorize_transform (bool): Apply winsorization to cap outliers.
        winsor_limits (tuple): Percentage of data to clip from both ends for winsorization.
    """
    self.log_transform = log_transform
    self.winsorize_transform = winsorize_transform
    self.winsor_limits = winsor_limits

  def fit(self, X: pd.DataFrame, y: pd.Series = None):
    return self  
  
  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy() 
    
    X["NEW_IncomeStability"] = X["MonthlyIncome"] / X["MonthlyRate"]
    X["NEW_OvertimeImpact"] = (X["OverTime"] == "Yes").astype(int) * X["JobSatisfaction"]
    X["NEW_AttritionRiskScore"] = (
        (X["OverTime"] == "Yes").astype(int) * 2 + 
        (4 - X["JobSatisfaction"]) + 
        (4 - X["WorkLifeBalance"]) + 
        (X["YearsSinceLastPromotion"] > 5).astype(int) * 2
    )
    X["NEW_FirstJob"] = (X["NumCompaniesWorked"] == 0).astype(int)
    X["NEW_FrequentTraveler"] = X["BusinessTravel"].apply(lambda x: 1 if x == "Travel_Frequently" else 0)
    X["NEW_MarriedWithHighWorkload"] = ((X["MaritalStatus"] == "Married") & (X["OverTime"] == "Yes")).astype(int)

    num_cols = [col for col in X.select_dtypes(exclude=["object"]).columns if X[col].nunique() > 25]
    cols_with_high_skew = [col for col in num_cols if X[col].skew() > 0.7 ]

    for feature in cols_with_high_skew:
        if self.winsorize_transform:
            X[feature] = mstats.winsorize(X[feature], limits=self.winsor_limits)
            
        if self.log_transform:
            X[feature] = np.log1p(X[feature])

    return X


class FeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_remove=None):
        """
        Constructor for FeatureRemover.

        Args:
            features_to_remove (list or None): List of features to remove from the dataset.
        """
        self.features_to_remove = features_to_remove 
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_remove)
      