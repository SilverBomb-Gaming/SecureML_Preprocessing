import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(df, method="standard"):
    """
    Scales numerical features in the dataset using either StandardScaler or MinMaxScaler.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        method (str): Scaling method - "standard" (default) or "minmax".
    
    Returns:
        pd.DataFrame: Scaled DataFrame.
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns  # Select only numerical columns
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_scaled

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def encode_categorical(df, method="onehot"):
    """
    Encodes categorical features using One-Hot Encoding or Label Encoding.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        method (str): Encoding method - "onehot" (default) or "label".
    
    Returns:
        pd.DataFrame: Encoded DataFrame.
    """
    df_encoded = df.copy()
    
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns  
    
    if method == "onehot":
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    elif method == "label":
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le  # Store encoders if needed later

    return df_encoded

from sklearn.feature_selection import VarianceThreshold

def select_features(df, correlation_threshold=0.8):
    """
    Selects important features by removing low-variance features and highly correlated features.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        correlation_threshold (float): Correlation threshold for removing redundant features.

    Returns:
        pd.DataFrame: Reduced feature set.
    """
    df_selected = df.copy()

    # Step 1: Remove Low-Variance Features
    selector = VarianceThreshold(threshold=0.01)  # Remove features with very little variance
    df_selected = df_selected.loc[:, selector.fit(df_selected).get_support()]

    # Step 2: Remove Highly Correlated Features
    corr_matrix = df_selected.corr().abs()
    upper_triangle = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > correlation_threshold)]
    df_selected.drop(columns=to_drop, inplace=True)

    return df_selected

def select_features(df, threshold=0.8):
    """ Select features by removing highly correlated ones """
    corr_matrix = df.corr()
    upper_triangle = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return df.drop(columns=to_drop, errors="ignore")

def save_preprocessed_data(df, filename="preprocessed_data.csv"):
    """Saves the preprocessed DataFrame to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"📂 Data saved successfully as '{filename}'")

