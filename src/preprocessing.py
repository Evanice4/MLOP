import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(csv_path)


def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Split DataFrame into features (X) and target (y).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline that:
      - Scales numeric features
      - One-hot encodes categorical features
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessors = []
    if numeric_features:
        preprocessors.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        preprocessors.append(("cat", categorical_transformer, categorical_features))

    if not preprocessors:
        raise ValueError(
            "No numeric or categorical features detected. Check your data."
        )

    preprocessor = ColumnTransformer(transformers=preprocessors)
    return preprocessor