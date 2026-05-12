from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix inconsistent labels
    df["Target"] = df["Target"].replace({
        "High risk": "High Risk",
        "High-risk": "High Risk"
    })

    # Label mapping (binary classification)
    label_mapping = {
        "No Disease": 0,
        "Low Risk": 1,
        "Moderate Risk": 1,
        "High Risk": 1,
        "Severe Disease": 1
    }
    df["Target"] = df["Target"].map(label_mapping)

    # Ordinal encoding
    ordinal_mappings = {
        "Appetite (good/poor)": {"poor": 0, "good": 1},
        "Physical activity level": {"low": 0, "moderate": 1, "high": 2}
    }
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Binary encoding
    binary_cols = {
        "Red blood cells in urine": {"normal": 0, "abnormal": 1},
        "Pus cells in urine": {"normal": 0, "abnormal": 1},
        "Pus cell clumps in urine": {"not present": 0, "present": 1},
        "Bacteria in urine": {"not present": 0, "present": 1},
        "Hypertension (yes/no)": {"no": 0, "yes": 1},
        "Diabetes mellitus (yes/no)": {"no": 0, "yes": 1},
        "Coronary artery disease (yes/no)": {"no": 0, "yes": 1},
        "Pedal edema (yes/no)": {"no": 0, "yes": 1},
        "Anemia (yes/no)": {"no": 0, "yes": 1},
        "Family history of chronic kidney disease": {"no": 0, "yes": 1},
        "Urinary sediment microscopy results": {"normal": 0, "abnormal": 1}
    }
    for col, mapping in binary_cols.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Manual encoding for nominal column
    if "Smoking status" in df.columns:
        df["Smoking status"] = df["Smoking status"].map({"no": 0, "yes": 1})

    return df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.125,
    random_state: int = 42
) -> SplitData:
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_state
    )

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )
