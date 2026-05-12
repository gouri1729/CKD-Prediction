import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    #fixing inconsistent labelss
    df['Target'] = df['Target'].replace({
    'High risk': 'High Risk',
    'High-risk': 'High Risk'
    })

    df["Target"]=df["Target"].apply(
    lambda x: x if x=="No Disease" else "CKD"
    )
    # ordinal mappings 
    ordinal_mappings = {
        'Appetite (good/poor)': {
            'poor': 0,
            'good': 1
        },
        'Physical activity level': {
            'low': 0,
            'moderate': 1,
            'high': 2
        },
        'Target': {
            'No Disease': 0,
            'CKD': 1
        }
    }

    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping)

    # binary mappings
    binary_cols = {
        'Red blood cells in urine': {'normal': 0, 'abnormal': 1},
        'Pus cells in urine': {'normal': 0, 'abnormal': 1},
        'Pus cell clumps in urine': {'not present': 0, 'present': 1},
        'Bacteria in urine': {'not present': 0, 'present': 1},
        'Hypertension (yes/no)': {'no': 0, 'yes': 1},
        'Diabetes mellitus (yes/no)': {'no': 0, 'yes': 1},
        'Coronary artery disease (yes/no)': {'no': 0, 'yes': 1},
        'Pedal edema (yes/no)': {'no': 0, 'yes': 1},
        'Anemia (yes/no)': {'no': 0, 'yes': 1},
        'Family history of chronic kidney disease': {'no': 0, 'yes': 1},
        'Urinary sediment microscopy results': {'normal': 0, 'abnormal': 1}
    }
    for col, mapping in binary_cols.items():
        df[col] = df[col].map(mapping)

    
    # one-hot encoding
    nominal_cols = ['Smoking status']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    return df

def split_data(df, target_column): #splitting train-test data

    x = df.drop(target_column, axis=1)
    y = df[target_column]

    X_temp, X_test, Y_temp, Y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    #validation set split from training data
    X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp,
    test_size=0.125,   # 0.25 of 80% = 20% total
    stratify=Y_temp,
    random_state=42
    )

    return X_train, X_val, Y_train, Y_val

