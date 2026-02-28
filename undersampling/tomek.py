import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_excel("CKD_dataset.xls")

# -------------------------------
# Clean Target Column
# -------------------------------
df["Target"] = df["Target"].str.strip().str.replace("High-risk", "High risk")

df["Target"] = df["Target"].apply(
    lambda x: x if x == "No Disease" else "CKD"
)

# -------------------------------
# Encoding
# -------------------------------

ordinal_mappings = {
    'Appetite (good/poor)': {'poor': 0, 'good': 1},
    'Physical activity level': {'low': 0, 'moderate': 1, 'high': 2},
    'Target': {'No Disease': 0, 'CKD': 1}
}

for col, mapping in ordinal_mappings.items():
    df[col] = df[col].map(mapping)

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

# One-hot encoding
df = pd.get_dummies(df, columns=['Smoking status'], drop_first=True)

# -------------------------------
# Separate Features & Target
# -------------------------------
y = df['Target']
X = df.drop("Target", axis=1)

# -------------------------------
# Train-Test Split (80-20)
# -------------------------------
from sklearn.model_selection import train_test_split

X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Train-Validation Split
# -------------------------------
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp,
    test_size=0.25,   # 0.25 of 80% = 20% total
    stratify=Y_temp,
    random_state=42
)

print("Training set BEFORE Tomek:")
print(Y_train.value_counts())

# -------------------------------
# Tomek Links Undersampling
# -------------------------------
from imblearn.under_sampling import TomekLinks

tomek = TomekLinks(sampling_strategy='majority')

X_train_balanced, Y_train_balanced = tomek.fit_resample(X_train, Y_train)

print("\nTraining set AFTER Tomek:")
print(pd.Series(Y_train_balanced).value_counts())

# -------------------------------
# Model Training
# -------------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',   # try None also as experiment
    random_state=42
)

model.fit(X_train_balanced, Y_train_balanced)

# -------------------------------
# Evaluation
# -------------------------------
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Validation
print("\nValidation Results:")
print(classification_report(Y_val, model.predict(X_val)))

# Test
print("\nTest Results:")
print(classification_report(Y_test, model.predict(X_test)))

# -------------------------------
# Confusion Matrix - Validation
# -------------------------------
cm_val = confusion_matrix(Y_val, model.predict(X_val))
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val,
                                  display_labels=['No Disease', 'CKD'])

fig, ax = plt.subplots(figsize=(6, 5))
disp_val.plot(ax=ax)
ax.set_title('Confusion Matrix - Validation')
plt.tight_layout()
plt.show()

# -------------------------------
# Confusion Matrix - Test
# -------------------------------
cm_test = confusion_matrix(Y_test, model.predict(X_test))
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test,
                                   display_labels=['No Disease', 'CKD'])

fig, ax = plt.subplots(figsize=(6, 5))
disp_test.plot(ax=ax)
ax.set_title('Confusion Matrix - Test')
plt.tight_layout()
plt.show()