#CLustering-Based Undersampling applied on Random Forest

import pandas as pd
import numpy as np

df=pd.read_excel("CKD_dataset.xls")
print(df)

#Clean and fix
df["Target"]=df["Target"].str.strip().str.replace("High-risk","High risk")

df["Target"]=df["Target"].apply(
    lambda x: x if x=="No Disease" else "CKD"
)

print(df[df["Target"]=="CKD"])

#Encoding

ordinal_mappings = {
    'Appetite (good/poor)': {'poor': 0, 'good': 1},
    'Physical activity level': {'low': 0, 'moderate': 1, 'high': 2},

    # -------- BINARY TARGET --------
    'Target': {
        'No Disease': 0,
        'CKD':1
    }
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
nominal_cols = ['Smoking status']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

y=df['Target']
x = df.drop("Target", axis=1)
print(x.columns)

#Train-Test split
from sklearn.model_selection import train_test_split
X_temp, X_test, Y_temp, Y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Validation set split
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp,
    test_size=0.25,   # 0.25 of 80% = 20% total
    stratify=Y_temp,
    random_state=42
)

#Cluster based undersampling
print(Y_train.value_counts())

# Combine training features and target
train_df=pd.concat([X_train,Y_train],axis=1)
print("Training set distribution BEFORE balancing:")
print(train_df["Target"].value_counts())

majority = train_df[train_df["Target"] == 0].copy()
minority = train_df[train_df["Target"] == 1].copy()

print("Majority samples:", len(majority))
print("Minority samples:", len(minority))

#Not understood
X_majority = majority.drop("Target", axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_majority_scaled = scaler.fit_transform(X_majority)

from sklearn.cluster import KMeans

k = len(minority)

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_majority_scaled)

centers = kmeans.cluster_centers_

from sklearn.metrics import pairwise_distances_argmin_min

closest, _ = pairwise_distances_argmin_min(centers, X_majority_scaled)

undersampled_majority = majority.iloc[closest]
balanced_train = pd.concat([undersampled_majority, minority])

balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)

print("Training set distribution AFTER balancing:")
print(balanced_train["Target"].value_counts())

X_train_balanced = balanced_train.drop("Target", axis=1)
Y_train_balanced = balanced_train["Target"]

#Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42)
model.fit(X_train_balanced, Y_train_balanced)

print("Validation Results:")
print(classification_report(Y_val, model.predict(X_val)))

print("Test Results:")
print(classification_report(Y_test, model.predict(X_test)))

# Confusion Matrix - Validation Set
cm_val = confusion_matrix(Y_val, model.predict(X_val))
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=['No Disease', 'CKD'])
fig, ax = plt.subplots(figsize=(8, 6))
disp_val.plot(ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.show()

# Confusion Matrix - Test Set
cm_test = confusion_matrix(Y_test, model.predict(X_test))
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['No Disease', 'CKD'])
fig, ax = plt.subplots(figsize=(8, 6))
disp_test.plot(ax=ax, cmap='Greens')
ax.set_title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix

# Validation set
y_val_pred = model.predict(X_val)
cm_val = confusion_matrix(Y_val, y_val_pred)

print("\nConfusion Matrix - Validation Set")
print(cm_val)

# Test set
y_test_pred = model.predict(X_test)
cm_test = confusion_matrix(Y_test, y_test_pred)

print("\nConfusion Matrix - Test Set")
print(cm_test)