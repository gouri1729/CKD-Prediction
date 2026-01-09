import pandas as pd
df=pd.read_excel("CKD_dataset.xls")
# print(df.columns)

df['Target']=df['Target'].replace(
    {
        'High risk':'High Risk',
        'High-risk':'High Risk'
    }
)
print(df['Target'].unique())

# No null values present 
# print(df.isnull().values.any())
# print("/n null values per column: ",df.isnull().sum())

num_cols= df.select_dtypes(include=['int64','float64']).columns
cat_cols= df.select_dtypes(include=['object']).columns

# print(cat_cols)

# #categorical to numerical conversion
# from sklearn.preprocessing import LabelEncoder

# # Encode feature categorical columns (excluding Target)
# feature_cat_cols = [col for col in cat_cols if col != 'Target']

# le = LabelEncoder()
# for col in feature_cat_cols:
#     df[col] = le.fit_transform(df[col])

# # Encode Target separately
# target_encoder = LabelEncoder()
# df['Target'] = target_encoder.fit_transform(df['Target'])

# print("Target class mapping:", target_encoder.classes_)

# # print(df.dtypes)

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
        'Low Risk': 1,
        'Moderate Risk': 2,
        'High Risk': 3,
        'Severe Disease': 4
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

nominal_cols = ['Smoking status']  # add more only if needed

df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

print("Final data types:")
print(df.dtypes)

print("\nAny missing values?")
print(df.isnull().sum().sort_values(ascending=False))

print("\nFinal dataset shape:", df.shape)
print(df.head())








# Dividing into training and testing sets
X= df.drop ('Target',axis=1)
Y= df['Target']


#Train-Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

#Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#SMOTE
from imblearn.over_sampling import SMOTE
import pandas as pd

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check balance
print("Before SMOTE:\n", pd.Series(y_train).value_counts())
print("\nAfter SMOTE:\n", pd.Series(y_train_smote).value_counts())


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=42)

# Select top 10 features
rfe = RFE(estimator=rfe_estimator, n_features_to_select=10)

X_train_rfe = rfe.fit_transform(X_train_smote, y_train_smote)
X_test_rfe  = rfe.transform(X_test)

print("Shape after RFE:", X_train_rfe.shape)

# PCA - Dimensionality Reduction
from sklearn.decomposition import PCA

# Reduce to 5 principal components
# pca = PCA(n_components=5)

# X_train_pca = pca.fit_transform(X_train_rfe)
# X_test_pca  = pca.transform(X_test_rfe)

# print("Shape after PCA:", X_train_pca.shape)

# Train Model (Random Forest)
model = RandomForestClassifier(n_estimators=300, random_state=42)
# model.fit(X_train_pca, y_train_smote)

# y_pred = model.predict(X_test_pca)
model.fit(X_train_rfe, y_train_smote)
y_pred = model.predict(X_test_rfe)


# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print(df['Target'].value_counts())

from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train_rfe, y_train_smote)

svm_pred = svm_model.predict(X_test_rfe)

print("\n========== SVM RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("\nClassification Report:\n", classification_report(y_test, svm_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, svm_pred))

