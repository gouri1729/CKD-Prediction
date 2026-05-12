import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline


from sklearn.feature_selection import RFE


from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline




# =====================================
# 2️⃣ Load Dataset
# =====================================


df = pd.read_excel("CKD_MODIFIED.xlsx", engine="openpyxl")


df.columns = df.columns.str.strip()
target_column = "Target"


print("Dataset Shape:", df.shape)




# =====================================
# 3️⃣ Clean Target Labels
# =====================================


df[target_column] = df[target_column].astype(str).str.strip()


df[target_column] = df[target_column].replace({
    "High risk": "High Risk",
    "High-risk": "High Risk"
})


df[target_column] = df[target_column].apply(
    lambda x: 0 if x in ["No Disease", "Non CKD"] else 1
)


print("\nTarget Distribution:")
print(df[target_column].value_counts())




# =====================================
# 4️⃣ Separate Features and Target
# =====================================


X = df.drop(target_column, axis=1)
y = df[target_column]




# =====================================
# 5️⃣ Identify Column Types
# =====================================


categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64','float64']).columns




# =====================================
# 6️⃣ Preprocessing Pipelines
# =====================================


num_pipeline = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])


cat_pipeline = SkPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ]
)




# =====================================
# 7️⃣ Train / Validation / Test Split
# =====================================


X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.125,
    stratify=y_temp,
    random_state=42
)


print("\nDataset Split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))




# =====================================
# Feature Selection (RFE)
# =====================================


rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    n_features_to_select=15
)




# =====================================
# Full Pipeline
# =====================================


model = Pipeline(steps=[


    ('preprocessing', preprocessor),


    # Feature Selection
    ('feature_selection', rfe),


    # SMOTE + Tomek Links
    ('smote_tomek', SMOTETomek(
        sampling_strategy=0.8,
        random_state=42
    )),


    # Random Forest Classifier
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    ))
])




# =====================================
#  Train Model
# =====================================


model.fit(X_train, y_train)




# =====================================
# Validation Evaluation
# =====================================


y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:,1]


print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))


print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))


print("\nValidation ROC-AUC Score:", roc_auc_score(y_val, y_val_prob))




# =====================================
#  Test Prediction
# =====================================


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]




# =====================================
#  Final Test Evaluation
# =====================================


print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nTest Classification Report:")
print(classification_report(y_test, y_pred))


print("\nTest ROC-AUC Score:", roc_auc_score(y_test, y_prob))




# # =====================================
# #  Show Selected Features from RFE
# # =====================================


# feature_names = model.named_steps['preprocessing'].get_feature_names_out()


# rfe_mask = model.named_steps['feature_selection'].support_


# selected_features = feature_names[rfe_mask]


# print("\nSelected Features by RFE:")
# for f in selected_features:
#     print(f)




# # =====================================
# #  Feature Ranking
# # =====================================


# ranking = model.named_steps['feature_selection'].ranking_


# feature_ranking = pd.DataFrame({
#     "Feature": feature_names,
#     "Rank": ranking
# }).sort_values(by="Rank")


# print("\nFeature Ranking (RFE):")
# print(feature_ranking)


#----
import shap

import matplotlib.pyplot as plt

print("\nRunning SHAP Explainability...")

# Extract pipeline steps
preprocessor = model.named_steps["preprocessing"]
rfe = model.named_steps["feature_selection"]
rf_model = model.named_steps["rf"]

# Transform training data using preprocessing
X_train_processed = preprocessor.transform(X_train)

# Apply RFE feature selection mask
X_train_selected = X_train_processed[:, rfe.support_]

# Get feature names
feature_names = preprocessor.get_feature_names_out()
selected_feature_names = feature_names[rfe.support_]

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_train_selected)

# ==============================
# SHAP Summary Plot (Global Importance)
# ==============================

shap.summary_plot(
    shap_values[:, :, 1],  # class 1 (CKD)
    X_train_selected,
    feature_names=selected_feature_names
)

# ==============================
# SHAP Bar Plot (Feature Importance)
# ==============================

shap.summary_plot(
    shap_values[:, :, 1],
    X_train_selected,
    feature_names=selected_feature_names,
    plot_type="bar"
)

# ==============================
# SHAP Force Plot (Single Patient)
# ==============================

sample_index = 0

shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][sample_index],
    X_train_selected[sample_index],
    feature_names=selected_feature_names,
    matplotlib=True
)


# from lime.lime_tabular import LimeTabularExplainer
# import numpy as np
# import matplotlib.pyplot as plt

# print("\nRunning LIME Explainability...")

# # Convert training data to numpy
# X_train_np = X_train.values
# X_test_np = X_test.values

# feature_names = X_train.columns.tolist()

# class_names = ["No CKD", "CKD"]

# # Create LIME explainer
# explainer = LimeTabularExplainer(
#     training_data=X_train_np,
#     feature_names=feature_names,
#     class_names=class_names,
#     mode="classification",
#     discretize_continuous=False   # important fix
# )

# # Choose instance to explain
# sample_index = 24

# exp = explainer.explain_instance(
#     X_test_np[sample_index],
#     model.predict_proba,   # full pipeline
#     num_features=10
# )

# print("\nLIME Explanation:")
# print(exp.as_list())

# # Plot explanation
# fig = exp.as_pyplot_figure()
# plt.title("LIME Explanation for Prediction")
# plt.show()