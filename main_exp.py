from preprocessing import load_data, preprocess_data, split_data
from models import get_class_weighted_model
from evaluation import evaluate_model
from imbalance_methods import apply_smote

file="CKD_data.xls"
df = load_data(file)

df = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(df, "Target")
X_train_smote, y_train_smote = apply_smote(X_train, y_train) # apply smote on training data 

model = get_class_weighted_model()
# model.fit(X_train_smote, y_train_smote)
# results=evaluate_model(model, X_test, y_test)


from tuning import tune_random_forest

best_model, best_params = tune_random_forest(model, X_train_smote, y_train_smote)
print("Best Parameters:", best_params)
best_model.fit(X_train_smote, y_train_smote)






#Flow of the code:
#1. Load and preprocess data
#2. Split data into training and testing sets
#3. Apply SMOTE to handle class imbalance on training data
#4. Train a class-weighted model (e.g., Random Forest)
#5. Evaluate the model on the test set
