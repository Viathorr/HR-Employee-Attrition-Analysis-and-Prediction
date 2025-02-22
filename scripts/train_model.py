import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Data Loading
DATA_PATH = os.path.join( "data", "processed", "hr_employee_attrition.csv")

df = pd.read_csv(DATA_PATH)

X, y = df.drop(columns=["Attrition"]), df["Attrition"]  
X.shape, y.shape

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))

# Compute Class Weights to deal with unbalanced data
class_names = np.array([0, 1])

class_weights = compute_class_weight(class_weight="balanced", classes=class_names, y=y_train) 
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weight_dict)


# Training Logistic Regression Model with RandomizedSearchCV
model = LogisticRegression(max_iter=500, class_weight=class_weight_dict, random_state=42)

param_dist = {
    "penalty": ["l1", "l2", "elasticnet", None],  
    "C": loguniform(1e-4, 1e2), 
    "solver": ["liblinear", "saga"], 
    "l1_ratio": loguniform(0.01, 1) 
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,
    scoring="f1_macro",
    cv=4,  
    verbose=0,
    random_state=42,
    n_jobs=-1, 
)

random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)
print("Best accuracy score: ", random_search.best_score_)


# Training Logistic Regression Model with best parameters
best_model = LogisticRegression(**random_search.best_params_, class_weight=class_weight_dict, random_state=42)  
best_model.fit(X_train, y_train)

# Evaluate model
class_names = ["No", "Yes"]

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))

y_probs = best_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_probs)
print(f"ROC AUC Score: {auc_score:.2f}")

# Save model
joblib.dump(best_model, os.path.abspath(os.path.join("models", "best_logistic_regression_model.joblib")))

