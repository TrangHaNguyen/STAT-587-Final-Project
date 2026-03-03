import pandas as pd
from data_preprocessing_and_cleaning import clean_data
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from helper_functions import get_cwd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)
cwd = get_cwd("STAT-587-Final-Project")

# X, y=clean_data()
X, y_regression = clean_data()
X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
def to_binary_class(y):
    return (y>=0).astype(int).to_numpy()
y_train=to_binary_class(y_train)
y_test=to_binary_class(y_test)

# Support Vector Machine Classification
print("------- Training Support Vector Machine with 10-fold CV to select C")

# Use a pipeline with scaling and SVC. Perform 10-fold CV to select C.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(cache_size=1000, class_weight='balanced', gamma='scale', random_state=1, tol=5e-2))
])

param_grid = [
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.01, 0.1, 1, 10]
    },
    {
        'svc__kernel': ['rbf'],
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto', 0.001, 0.01]
    }
]

tscv=TimeSeriesSplit(n_splits=3)
grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='balanced_accuracy', n_jobs=-2, verbose=3, return_train_score=True)
grid.fit(X_train, y_train)

input("Press Enter to continue...")

print("Finished CV search. Best params:", grid.best_params_)
cv_results_df = pd.DataFrame(grid.cv_results_)
cv_results_df.to_csv(cwd / 'output' / 'svm_cv_results.csv', index=False)

best_model = grid.best_estimator_
print("Finished Training SVM (best estimator selected) -------")

print("------- Applying SVM Model to Test Set")
y_pred_svm = best_model.predict(X_test)
print("Finished Applying SVM Model -------")

print("------- SVM Model Performance")
accuracy = accuracy_score(y_test, y_pred_svm)
precision = precision_score(y_test, y_pred_svm, zero_division=0)
recall = recall_score(y_test, y_pred_svm, zero_division=0)
f1 = f1_score(y_test, y_pred_svm, zero_division=0)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, zero_division=0))
print("------- SVM Model Performance Completed -------")

# # # Store results in ModelResults for comparison with other models
# print("------- Saving SVM Results for Model Comparison")
# model_results = ModelResults()
# model_results.add_result('Support Vector Machine (SVM)', accuracy, precision, recall, f1)
# model_results.display_results()
# model_results.save_results(cwd / "output" / "svm_results.csv")
# print("------- SVM Results Saved -------")

