import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing_and_cleaning import clean_data
from model_evaluation import ModelResults
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)

cwd=Path.cwd()
for _ in range(5): 
    if cwd.name!="STAT-587-Final-Project":
        cwd=cwd.parent
    else:
        break
else:
    raise FileNotFoundError("Could not find correct workspace folder.")

lookup_df = pd.read_csv(cwd / "PyScripts" / "stock_lookup_table.csv")

# X, y=clean_data()
X, y_regression = clean_data()
X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
def to_binary_class(y):
    return (y>=0).astype(int).to_numpy()
y_train=to_binary_class(y_train)
y_test=to_binary_class(y_test)

# Support Vector Machine Classification
print("------- Training Support Vector Machine")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=1, verbose=1, tol=1e-2)
svm_model.fit(X_train, y_train)
print("Finished Training SVM -------")

print("------- Applying SVM Model to Test Set")
y_pred_svm = svm_model.predict(X_test)
print("Finished Applying SVM Model -------")

print("------- SVM Model Performance")
accuracy = accuracy_score(y_test, y_pred_svm)
precision = precision_score(y_test, y_pred_svm)
recall = recall_score(y_test, y_pred_svm)
f1 = f1_score(y_test, y_pred_svm)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))
print("------- SVM Model Performance Completed -------")

# Store results in ModelResults for comparison with other models
print("------- Saving SVM Results for Model Comparison")
model_results = ModelResults()
model_results.add_result('Support Vector Machine (SVM)', accuracy, precision, recall, f1)
model_results.display_results()
model_results.save_results(cwd / "output" / "svm_results.csv")
print("------- SVM Results Saved -------")

# Export X and y_regression to files
print("------- Exporting Data to Files")
# Export as Parquet (much faster than CSV for large datasets)
X.to_parquet(cwd / "output" / "X_features.parquet", compression='gzip')
y_regression.to_frame().to_parquet(cwd / "output" / "y_regression.parquet", compression='gzip')
print("Finished Exporting Data -------")
print(f"X exported to: {cwd / 'output' / 'X_features.parquet'}")
print(f"y_regression exported to: {cwd / 'output' / 'y_regression.parquet'}")
