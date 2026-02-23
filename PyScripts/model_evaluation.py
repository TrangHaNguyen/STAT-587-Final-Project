import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.model_selection import TimeSeriesSplit

def classification_accuracy(predictions, actuals, correction: bool =False) -> tuple[float, float]:
    if (correction):
        predictions=(pd.Series(predictions)>0).astype(int).to_numpy()
    avg_pred_direction=np.mean(predictions)
    accuracy=np.mean(predictions==actuals)
    return accuracy, avg_pred_direction

def display_feat_importances(model, X: pd.DataFrame, n: int =50) -> None:
    model_feature_df=pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    model_feature_df.head(n=n).plot(kind='barh', x="Feature", y="Importance")
    plt.xlabel("Feature Importance")
    plt.xticks(rotation=45)
    plt.ylabel("Feature Name")
    plt.show()

def display_feat_importances_logistic(model, X: pd.DataFrame, n: int =50) -> None:
    importances = np.abs(model.coef_[0]) 

    model_feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    model_feature_df.head(n=n).plot(kind='barh', x="Feature", y="Importance")
    plt.gca().invert_yaxis() # Often looks better for barh
    plt.show()
    
def classification_cv_eval(model, X_train: pd.DataFrame, y_train: pd.DataFrame, k: int =10) -> None:
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    scoring_metrics = ['accuracy', 'precision', 'recall']
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, n_jobs=-1)
    results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring_metrics, n_jobs=-1)

    print(f"Average Accuracy:   {results['test_accuracy'].mean():.2%}")
    print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")
    print(f"Average Precision:  {results['test_precision'].mean():.2%}")
    print(f"Average Recall:     {results['test_recall'].mean():.2%}")

# def regression_cv_eval(model, X_train: pd.DataFrame, y_train: pd.DataFrame, k: int =10) -> None:
#     kf = KFold(n_splits=k, shuffle=True, random_state=1)
#     scoring_metrics = ['r2', 'neg_mean_squared_error']
#     results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring_metrics)

#     print(f"Avg. R-squared:        {results['test_r2'].mean():.2}")
#     print(f"Std. Dev. R-squared:   {results['test_r2'].std():.2f}%")
#     n=X_train.shape[0]
#     p=X_train.shape[1]
#     if p+1<n:
#         print(f"Avg. Adj. R-squared:   {1-(1-results['test_r2'].mean())*(n-1)/(n-p-1)}")
#     print(f"Average Error (MSE):   {abs(results['test_neg_mean_squared_error'].mean())}")
#     print(f"Average Error (RMSE):  {abs(results['test_neg_mean_squared_error'].mean()) ** 0.5:.4f}")

def classification_wfv_eval(model, X_train: pd.DataFrame, y_train: pd.DataFrame, n: int =24, max_train_size: int =21, test_size: int =5):
    tscv=TimeSeriesSplit(n_splits=n, max_train_size=max_train_size, test_size=test_size)
    scoring_metrics = ['accuracy', 'precision', 'recall']
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, n_jobs=-1)
    results = cross_validate(model, X_train, y_train, cv=tscv, scoring=scoring_metrics)

    print(f"Average Accuracy:   {results['test_accuracy'].mean():.2%}")
    print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")
    print(f"Average Precision:  {results['test_precision'].mean():.2%}")
    print(f"Average Recall:     {results['test_recall'].mean():.2%}")