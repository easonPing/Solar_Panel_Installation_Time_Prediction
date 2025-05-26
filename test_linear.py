from data.datasets.data_loader import load_data
from models.linear_regression.linear_model_sklearn import LinearRegressionModel
from sklearn.model_selection import train_test_split
from utils.visualization import plot_pred_vs_true, print_metrics, print_compliance_stats_with_xdf
import numpy as np

file_path = "data/raw_data/UPDATED Dataset - Predictive Tool Development for Residential Solar Installation Duration - REV1.xlsx"
X_df, y, category_options = load_data(file_path, verbose = False)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Linear Regression
model_linear = LinearRegressionModel(method="linear")
model_linear.train(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
print("\nLinear Regression:")
print_metrics(y_test, y_pred_linear, model=model_linear, feature_names=X_df.columns)
print_compliance_stats_with_xdf(y_test, y_pred_linear, X_test)
plot_pred_vs_true(y_test, y_pred_linear, title="Linear Regression: Pred vs True")

# Ridge Regression
model_ridge = LinearRegressionModel(method="ridge", alpha=1.0)
model_ridge.train(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)
print("\nRidge Regression (alpha=1.0):")
print_metrics(y_test, y_pred_ridge, model=model_ridge, feature_names=X_df.columns)
print_compliance_stats_with_xdf(y_test, y_pred_ridge, X_test)
plot_pred_vs_true(y_test, y_pred_ridge, title="Ridge Regression: Pred vs True")

# Lasso Regression
model_lasso = LinearRegressionModel(method="lasso", alpha=0.5)
model_lasso.train(X_train, y_train)
y_pred_lasso = model_lasso.predict(X_test)
print("\nLasso Regression (alpha=0.1):")
print_metrics(y_test, y_pred_lasso, model=model_lasso, feature_names=X_df.columns)
print_compliance_stats_with_xdf(y_test, y_pred_lasso, X_test)
plot_pred_vs_true(y_test, y_pred_lasso, title="Lasso Regression: Pred vs True")

# LassoCV (automatic hyperparameter tuning)
alphas = np.logspace(-3, 1, 30)
model_lassocv = LinearRegressionModel(method="lasso", auto_alpha=True, alphas=alphas, cv=5)
model_lassocv.train(X_train, y_train)
y_pred_lassocv = model_lassocv.predict(X_test)
print("\nLassoCV (auto alpha, 5-fold):")
print_metrics(y_test, y_pred_lassocv, model=model_lassocv, feature_names=X_df.columns)
print("Best alpha found by CV:", model_lassocv.model.alpha_)
print_compliance_stats_with_xdf(y_test, y_pred_lassocv, X_test)
plot_pred_vs_true(y_test, y_pred_lassocv, title="LassoCV: Pred vs True")



