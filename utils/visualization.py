import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def plot_pred_vs_true(y_true, y_pred, title="Predicted vs. True Values"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_metrics(y_true, y_pred, model = None, feature_names=None, topn_coef=10):
    """
    Print metrics and first 10 predictions.
    Print model coefficients and intercept directly from a model object.
    Only works for models with .coef_ and .intercept_ attributes (e.g., sklearn linear models).
    Prints top N coefficients by absolute value.
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 score:", r2_score(y_true, y_pred))
    print("First 10 predictions:", y_pred[:10])
    print("First 10 true values:", y_true.head(10).values)

    coefs = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    if coefs is not None:
        abs_coefs = np.abs(coefs)
        idx = abs_coefs.argsort()[::-1][:topn_coef]
        print(f"Top {topn_coef} coefficients (by absolute value):")
        if feature_names is not None:
            for i in idx:
                print(f"{feature_names[i]}: {coefs[i]:.4f}")
        else:
            for i in idx:
                print(f"Feature {i}: {coefs[i]:.4f}")
    else:
        print("No coefficients found in this model.")
    if intercept is not None:
        print("Intercept:", intercept)
    else:
        print("No intercept found in this model.")

def print_compliance_stats_with_xdf(y_true, y_pred, X_test, max_over=120, employee_col="Total # Hourly Employees on Site"):
    """
    Print the number and proportion of samples that satisfy:
    - No underestimation (predicted >= true)
    - Overestimation does NOT exceed max_over minutes per hourly employee

    x_test: containing 'Total # Hourly Employees on Site' as a column.
    max_over: maximum allowed overestimate per employee, in minutes (default 120 = 2 hours)
    employee_col: name of the hourly employees column in X_df

    Output: number and ratio of compliant samples.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_employees = np.array(X_test[employee_col])
    not_under = y_pred >= y_true
    not_over = (y_pred - y_true) <= (n_employees * max_over)
    compliant = not_under & not_over
    num_compliant = compliant.sum()
    total = len(y_true)
    print(f"Compliant predictions: {num_compliant}/{total} ({num_compliant/total:.1%})")
    print(f"- No underestimation, and overestimation ≤ {max_over} min per hourly employee.")
    return num_compliant, total, compliant