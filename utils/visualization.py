import matplotlib.pyplot as plt
import numpy as np

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

def plot_feature_coefficients(feature_names, coefs, title="Feature Coefficients"):
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names), coefs)
    plt.title(title)
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()