
from data.datasets.data_loader import load_data
from models.tree.tree_model import TreeRegressionModel
from utils.visualization import print_metrics, plot_pred_vs_true

X_df, y, _ = load_data("data/raw_data/UPDATED Dataset - Predictive Tool Development for Residential Solar Installation Duration - REV1.xlsx")  # 填写你的数据路径

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_df, y, test_size=0.2, random_state=42)

model_types = ["decision_tree", "random_forest", "gbdt"]

for method in model_types:
    print(f"\n========== Testing {method} ==========")
    if method == "decision_tree":
        model = TreeRegressionModel(method=method, max_depth=6, min_samples_leaf=3)
    else:
        model = TreeRegressionModel(method=method, n_estimators=100, max_depth=8, min_samples_leaf=3)
    model.train(X_train, y_train)
    y_pred = model.predict(X_val)
    print_metrics(y_val, y_pred, model=model.model, feature_names=X_df.columns)
    plot_pred_vs_true(y_val, y_pred, title=f"{method} Predicted vs. True")

    importances = model.feature_importances_
    if importances is not None:
        top_features = sorted(zip(X_df.columns, importances), key=lambda x: -x[1])
        print(f"Top 10 features for {method}:")
        for f, imp in top_features[:10]:
            print(f"{f}: {imp:.3f}")
    else:
        print("This model does not support feature importances.")