from data.datasets.data_loader import load_data
from models.linear_regression.linear_model import LinearRegressionModel
from sklearn.model_selection import train_test_split
from utils.visualization import plot_pred_vs_true, print_metrics

file_path = "data/raw_data/UPDATED Dataset - Predictive Tool Development for Residential Solar Installation Duration - REV1.xlsx"
X_df, y, category_options = load_data(file_path, verbose = False)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

model = LinearRegressionModel(method="linear")
model.train(X_train, y_train)
y_pred = model.predict(X_test)

print("Linear Regression:")
print_metrics(y_test, y_pred, model=model, feature_names=X_df.columns)
plot_pred_vs_true(y_test, y_pred)
