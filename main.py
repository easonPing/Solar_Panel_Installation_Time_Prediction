from data.datasets.data_loader import load_data
from models.linear_regression.linear_model import LinearRegressionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

file_path = "data/raw_data/UPDATED Dataset - Predictive Tool Development for Residential Solar Installation Duration - REV1.xlsx"
X_df, y, category_options = load_data(file_path, verbose=True)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

model = LinearRegressionModel(method="linear")
model.train(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))
print("First 10 predictions:", y_pred[:10])
print("First 10 true values:", y_test.head(10).values)
print("Coefficients:", model.get_coefficients(feature_names=X_df.columns))
print("Intercept:", model.get_intercept())