import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso

class LinearRegressionModel:
    """
    A linear regression wrapper supporting standard LinearRegression, Ridge, and Lasso.
    Provides train, predict, score, save, and load methods.
    """

    def __init__(self, method="linear", alpha=1.0):
        """
        Initialize the linear model.
        method: 'linear', 'ridge', or 'lasso'
        alpha: regularization parameter for Ridge/Lasso
        """
        if method == "ridge":
            self.model = Ridge(alpha=alpha)
        elif method == "lasso":
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()

    def train(self, X, y):
        """
        Train the linear regression model on X (features) and y (target).
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict target values for given feature matrix X.
        Returns a numpy array of predictions.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Return the R^2 coefficient of determination of the prediction.
        """
        return self.model.score(X, y)

    def save(self, path):
        """
        Save the trained model to a file using joblib.
        """
        joblib.dump(self.model, path)

    def load(self, path):
        """
        Load a trained model from a file.
        """
        self.model = joblib.load(path)

    def get_coefficients(self, feature_names=None):
        """
        Return the model coefficients (and intercept) as a dict.
        If feature_names is provided, returns a mapping from name to coef.
        """
        if hasattr(self.model, "coef_"):
            if feature_names is not None:
                return dict(zip(feature_names, self.model.coef_))
            return self.model.coef_
        else:
            return None

    def get_intercept(self):
        """
        Return the model intercept.
        """
        return getattr(self.model, "intercept_", None)