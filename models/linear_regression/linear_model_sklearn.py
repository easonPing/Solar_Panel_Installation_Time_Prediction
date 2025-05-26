import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV

class LinearRegressionModel:
    """
    A linear regression wrapper supporting standard LinearRegression, Ridge, and Lasso.
    Provides train, predict, score, save, and load methods.
    """

    def __init__(self, method="linear", alpha=0.2, auto_alpha=False, alphas=None, cv=5):
        """
        method: 'linear', 'ridge', 'lasso'
        auto_alpha: if True, use LassoCV or RidgeCV to tune alpha automatically
        alphas: list or array of candidate alphas to search
        cv: number of folds for cross-validation
        """
        if method == "ridge":
            if auto_alpha:
                self.model = RidgeCV(alphas=alphas, cv=cv, store_cv_values=True)
            else:
                self.model = Ridge(alpha=alpha)
        elif method == "lasso":
            if auto_alpha:
                self.model = LassoCV(alphas=alphas, cv=cv)
            else:
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

    @property
    def coef_(self):
        return getattr(self.model, "coef_", None)

    @property
    def intercept_(self):
        return getattr(self.model, "intercept_", None)