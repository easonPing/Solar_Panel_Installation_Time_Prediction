
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

class TreeRegressionModel:
    """
    A general wrapper for tree-based regression models, supporting
    Random Forest, Decision Tree, and Gradient Boosting.
    """

    def __init__(self, method="random_forest", n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42):
        """
        Initialize the tree regression model.

        method: "random_forest", "decision_tree", or "gbdt"
        n_estimators: number of trees (only for RF and GBDT)
        max_depth: maximum tree depth
        min_samples_leaf: minimum samples per leaf
        random_state: random seed for reproducibility
        """
        self.method = method
        self.model = None
        if method == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1
            )
        elif method == "decision_tree":
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        elif method == "gbdt":
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown tree method: {method}")

    def train(self, X, y):
        """
        Fit the model to training data.

        X: feature matrix (DataFrame or ndarray)
        y: target vector
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict target values for X.

        X: feature matrix (DataFrame or ndarray)
        return: predicted values (ndarray)
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        X: feature matrix
        y: true target values
        return: R^2 score (float)
        """
        return self.model.score(X, y)

    def save(self, path):
        """
        Save the trained model to a file.

        path: output file path
        """
        joblib.dump(self.model, path)

    def load(self, path):
        """
        Load a model from a file.

        path: model file path
        """
        self.model = joblib.load(path)

    @property
    def feature_importances_(self):
        """
        Return the feature importances (if available).

        return: array of importance scores or None
        """
        return getattr(self.model, "feature_importances_", None)