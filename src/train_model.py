import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from preprocess import load_data, split_data, split_features_target
import os



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "save_models", "lasso_model.pkl")

def train_and_save_model():
    data = load_data()
    x, y = split_features_target(data)
    x_train, x_test, y_train, y_test = split_data(x, y)
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter=1000)),
    ])

    param_grid = {"lasso__alpha": [0.001, 0.01, 0.1, 0.5, 1, 5, 10]}
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
    )

    grid_search.fit(x_train, y_train)

    print("Best Alpha: ", grid_search.best_params_["lasso__alpha"])
    print("Best CV (MSE): ", grid_search.best_score_)

    joblib.dump(grid_search, MODEL_PATH)
    print("Saved model in {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()