import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from preprocess import load_data, split_data, split_features_target

def evaluate_model():
    data = load_data()
    x, y = split_features_target(data)
    x_train, x_test, y_train, y_test = split_data(x, y)
    model = joblib.load('/mnt/c/Learning/project/project for apply/predict phone addiction/save_models/lasso_model.pkl')
    y_pred = model.predict(x_test)
    print('MSE: ',  mean_squared_error(y_test, y_pred))
    print('MAE: ',  mean_absolute_error(y_test, y_pred))
    print('R2: ',  r2_score(y_test, y_pred))

if __name__ == '__main__':
    evaluate_model()