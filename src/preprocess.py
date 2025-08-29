import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Doc du lieu
def load_data(path='C:/Learning/project/project for apply/predict phone addiction/data/teen_phone_addiction_dataset.csv'):
    data = pd.read_csv(path)
    #Chi lay cac cot so
    data = data.select_dtypes(include=["number"])
    return data

def split_features_target(data):
    #Chon feature va label
    target = "Addiction_Level"
    x = data.drop(target, axis=1)
    y = data[target]

    return x, y

def split_data(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def scale_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test
