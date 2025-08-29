import joblib
import pandas as pd

MODEL_PATH = r"C:\Learning\project\project for apply\predict phone addiction\save_models\lasso_model.pkl"
CSV_PATH = r"C:\Learning\project\project for apply\predict phone addiction\data\teen_phone_addiction_dataset.csv"

def predict_new(sample_dict):
    # Load model
    model = joblib.load(MODEL_PATH)

    # Lấy danh sách feature numeric từ dataset
    data = pd.read_csv(CSV_PATH)
    feature_names = data.select_dtypes(include="number").drop("Addiction_Level", axis=1).columns

    # Bảo đảm sample có đủ cột, nếu thiếu sẽ điền 0
    input_df = pd.DataFrame([sample_dict])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]
    return prediction

if __name__ == "__main__":
    # ⚠️ PHẢI có đủ 17 cột numeric
    sample = {
        "Daily_Usage_Hours": 5.0,
        "Sleep_Hours": 6.0,
        "Academic_Performance": 75,
        "Social_Interactions": 5,
        "Exercise_Hours": 1.0,
        "Anxiety_Level": 6,
        "Depression_Level": 4,
        "Self_Esteem": 7,
        "Parental_Control": 0,
        "Screen_Time_Before_Bed": 1.2,
        "Phone_Checks_Per_Day": 100,
        "Apps_Used_Daily": 15,
        "Time_on_Social_Media": 3.0,
        "Time_on_Gaming": 2.0,
        "Time_on_Education": 1.0,
        "Family_Communication": 4,
        "Weekend_Usage_Hours": 7.0
    }

    print("🔮 Addiction_Level dự đoán:", predict_new(sample))
