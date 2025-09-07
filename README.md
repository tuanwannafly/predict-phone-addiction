# 📱 Teen Phone Addiction Predictor

Day la project du doan muc do su dung dien thoai cua hoc sinh dua tren dataset tu kaggle.
Duoc xay tu **scikit-learn** va deploy tren **Streamlit**.

---

## Dataset 
link dataset: https://www.kaggle.com/datasets/sumedh1507/teen-phone-addiction
Tac gia : https://www.kaggle.com/sumedh1507


## 📂 Project Structure
predict-phone-addiction/
│── data/teen_phone_addiction_dataset.csv # dataset
│── save_models/lasso_model.pkl # trained model 
│── src/ # training & evaluation scripts
│ ├── preprocess.py
│ ├── train_model.py
│ ├── evaluate.py
│ └── predict.py
│── app.py # Streamlit app entry point
│── requirements.txt # dependencies
│── README.md # documentation


---

## 🚀 Run Locally
```bash
# 1. Clone repo
git clone https://github.com/tuanwannafly/predict-phone-addiction.git
cd predict-phone-addiction

# 2. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Train model
python src/train_model.py

# 5. Run Streamlit app
streamlit run app.py
👉 App will be available at http://localhost:8501


## Deploy
https:https://predict-phone-addiction-fsz4tqqyz2eonx66wuj5b8.streamlit.app/

📦 requirements.txt
streamlit
scikit-learn
pandas
numpy
joblib

✨ Usage
Via script: python src/predict.py

Via web app: streamlit run app.py

📜 License
MIT License – feel free to use and modify.


