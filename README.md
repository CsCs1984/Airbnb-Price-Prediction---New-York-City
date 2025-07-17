# 🏡 Airbnb NYC Price Prediction (2024)

This project predicts **Airbnb rental prices in New York City** using the latest 2024 dataset from Kaggle.  
We use **RandomForestRegressor** to model the relationship between price and features like **neighbourhood, room type, minimum nights, and reviews**.

---

## 📂 Project Structure

- `data/` → Kaggle dataset (`new-york-dataset.csv`)  
- `notebooks/` → Jupyter/Colab notebook with full analysis and EDA  
- `src/model.py` → clean Python script for data cleaning and model training  
- `requirements.txt` → required libraries  

---

## 🚀 How to Run

1️⃣ Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Airbnb-NYC-Price-Prediction-2024.git
cd Airbnb-NYC-Price-Prediction-2024


## 📊 Results

- Cleaned and analyzed 2024 NYC Airbnb dataset  
- Explored price distribution across neighbourhoods and room types  
- Trained a **RandomForestRegressor** with good performance (MAE ~ …, RMSE ~ …)

---

## 🔮 Future Work

- Try **hyperparameter tuning** with GridSearchCV for better accuracy  
- Compare RandomForest with **XGBoost / LightGBM** models  
- Include more features like `availability_365` or `reviews_per_month`  
- Deploy the model as a simple web app (Streamlit or Flask)

