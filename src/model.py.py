import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def load_and_clean_data(filepath):
    """Load and clean Airbnb dataset"""
    df = pd.read_csv(filepath)
    
    # تنظيف السعر
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df = df[(df['price'] >= 20) & (df['price'] <= 1000)]
    
    # إزالة minimum_nights الغير منطقية
    df = df[df['minimum_nights'] <= 365]
    
    # إزالة القيم الفارغة
    df.dropna(subset=['price', 'room_type', 'neighbourhood'], inplace=True)
    
    # اختيار الأعمدة المهمة
    df = df[['price', 'room_type', 'neighbourhood', 'minimum_nights', 'number_of_reviews']]
    
    # One-hot encoding للأعمدة النصية
    df = pd.get_dummies(df, columns=['room_type', 'neighbourhood'], drop_first=True)
    
    return df

def train_random_forest(df):
    """Train a RandomForest model and return model + metrics"""
    X = df.drop('price', axis=1)
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    return model, X_test
