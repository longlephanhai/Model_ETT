import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score

def create_ts_data_pro(data, window_size=24, target_size=1, target_col="OT"):
    df = data.copy()
    
    df['hour'] = df['date'].dt.hour
    
    all_cols = [col for col in df.columns if col != "date"]
    new_cols = []


    for i in range(1, window_size + 1):
        for col in all_cols:
            shifted = df[col].shift(i)
            shifted.name = f"{col}_lag_{i}"
            new_cols.append(shifted)

    for i in range(target_size):
        target_shifted = df[target_col].shift(-i-1)
        target_shifted.name = f"target_{i+1}"
        new_cols.append(target_shifted)

    df = pd.concat([df] + new_cols, axis=1).dropna()
    return df


data = pd.read_csv("./ETT-small/ETTh1.csv") 
data["date"] = pd.to_datetime(data["date"])

window_size = 24 
target_size = 1 

df_final = create_ts_data_pro(data, window_size, target_size)


targets = [f"target_{i+1}" for i in range(target_size)]
X = df_final.drop(["date"] + targets, axis=1)
y = df_final[targets[0]] 


split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# Pipeline 1: Linear Regression + PCA 
pipe_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)), # Giữ lại 95% phương sai
    ('lr', LinearRegression())
])

# Pipeline 2: Ridge Regression (
pipe_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

# Pipeline 3: Random Forest 
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
])

# Huấn luyện và Đánh giá
models = {"Linear+PCA": pipe_linear, "Ridge": pipe_ridge, "Random Forest": pipe_rf}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred)
    }

# --- KẾT QUẢ ---
print("Bảng so sánh mô hình:")
print(pd.DataFrame(results).T)

# Trực quan hóa so sánh (Chương 1)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[-100:], label="Thực tế", color='black', alpha=0.5)
for name, model in models.items():
    plt.plot(model.predict(X_test)[-100:], label=f"Dự báo {name}", linestyle='--')

plt.title("So sánh các mô hình dự báo OT (100 giờ cuối)")
plt.legend()
plt.savefig("output.png")