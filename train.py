import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report


import pandas as pd


def create_ts_data(data, window_size=5, target_size=1, target_col="OT"):
    df = data.copy()

    all_cols = [col for col in df.columns if col != "date"]
    new_columns = {}
    for i in range(1, window_size):
        for col in all_cols:
            new_columns["{}_lag_{}".format(col, i)] = df[col].shift(i)

    new_columns["target"] = df[target_col].shift(-1)

    new_cols_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    df = df.dropna()

    return df


data = pd.read_csv("./ETT-small/ETTh1.csv")
# profile = ProfileReport(data, title="ETTh1 Report", explorative=True)
# profile.to_file("ETTh1.html")

data["date"] = pd.to_datetime(data["date"])

window_size = 5
target_size = 1
data = create_ts_data(data, window_size, target_size, target_col="OT")


x = data.drop(["date", "target"], axis=1)
y = data["target"]

train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

pipe_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipe_linear_no_scaler = Pipeline([
    # ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
])


models = {
    "Linear": pipe_linear,
    "Linear (No Scaler)": pipe_linear_no_scaler,
    "Random Forest": pipe_rf
}
results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "R2": r2_score(y_test, pred)
    }

print("Bảng so sánh mô hình:")
print(pd.DataFrame(results).T)

params = {
    "rf__n_estimators": [100, 200, 300],
    "rf__criterion": ["squared_error", "friedman_mse"], # Tiêu chuẩn tính độ lỗi hồi quy
    "rf__max_depth": [5, 10, 15]
}

tscv = TimeSeriesSplit(n_splits=4)

grid_search = GridSearchCV(
    estimator=pipe_rf, 
    param_grid=params, 
    cv=tscv, 
    scoring="r2",
    verbose=2,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)


y_predicted = grid_search.predict(x_test)


print("--- KẾT QUẢ RANDOM FOREST SAU TINH CHỈNH ---")
print(f"Tham số tốt nhất: {grid_search.best_params_}")
print(f"MAE: {mean_absolute_error(y_test, y_predicted):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_predicted):.4f}")
print(f"R2 Score: {r2_score(y_test, y_predicted):.4f}")

# # Lưu mô hình
joblib.dump(pipe_linear, "linear_models.pkl")
loaded_model = joblib.load("linear_models.pkl")
