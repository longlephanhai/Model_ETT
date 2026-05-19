import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import joblib


import pandas as pd


def create_ts_data(data, window_size=5, target_size=1, target_col="OT"):
    df = data.copy()

    all_cols = [col for col in df.columns if col != "date"]
    new_columns = {}
    for i in range(1, window_size):
        for col in all_cols:
            new_columns["{}_lag_{}".format(col, i)] = df[col].shift(i)
    for i in range(0, target_size):
        new_columns["target_{}".format(i + 1)] = df[target_col].shift(-i - 1)
    new_cols_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    df = df.dropna()

    return df


data = pd.read_csv("./ETT-small/ETTh1.csv")
# profile = ProfileReport(data, title="ETTh1 Report", explorative=True)
# profile.to_file("ETTh1.html")

data["date"] = pd.to_datetime(data["date"])

window_size = 24
target_size = 3
data = create_ts_data(data, window_size, target_size, target_col="OT")

targets = ["target_{}".format(i+1) for i in range(target_size)]
x = data.drop(["date"] + targets, axis=1)
y = data[targets]

train_ratio = 0.8
num_samples = len(x)

x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

pipelines = [
    Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
        # ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ]) for _ in range(target_size)
]

for i, pipe in enumerate(pipelines):
    pipe.fit(x_train, y_train["target_{}".format(i+1)])


r2 = []
mse = []
mae = []
for i, pipe in enumerate(pipelines):
    y_predict = pipe.predict(x_test)
    mae.append(mean_absolute_error(y_test["target_{}".format(i+1)], y_predict))
    mse.append(mean_squared_error(y_test["target_{}".format(i+1)], y_predict))
    r2.append(r2_score(y_test["target_{}".format(i+1)], y_predict))

print("R2: {}".format(r2))
print("MSE: {}".format(mse))
print("MAE: {}".format(mae))


test_dates = data["date"].iloc[int(
    num_samples * train_ratio):].reset_index(drop=True)

last_n = 200

plt.figure(figsize=(15, 12))

for i in range(target_size):
    plt.subplot(target_size, 1, i+1)

    y_predict_step = pipelines[i].predict(x_test)

    actual = y_test["target_{}".format(i+1)].iloc[-last_n:].values
    predicted = y_predict_step[-last_n:]
    dates = test_dates.iloc[-last_n:]

    plt.plot(dates, actual, label="Thực tế (Actual)", color='blue', alpha=0.7)
    plt.plot(dates, predicted, label="Dự báo (Predicted)",
             color='red', linestyle='--')

    plt.title(f"So sánh dự báo Nhiệt độ dầu (OT) - Bước T+{i+1}")
    plt.ylabel("Giá trị")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.xlabel("Thời gian")
plt.tight_layout()
plt.savefig("output_linear.png")


# Lưu mô hình
joblib.dump(pipelines, "linear_models.pkl")
loaded_model = joblib.load("linear_models.pkl")
