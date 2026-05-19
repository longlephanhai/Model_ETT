# BÁO CÁO KỸ THUẬT

## Dự Báo Nhiệt Độ Dầu Máy Biến Áp (OT) Trên Bộ Dữ Liệu ETTh1 Bằng Machine Learning

Sinh viên: ...............................................................  
Mã sinh viên: ...............................................................  
Lớp: ...............................................................  
Giảng viên hướng dẫn: ...............................................................  
Thời gian thực hiện: ...............................................................

---

## MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Kiến trúc tổng thể của hệ thống](#2-kiến-trúc-tổng-thể-của-hệ-thống)
3. [Dữ liệu và tiền xử lý](#3-dữ-liệu-và-tiền-xử-lý)
4. [Kỹ thuật học máy áp dụng](#4-kỹ-thuật-học-máy-áp-dụng)
5. [So sánh, đánh giá](#5-so-sánh-đánh-giá)
6. [Kết luận](#6-kết-luận)
7. [Hướng phát triển](#7-hướng-phát-triển)

---

## 1. Giới thiệu

Đề tài tập trung vào bài toán dự báo chuỗi thời gian cho biến mục tiêu `OT` (Oil Temperature) trong bộ dữ liệu ETTh1. Đây là bài toán thực tế trong giám sát vận hành hệ thống điện, nơi dự báo ngắn hạn giúp tăng khả năng cảnh báo sớm và tối ưu khai thác thiết bị.

Mục tiêu chính của đề tài:

- Khảo sát và phân tích dữ liệu ETTh1 theo hướng chuỗi thời gian.
- Xây dựng pipeline tiền xử lý và tạo đặc trưng trễ (lag features).
- Huấn luyện mô hình dự báo bằng các thuật toán hồi quy máy học.
- Đánh giá kết quả bằng các chỉ số MAE, MSE, R2 và so sánh giữa các mô hình.

---

## 2. Kiến trúc tổng thể của hệ thống

Hệ thống trong đề tài gồm 3 phần chính:

1. Phân tích dữ liệu và trực quan hóa:
- Thực hiện trong notebook `ETTh1_visualization.ipynb`.
- Bao gồm: phân phối dữ liệu, tương quan, decomposition, ACF/PACF, rolling statistics và báo cáo chất lượng dữ liệu.

2. Huấn luyện mô hình:
- Thực hiện bằng các script `train.py` và `train2.py`.
- Pipeline tạo dữ liệu supervised từ chuỗi thời gian và huấn luyện mô hình.
- Mô hình sau huấn luyện được lưu bằng `joblib` để tái sử dụng.

3. Triển khai dự báo:
- Backend FastAPI trong `main.py` cung cấp API `POST /api/predict`.
- Frontend React/TypeScript trong thư mục `ETDataset-frontend` phục vụ nhập dữ liệu và hiển thị kết quả dự báo.

---

## 3. Dữ liệu và tiền xử lý

### 3.1. Các tập dữ liệu

Tập dữ liệu sử dụng chính trong báo cáo là `ETT-small/ETTh1.csv`.

Các cột dữ liệu:

- `date`
- `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`
- `OT` (biến mục tiêu dự báo)

Thông tin thống kê chính:

| Thuộc tính | Giá trị |
| --- | ---: |
| Số bản ghi | 17,420 |
| Số cột | 8 |
| Khoảng thời gian | 2016-07-01 00:00:00 đến 2018-06-26 19:00:00 |
| Số giá trị thiếu | 0 |
| Số dòng trùng lặp | 0 |

Thống kê nhanh của `OT`:

| Chỉ số | Giá trị |
| --- | ---: |
| Mean | 13.3247 |
| Std | 8.5669 |
| Min | -4.0800 |
| Max | 46.0070 |

### 3.2. Tiền xử lý dữ liệu

Các bước tiền xử lý đã áp dụng:

1. Chuyển cột thời gian `date` sang kiểu datetime.
2. Sắp xếp dữ liệu theo thời gian và giữ nguyên thứ tự khi chia train/test.
3. Tạo dữ liệu supervised bằng cửa sổ lịch sử `window_size = 24`.
4. Tạo đặc trưng trễ cho tất cả biến đầu vào theo dạng `feature_lag_i`.
5. Tạo nhãn dự báo tương lai theo `target_1`, `target_2`, `target_3` (tùy script).
6. Chuẩn hóa đặc trưng bằng `StandardScaler` trước khi huấn luyện.

Nhận xét dữ liệu:

- Dữ liệu sạch, không có missing value và duplicate.
- OT có tính chu kỳ theo ngày, thể hiện rõ qua decomposition và phân tích theo giờ.
- OT có tự tương quan tốt, phù hợp với mô hình dựa trên lag features.

---

## 4. Kỹ thuật học máy áp dụng

Các mô hình đã thử nghiệm trong đề tài:

- Linear Regression + StandardScaler
- Ridge Regression + StandardScaler
- Linear Regression + PCA
- Random Forest Regressor

Chiến lược chia tập:

- Train: 80% dữ liệu đầu theo thời gian
- Test: 20% dữ liệu cuối theo thời gian

Chỉ số đánh giá:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- R2 Score

---

## 5. So sánh, đánh giá

### 5.1. Kết quả multi-step từ `train.py` (Linear Regression)

| Bước dự báo | MAE | MSE | R2 |
| --- | ---: | ---: | ---: |
| T+1 | 0.4618 | 0.4251 | 0.9642 |
| T+2 | 0.6355 | 0.7983 | 0.9328 |
| T+3 | 0.7761 | 1.1508 | 0.9030 |

Đánh giá:

- Dự báo bước gần (T+1) đạt độ chính xác rất tốt.
- Sai số tăng dần theo chân trời dự báo là phù hợp với đặc tính tích lũy lỗi của bài toán multi-step.

### 5.2. Kết quả so sánh mô hình từ `train2.py` (one-step)

| Mô hình | MAE | MSE | R2 |
| --- | ---: | ---: | ---: |
| Linear + PCA | 1.0811 | 2.1293 | 0.8206 |
| Ridge | 0.4531 | 0.4184 | 0.9647 |
| Random Forest | 0.4726 | 0.4575 | 0.9614 |

Nhận xét:

- Ridge cho kết quả tốt nhất trong nhóm mô hình one-step theo cả MAE và R2.
- Random Forest bám sát Ridge nhưng thời gian huấn luyện lớn hơn.
- Linear + PCA giảm đáng kể chất lượng trên tập test, cho thấy mất mát thông tin khi giảm chiều với cấu hình hiện tại.

Kết luận đánh giá:

- Với dữ liệu ETTh1 và tập đặc trưng hiện tại, mô hình tuyến tính có regularization (Ridge) hoặc Linear baseline cho hiệu quả cao và ổn định.

---

## 6. Kết luận

Các nội dung đã thực hiện:

- Xây dựng quy trình phân tích dữ liệu ETTh1 đầy đủ (EDA + kiểm tra chất lượng dữ liệu).
- Thiết kế pipeline tạo đặc trưng trễ từ chuỗi thời gian.
- Huấn luyện và so sánh nhiều mô hình hồi quy.
- Tích hợp mô hình vào API để phục vụ dự báo và kết nối giao diện frontend.

Ưu điểm:

- Pipeline rõ ràng, tái lập được.
- Kết quả dự báo tốt ở ngắn hạn.
- Dễ triển khai thực tế qua FastAPI + frontend.

Hạn chế:

- Chưa khai thác các mô hình deep learning theo chuỗi.
- Chưa tối ưu siêu tham số toàn diện.
- Chưa đánh giá sâu theo từng mùa hoặc từng miền thời gian đặc thù.

---

## 7. Hướng phát triển

- Thử nghiệm thêm XGBoost/LightGBM và các mô hình LSTM, Transformer cho time series.
- Mở rộng đánh giá với RMSE, MAPE và phân tích sai số theo mùa.
- Tăng cường đặc trưng thời gian (day-of-week, month, sin/cos cyclical features).
- Triển khai quy trình retrain định kỳ và theo dõi drift dữ liệu.

---

## Tài liệu tham khảo

1. ETT Dataset (Electricity Transformer Temperature).
2. Tài liệu `pandas`, `scikit-learn`, `statsmodels`, `FastAPI`.
3. Các notebook và script trong dự án ETDataset.
