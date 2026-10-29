# Đánh giá lại GMM trên Online Retail — 20/10/2026

## Phạm vi

Đây là GMM độc lập trong `src/gmm_modeling.py`, không phải GMM đối chứng nằm
trong notebook K-Means của PR #1. Không dùng dữ liệu Bibica. Đã chạy lại trên
hai bảng RFM được tạo từ cùng bộ Online Retail công khai:

| Đầu vào | Khách | Định nghĩa Monetary | Thư mục kết quả |
| --- | ---: | --- | --- |
| `data/rfm.csv` của nhánh `main` | 4.338 | Tổng mua dương sau ETL cơ bản | `results/gmm_regularized/` |
| `data/customer_features.csv` của PR #1, commit `1b16eb0` | 4.320 | Bảng đã loại dòng trùng, lọc mã sản phẩm và trừ giá trị hủy theo `src/features.py` của PR | `results/gmm_group_features/` |

Bảng thứ hai được đọc từ worktree riêng `D:\Data Mining\KMeans_PR1_Review`;
PR hiện chưa merge vào `main`. Hai lần chạy dùng cùng mã GMM, chặn trên từng
biến tại phân vị 99, `log1p`, `StandardScaler`, `reg_covar=0.1`, lưới 2–10
thành phần × 4 dạng covariance. Chỉ chọn ứng viên hội tụ và không có cụm dưới
`max(20 khách, 1% số khách)`; BIC thấp nhất trong các ứng viên đó quyết định cấu
trúc. Mô hình cuối được fit lại với 30 khởi tạo.

## Kết quả và kiểm chứng

| Chỉ số | RFM `main` | Bảng đặc trưng PR #1 |
| --- | ---: | ---: |
| Số ứng viên hợp lệ / 36 | 33 | 30 |
| Mô hình được chọn | 7 thành phần, `full` | 7 thành phần, `full` |
| BIC của mô hình cuối | 29.711,29 | 29.575,02 |
| Silhouette trên mẫu cố định 2.000 khách | 0,3037 | 0,3028 |
| Davies–Bouldin | 0,9875 | 0,9796 |
| Cụm nhỏ nhất / lớn nhất | 247 / 1.000 | 256 / 1.009 |
| Trung vị xác suất gán nhóm lớn nhất | 0,6313 | 0,6328 |
| Tỷ lệ khách có xác suất lớn nhất dưới 0,7 | 62,1% | 61,6% |

`src/verify_gmm.py` **PASS** trên cả hai bộ: đủ ID, nhãn và xác suất khớp mô
hình đã lưu; mỗi hàng xác suất cộng thành 1; BIC/AIC và chân dung cụm khớp.
Biểu đồ chân dung của lần chạy 4.320 khách đã được mở kiểm tra trực quan.

Trên 4.320 khách chung, nhãn từ hai phiên bản RFM có ARI **0,9379** và
**96,85%** giữ cùng số nhóm sau khi sắp nhóm theo giá trị RFM. Điều này cho thấy
kết quả khá nhất quán trước bước tinh chỉnh dữ liệu của PR, nhưng không xác
nhận định nghĩa Monetary nào là chuẩn nghiệp vụ.

Kiểm tra độ nhạy trên bảng 4.320 khách: `reg_covar` 0,03, 0,05 và 0,10 đều
chọn 7 thành phần `full`; tại 0,20 chuyển thành 2 thành phần. Với cấu trúc 7
thành phần tại 0,10, 10 cặp so sánh từ 5 seed có ARI tối thiểu **0,9200**,
trung vị **0,9415**. Số cụm vì vậy vẫn phụ thuộc mức điều chuẩn.

## Đối chiếu với K-Means của PR

**Cập nhật:** so sánh trên ma trận đầu vào hoàn toàn giống notebook đã được
chạy và ghi tại [KMEANS_GMM_COMPARISON_2026-10-20.md](KMEANS_GMM_COMPARISON_2026-10-20.md).
Các con số ngay dưới đây là lần đánh giá trước đó khi GMM còn chặn ngoại lệ;
không dùng chúng làm bảng so sánh cuối.

Trên cùng 4.320 ID khách, K-Means của notebook có 4 cụm; GMM độc lập có 7.
ARI giữa hai cách gán nhãn là **0,4119**. Khi *đánh giá nhãn* trên cùng không
gian `log1p` + `StandardScaler` không chặn ngoại lệ, K-Means có silhouette
**0,3388** và Davies–Bouldin **1,0124**; GMM có **0,3002** và **0,9786**.
Đây là thước đo hai cách phân nhóm với số cụm khác nhau, không phải bằng chứng
thuật toán nào tốt hơn về mọi mặt. GMM hiện được fit với chặn phân vị 99, còn
K-Means trong notebook được fit không chặn; cần thống nhất biến đổi đầu vào nếu
muốn lập bảng so sánh mô hình cuối. BIC chỉ dùng trong lựa chọn GMM và không so
trực tiếp BIC giữa hai bảng RFM hoặc với inertia của K-Means.

## Kết luận về mức hoàn thiện

**Phần GMM độc lập đã hoàn chỉnh về mặt chạy và kiểm chứng trên Online Retail**:
có xử lý đầu vào, chọn mô hình, kiểm tra độ nhạy/seed, nhãn mềm, bảng chân dung,
biểu đồ, mô hình lưu và lệnh xác minh. Trên bảng đặc trưng của PR, kết quả 7
nhóm có thể dùng làm **phân tích GMM độc lập**. Các chiến lược marketing là giả
thuyết, chưa được kiểm chứng bằng kết quả chiến dịch; xác suất hậu nghiệm không
phải độ chính xác dự báo.

Để dùng như **kết quả chung cuối của nhóm**, sau khi PR #1 được merge cần chạy
lại với đường dẫn `data/customer_features.csv` trong repo chính, xác nhận SHA
của đầu vào, rồi cập nhật báo cáo theo đúng cohort và biến đổi mà nhóm thống
nhất. Hướng dẫn GMM hiện tại vẫn mô tả lần chạy 4.338 khách của `main`;
không trộn số liệu của hai lần chạy trong cùng một bảng kết quả.

Lệnh chạy lại trên bảng đặc trưng của nhóm sau khi bảng này có trong repo chính:

```powershell
python src/gmm_modeling.py --input data/customer_features.csv --output-dir results/gmm_group_features
python src/gmm_robustness.py --input data/customer_features.csv --output-dir results/gmm_group_features
python src/verify_gmm.py --input data/customer_features.csv --output-dir results/gmm_group_features
```
