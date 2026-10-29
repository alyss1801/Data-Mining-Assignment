# Đối chiếu K-Means và GMM trên cùng dữ liệu Online Retail

## Thiết kế so sánh

`customer_features.csv` do `src/features.py` trong PR #1 của bạn làm K-Means
tạo ra từ Online Retail. Đây là **đầu vào chung trước modeling**, gồm 4.320
khách. Cả hai thuật toán bên dưới dùng đúng ba cột RFM, cùng 4.320 ID và cùng
ma trận `X = StandardScaler(log1p(RFM))`, **không chặn ngoại lệ**. Dữ liệu được
đọc từ bản PR commit `1b16eb0` trong worktree local, không dùng Bibica.
Đối chiếu ma trận đã biến đổi giữa hai phần cho sai khác lớn nhất bằng **0**.

- **K-Means k=4:** chạy lại đúng cấu hình notebook: `n_init=20`, seed 42. Nhãn
  tái lập nhãn K-Means đã lưu trong PR với ARI = **1,000**.
- **GMM k=4:** cùng số cụm để đối chiếu mức chi tiết như nhau; covariance
  `full`, `reg_covar=0.1`, 30 lần khởi tạo, seed 42.
- **GMM do BIC chọn:** mã GMM độc lập thử 2–10 thành phần × 4 covariance,
  chọn trong các mô hình hội tụ và có mọi cụm ít nhất 1% khách, rồi fit lại với
  30 lần khởi tạo. Trên ma trận không chặn ngoại lệ này, BIC chọn **8 nhóm**.

## Kết quả trên cùng ma trận X

Các chỉ số dưới đây đều tính trên **toàn bộ 4.320 khách** sau khi fit; silhouette
trong `run_summary.json` của GMM là bản lấy mẫu 2.000 khách, nên có thể khác
chữ số cuối.

| Mô hình | Nhóm | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Nhóm nhỏ nhất | Nhóm lớn nhất |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K-Means | 4 | 0,3388 | 1,0124 | 3.330,86 | 685 (15,9%) | 1.622 (37,5%) |
| GMM cố định k=4 | 4 | 0,3376 | 0,9761 | 3.156,36 | 554 (12,8%) | 1.858 (43,0%) |
| GMM BIC chọn k=8 | 8 | 0,2894 | 1,0140 | 2.679,39 | 69 (1,6%) | 986 (22,8%) |

K-Means và GMM cùng k=4 có silhouette gần như ngang nhau. K-Means nhỉnh hơn
về Calinski–Harabasz và độ cân đối kích thước nhóm; GMM k=4 có
Davies–Bouldin thấp hơn và cho xác suất thuộc nhóm. ARI giữa nhãn hai mô hình
k=4 là **0,7229**: có phần lớn cấu trúc chung nhưng không trùng hoàn toàn.
Trung vị xác suất hậu nghiệm lớn nhất của GMM k=4 là **0,7363**; 43,8% khách
có giá trị này dưới 0,7. Đây là độ thuộc nhóm theo mô hình, không phải độ
chính xác của nhãn kinh doanh.

GMM k=8 có BIC cuối **29.526,65**, thấp hơn GMM k=4 (**29.643,90**) trên
cùng ma trận. Nhưng k=8 tách một nhóm chỉ 69 khách và silhouette giảm xuống
0,2894; 66,2% khách có xác suất nhóm lớn nhất dưới 0,7. BIC chọn mô hình mật
độ theo tiêu chí của GMM, không tự động chọn phương án phân khúc dễ triển khai.
Không so BIC với inertia của K-Means.

## Độ ổn định và GMM trong notebook

| Cấu hình | Lần chạy | ARI cặp thấp nhất | ARI cặp trung vị |
| --- | --- | ---: | ---: |
| K-Means k=4, `n_init=20` | 10 seed (0–9) | 0,9813 | 0,9913 |
| GMM k=4, `n_init=30` | 5 seed (0–4) | 0,9190 | 0,9441 |
| GMM k=8, `n_init=30` | 5 seed (0–4) | 0,6784 | 0,8599 |

GMM k=8 còn nhạy với `reg_covar`: ở 0,05 và 0,10 BIC chọn 8 `full`; ở 0,03
chọn 10 `diag`; ở 0,20 chọn 4 `full`. Kiểm tra 5 seed với 10 lần khởi tạo
cho k=8 có ARI cặp thấp nhất **0,6461**. Vì vậy **chưa nên dùng 8 nhóm làm
nhãn kinh doanh cuối** chỉ vì BIC thấp nhất.

Notebook của PR cũng có GMM k=4 nhưng chỉ dùng 5 lần khởi tạo. Chạy lại đúng
cấu hình đó tái lập nhãn đã lưu với ARI = 1,000 và BIC = **29.779,54**. Khi
tăng lên 30 lần khởi tạo, BIC cải thiện thành **29.643,90**, còn nhãn so với
bản notebook có ARI **0,6018**. Đây là dấu hiệu GMM ở 5 lần khởi tạo đã chọn
một nghiệm cục bộ khác; bảng so sánh cũ trong notebook không nên được diễn
giải là kết quả GMM duy nhất. Không chỉnh mã hoặc kết quả của bạn làm K-Means.

## Kết luận cho bài nhóm

Với mục tiêu chính là bốn nhóm hành động marketing rõ ràng, **K-Means k=4
có cơ sở tốt để giữ làm mô hình chính**: nhãn cân đối hơn và ổn định hơn trong
các lần khởi tạo đã thử. **GMM k=4 là mô hình đối chiếu trực tiếp** trên cùng
ma trận: chất lượng tách nhóm gần tương đương và bổ sung thông tin xác suất
cho khách nằm giữa các nhóm. **GMM k=8 là phân tích khám phá theo BIC**, cần
nêu rõ độ nhạy và không ép tám nhóm thành chiến lược cố định.

Bản GMM độc lập trước đó trên cùng bảng 4.320 khách nhưng có chặn phân vị 99
chọn 7 nhóm. Đó là **một cấu hình tiền xử lý khác**; số liệu của nó nằm trong
`docs/GMM_REVIEW_2026-10-20.md`, không trộn với bảng so sánh cùng ma trận ở
đây. Các khuyến nghị marketing chỉ là giả thuyết khi chưa có thử nghiệm thực tế.

## Tái lập khi `customer_features.csv` đã có trong repo chính

```powershell
python src/gmm_modeling.py --input data/customer_features.csv --output-dir results/gmm_shared_matrix --winsor-quantile 1.0
python src/gmm_robustness.py --input data/customer_features.csv --output-dir results/gmm_shared_matrix --winsor-quantile 1.0
python src/verify_gmm.py --input data/customer_features.csv --output-dir results/gmm_shared_matrix
python src/compare_public_models.py --input data/customer_features.csv --gmm-dir results/gmm_shared_matrix
```

Kết quả đối chiếu có thể đọc trong
`results/public_model_comparison/tables/` và file tổng quan
`results/public_model_comparison/comparison_summary.json`. Khi
chạy trên bản PR local hiện tại, truyền đường dẫn tuyệt đối của file trong
worktree; `run_summary.json` ghi SHA-256 đầu vào để kiểm tra đúng phiên bản.
