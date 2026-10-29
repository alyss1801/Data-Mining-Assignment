# GMM cho phân khúc khách hàng RFM — hướng dẫn đọc và tái lập

Tài liệu này giải thích **phần GMM độc lập của bài Data Mining**. Dữ liệu đầu
vào là bộ Online Retail công khai trong `data/rfm.csv`. Phần thử riêng trên
Bibica không nằm trong repo này và không được dùng để tạo các con số dưới đây.
GMM chưa được ghép vào notebook K-Means của bạn cùng nhóm.

## 1. Câu hỏi nghiên cứu và dữ liệu

Mục tiêu: từ hành vi mua hàng, chia khách thành các nhóm có đặc điểm tương
đối giống nhau để **mô tả** và đề xuất hướng chăm sóc/marketing cần thử nghiệm.
Không có nhãn “khách VIP thật” để huấn luyện hay đo accuracy.

`src/etl_rfm.py` tạo một dòng cho mỗi `CustomerID`:

| Biến | Định nghĩa trong dữ liệu hiện tại | Chiều được xem là tốt hơn |
| --- | --- | --- |
| `Recency` (R) | Số ngày từ lần mua cuối đến ngày tham chiếu (ngày giao dịch cuối + 1) | Thấp hơn |
| `Frequency` (F) | Số hóa đơn mua khác nhau | Cao hơn |
| `Monetary` (M) | Tổng `Quantity × UnitPrice` của các dòng mua hợp lệ | Cao hơn |

Đầu vào hiện có **4.338 khách**. ETL bỏ hóa đơn hủy có mã bắt đầu bằng `C`,
dòng số lượng/giá không dương và dòng thiếu mã khách. Vì thế M đang là **tổng
giá trị mua dương**, chưa trừ các khoản hủy/hoàn sau mua, chưa xử lý mọi dòng
trùng hoặc mã hàng không phải sản phẩm. Không gọi M này là doanh thu ròng.

## 2. GMM hoạt động thế nào?

GMM (*Gaussian Mixture Model*) giả định dữ liệu sau biến đổi là hỗn hợp của
`k` phân phối chuẩn nhiều chiều. Mỗi thành phần có trọng số, tâm và ma trận
hiệp phương sai. Thuật toán EM luân phiên ước lượng:

1. **E-step:** xác suất tương đối để mỗi khách thuộc từng thành phần.
2. **M-step:** cập nhật trọng số, tâm và hiệp phương sai từ các xác suất đó.

Kết quả “mềm”: một khách có thể thuộc S2 với xác suất 0,55 và S3 với 0,40.
Để lập bảng, code lấy thành phần có xác suất lớn nhất làm nhãn cứng. Cột
`AssignmentConfidence` là **xác suất hậu nghiệm lớn nhất theo mô hình**,
không phải xác suất nhãn marketing đúng. `ProbabilityMargin` là khoảng cách
giữa hai xác suất lớn nhất; nhỏ nghĩa là khách nằm gần ranh giới. Bảy cột
`Probability_S1` đến `Probability_S7` cộng thành 1 trên từng khách.

## 3. Từ dữ liệu đến mô hình: từng bước trong code

1. **Kiểm tra đầu vào:** `CustomerID` không trùng, R/F/M đều có giá trị hữu
   hạn và dương. Không âm thầm điền giá trị thiếu.
2. **Chặn ngoại lệ trên:** từng biến được chặn tại phân vị 99; trong lần chạy
   này R = 369 ngày, F = 30 hóa đơn, M ≈ 19.881. Giá trị gốc vẫn được giữ
   để báo cáo chân dung nhóm.
3. **Giảm lệch và chuẩn hóa:** áp dụng `log1p`, rồi `StandardScaler`. GMM học
   trên ma trận đã biến đổi có 4.338 dòng × 3 cột, không học trực tiếp trên
   các đơn vị ngày/hóa đơn/tiền.
4. **Tìm cấu trúc:** thử `k=2…10` và bốn loại covariance `full`, `tied`,
   `diag`, `spherical`: **36 ứng viên**. Mỗi ứng viên thử 10 khởi tạo. Chỉ xét
   ứng viên hội tụ và mọi cụm có ít nhất `max(20, 1% số khách)` khi gán nhãn.
5. **Chọn ứng viên:** trong số hợp lệ, lấy BIC nhỏ nhất tại **cùng dữ liệu,
   cùng cách biến đổi và cùng `reg_covar`**. BIC cân đối mức khớp và số tham
   số: `BIC = -2 × log-likelihood + p × log(n)`. BIC không chứng minh có đúng
   `k` kiểu khách hàng ngoài đời.
6. **Fit cuối:** dùng 30 khởi tạo, tính lại BIC/AIC và chỉ số cụm từ **mô
   hình cuối**, lưu mô hình cùng scaler, ngưỡng chặn và ánh xạ tên cụm.

`reg_covar=0.1` là lượng cộng vào đường chéo covariance để hạn chế phương sai
tiến về 0. Đây là lựa chọn mô hình cần kiểm tra độ nhạy, không phải hằng số
đúng cho mọi tập dữ liệu.

## 4. Tại sao bản hiện tại dùng 7 cụm?

`Frequency` là biến đếm nguyên; **1.493 khách chỉ có F=1**. Ở bản thử trước
với `reg_covar=1e-6`, một phương sai của F co đúng xuống `1e-6`. BIC chọn 9
thành phần `diag`, nhưng mô hình gần như tách khách theo số lần mua nguyên;
68,8% khách có xác suất hậu nghiệm lớn nhất trên 0,99. Con số đó dễ tạo cảm
giác chắc chắn quá mức.

Lần chạy mặc định với `reg_covar=0.1` chọn **7 thành phần, covariance `full`**.
`full` cho phép mỗi thành phần có ma trận covariance riêng và mô tả quan hệ
giữa R, F, M. Kiểm tra lại bằng `src/gmm_robustness.py` cho thấy:

| `reg_covar` | Ứng viên BIC tốt nhất còn hợp lệ | Cụm nhỏ nhất |
| ---: | --- | ---: |
| 0,03 | 8 thành phần, `full` | 76 |
| 0,05 | 7 thành phần, `full` | 261 |
| 0,10 | 7 thành phần, `full` | 247 |
| 0,20 | 2 thành phần, `full` | 1.195 |

Như vậy **7 cụm là một phương án phân tích hợp lý trong vùng 0,05–0,10**,
nhưng số cụm nhạy với regularization mạnh hơn. Với cấu trúc 7 cụm tại 0,1,
10 cặp so sánh từ 5 seed cho Adjusted Rand Index (ARI) tối thiểu **0,9666**,
trung vị **0,9900**. ARI gần 1 nghĩa là nhãn giữa hai lần khởi tạo gần giống
nhau sau khi bỏ qua việc đổi số thứ tự nhãn.

## 5. Kết quả mặc định cần đọc như thế nào?

| Chỉ số | Kết quả | Cách đọc |
| --- | ---: | --- |
| BIC | 29.711,29 | Dùng để chọn giữa ứng viên trên cùng ma trận đầu vào |
| AIC | 29.271,41 | Tiêu chí khác, không phải luật chọn cuối ở đây |
| Silhouette | 0,3037 | Độ tách biệt của nhãn cứng, tính trên mẫu cố định 2.000 khách |
| Davies–Bouldin | 0,9875 | Thước đo độ gọn/tách biệt; thấp hơn thường tốt hơn |
| Trung vị hậu nghiệm lớn nhất | 0,6313 | Nhiều khách nằm ở vùng chuyển tiếp; không phải “accuracy” |
| Cụm nhỏ nhất/lớn nhất | 247 / 1.000 khách | Không có cụm rỗng hoặc quá nhỏ theo quy tắc đã đặt |

Silhouette không phải mục tiêu tối ưu của GMM; BIC đo chất lượng mô hình mật
độ còn silhouette đo độ tách của nhãn cứng. Không so sánh trực tiếp BIC giữa
hai tập dữ liệu khác nhau.

| Nhóm | Khách | % khách | R trung vị | F trung vị | M trung vị | % tổng M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 — Giá trị vượt trội | 247 | 5,7% | 5 | 18 | 7.454 | 45,9% |
| S2 — Mua lặp lại gần đây | 385 | 8,9% | 5 | 7 | 2.256 | 12,8% |
| S3 — Giá trị cao, chậm lại | 667 | 15,4% | 35 | 6 | 2.219 | 21,5% |
| S4 — Mới phát triển | 545 | 12,6% | 15 | 3 | 693 | 4,6% |
| S5 — Mua thỉnh thoảng | 836 | 19,3% | 90 | 2 | 788 | 9,9% |
| S6 — Mua một lần, còn gần | 658 | 15,2% | 39,5 | 1 | 256 | 2,1% |
| S7 — Mua một lần, đã lâu | 1.000 | 23,1% | 239 | 1 | 244 | 3,2% |

M là theo đơn vị tiền của bộ dữ liệu nguồn. `% tổng M` cộng tổng Monetary
**gốc ở cấp khách**, không cộng nhầm tiền đơn ở cấp dòng. S1 có tỷ trọng M
rất lớn; đây là dấu hiệu để xem xét nhóm khách mua nhiều/chi nhiều, không
chứng minh mọi khách S1 đều mang lợi nhuận cao. Các chiến lược trong
`segment_strategies.csv` chỉ là **giả thuyết marketing cần thử nghiệm**.

## 6. Giới hạn và cách viết kết luận trung thực

- M hiện là tổng mua dương, không phải doanh thu/lợi nhuận ròng. Nếu thay
  định nghĩa RFM thì phải fit và chọn GMM lại.
- Số cụm phụ thuộc quy tắc chặn ngoại lệ, `reg_covar`, khoảng `k` và tiêu chí
  kích thước tối thiểu. BIC thấp nhất trong lưới không bảo đảm “đúng” về kinh
  doanh.
- GMM giả định hỗn hợp Gaussian trên dữ liệu đã biến đổi; F là biến đếm rời
  rạc. Cần giải thích regularization và độ nhạy, không giấu chúng.
- Không có nhãn chuẩn hay thử nghiệm chiến dịch, nên không tính accuracy hay
  tuyên bố chiến lược sẽ tăng doanh thu.
- Notebook của bạn cùng nhóm dùng `customer_features.csv` đã tinh chỉnh,
  khoảng 4.320 khách và Monetary ròng theo một quy tắc khác. Khi ghép bài,
  phải chạy lại GMM trên **cùng cohort và cùng cách biến đổi** trước khi so
  K-Means/GMM. Không bê các chỉ số ở trang này sang bảng so sánh đó.
- Pilot Bibica là phân tích riêng tư ngoài repo; kết quả của nó không được
  trộn với các con số của Online Retail.

## 7. Chạy lại và tìm file nào để đọc

Từ thư mục gốc `Data-Mining-Assignment`:

```powershell
python -m pip install -r requirements.txt
python src/gmm_modeling.py
python src/gmm_robustness.py
python src/verify_gmm.py
```

Nếu cần tạo lại `data/rfm.csv` từ `Online Retail.xlsx`, chạy
`python src/etl_rfm.py` trước. `gmm_modeling.py --help` cho phép đổi đường
dẫn đầu vào/đầu ra, khoảng số cụm, ngưỡng chặn, regularization và seed.

Kết quả mặc định ở `results/gmm_regularized/`:

| File | Dùng để làm gì |
| --- | --- |
| `run_summary.json` | Cấu hình, SHA-256 đầu vào, phiên bản thư viện, mô hình được chọn và chỉ số cuối |
| `tables/model_selection.csv` + `figures/model_selection.png` | 36 ứng viên và đường BIC/AIC |
| `tables/regularization_sensitivity.csv`, `tables/seed_stability.csv` | Độ nhạy tham số và độ ổn định khởi tạo |
| `tables/customer_segments.csv` | Nhãn và xác suất của từng khách trên bộ công khai |
| `tables/segment_profiles.csv` | Quy mô, RFM trung vị, tỷ trọng giá trị từng nhóm |
| `tables/segment_strategies.csv` | Gợi ý hành động cần kiểm chứng |
| `figures/pca_segments.png`, `figures/segment_profile_heatmap.png`, `figures/assignment_confidence.png` | Hình chẩn đoán, không thay cho kiểm định |
| `model/gmm_model.joblib` | Mô hình, scaler, ngưỡng chặn và ánh xạ nhãn để tái sử dụng |

`results/gmm/` là **lần thử cũ giữ ở local** với `reg_covar=1e-6`; không đưa
vào phần nộp và không dùng làm kết quả chính. Chỉ nạp file joblib từ nguồn đáng
tin cậy và môi trường Python tương thích.

## 8. Năm câu trả lời ngắn khi thuyết trình

1. **Vì sao GMM?** GMM cho xác suất thuộc từng nhóm, phù hợp khi ranh giới
   khách hàng không sắc nét; nó là một cách phân khúc bổ sung cho K-Means.
2. **Vì sao 7 nhóm?** BIC thấp nhất trong 33 ứng viên hợp lệ trên 36 cấu hình
   đã thử ở cấu hình mặc định; 7 cũng lặp lại ở `reg_covar=0,05`. Đây là lựa chọn phân tích, không
   phải số nhóm khách hàng tuyệt đối.
3. **Vì sao không chọn bản 9 nhóm BIC âm?** Phương sai F gần 0 khiến mô hình
   quá bám vào các giá trị đếm nguyên và xác suất gần 1 quá nhiều.
4. **Xác suất 0,6 nghĩa là gì?** Thành phần được gán có hậu nghiệm 0,6 theo
   mô hình đã fit; chưa biết xác suất nhãn kinh doanh là đúng.
5. **Đã so công bằng với K-Means chưa?** Chưa: hai phần đang dùng định nghĩa
   dữ liệu khác nhau. Cần thống nhất cohort/RFM và chạy lại trước khi lập
   bảng so sánh cuối.

Tham khảo phương pháp: [ví dụ chọn GMM bằng BIC của scikit-learn](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)
và [tài liệu GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

## 9. Đoạn viết mẫu cho báo cáo

> Nhóm xây dựng một phương án phân khúc khách hàng bằng Gaussian Mixture Model
> trên ba biến Recency, Frequency và Monetary của 4.338 khách hàng. Sau khi
> chặn các giá trị cực trị ở phân vị 99, các biến được biến đổi `log1p` và
> chuẩn hóa. Nhóm thử 36 cấu hình gồm 2–10 thành phần và bốn cấu trúc
> covariance. Trong số các mô hình hội tụ và không tạo cụm dưới 1% khách,
> BIC chọn mô hình 7 thành phần với covariance `full` tại `reg_covar=0,1`.
> Mô hình cuối có silhouette 0,304 và các cụm từ 247 đến 1.000 khách.
>
> Kết quả cho thấy nhóm S1 chỉ chiếm 5,7% khách nhưng đóng góp 45,9% tổng
> Monetary theo định nghĩa mua dương của dữ liệu hiện tại. Ngược lại, S7
> gồm khách chủ yếu mua một lần từ lâu. Các xác suất hậu nghiệm cho thấy nhiều
> khách nằm giữa các nhóm: trung vị xác suất lớn nhất chỉ 0,631. Vì số thành
> phần thay đổi khi tăng regularization và Monetary chưa phải doanh thu ròng,
> nhóm xem đây là một cách mô tả hành vi để hình thành giả thuyết marketing,
> chưa phải bằng chứng về hiệu quả chiến dịch.
