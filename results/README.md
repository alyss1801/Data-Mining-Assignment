# Bố cục kết quả Online Retail

Mỗi lần chạy GMM có một thư mục riêng. Bên trong:

```text
<tên_lần_chạy>/
├── run_summary.json     # Cấu hình, dữ liệu đầu vào và chỉ số chính
├── tables/              # Các bảng CSV
├── figures/             # Các biểu đồ PNG
└── model/               # Mô hình GMM đã lưu (.joblib)
```

| Thư mục | Ý nghĩa |
| --- | --- |
| `gmm_regularized/` | GMM mặc định trên `data/rfm.csv`, 4.338 khách, chặn phân vị 99 |
| `gmm_group_features/` | GMM trên `customer_features.csv` của PR, 4.320 khách, chặn phân vị 99 |
| `gmm_shared_matrix/` | GMM trên cùng 4.320 khách và đúng ma trận không chặn ngoại lệ của notebook K-Means |
| `public_model_comparison/` | So sánh K-Means/GMM trên ma trận chung; CSV ở `tables/`, JSON tổng quan ở thư mục gốc |
| `gmm/` | Lần GMM thử cũ, chỉ giữ ở local; không đưa vào phần nộp |

Trong mỗi lần chạy GMM, bảng kết quả từng khách là
`tables/customer_segments.csv`; chân dung nhóm là
`tables/segment_profiles.csv`; các hình nằm trong `figures/`.
Thư mục này chỉ chứa kết quả của Online Retail công khai, không có dữ liệu
Bibica.
