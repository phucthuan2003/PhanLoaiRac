# Thư mục chứa kết quả tải từ Google Drive (`MyDrive/PhanLoaiRac/runs/`)

Hãy tải các file từ Google Drive về đúng cấu trúc dưới đây để viết Mục 5 báo cáo (và để AI đọc trực tiếp, không tự bịa số liệu):

```text
results_drive/
├── runs_table.csv                                        # từ MyDrive/PhanLoaiRac/runs/runs_table.csv
├── ablation/
│   ├── convnext_tiny_flat_clean_summary.json             # từ runs/convnext_tiny_flat_clean_advanced_weighted_ce/summary.json
│   ├── convnext_tiny_flat_leaky_summary.json             # từ runs/convnext_tiny_flat_leaky_advanced_weighted_ce/summary.json
│   └── convnext_tiny_fine_grained_clean_summary.json     # từ runs/convnext_tiny_fine_grained_clean_advanced_weighted_ce/summary.json
├── eval_letterbox/                                       # từ runs/convnext_tiny_flat_clean_advanced_weighted_ce/eval_letterbox/
│   ├── summary.csv
│   ├── results.json
│   ├── per_label_recall.csv
│   ├── confusion_matrices.png
│   └── risk_coverage.png
├── eval_center_crop/                                     # từ runs/convnext_tiny_flat_clean_advanced_weighted_ce/eval_center_crop/
│   └── summary.csv
├── gradcam/                                              # từ runs/convnext_tiny_flat_clean_advanced_weighted_ce/gradcam/
│   └── *.png
└── demo_screenshots/                                     # ảnh chụp màn hình Demo từ http://localhost:8501
    └── *.png
```
