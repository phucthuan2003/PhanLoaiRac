# FACTS FOR REPORT — HỒ SƠ SỰ THẬT CỐ ĐỊNH (SINGLE SOURCE OF TRUTH)

> **CHỈ THỊ BẮT BUỘC CHO MỌI AI / THÀNH VIÊN KHI VIẾT BÁO CÁO HOẶC SỬA CODE:**
> 1. **Workspace duy nhất là `D:\Code\PhanLoaiRac`** (GitHub: `https://github.com/phucthuan2003/PhanLoaiRac`). Không sử dụng tài liệu/code nháp SPEC v1 cũ.
> 2. **Cấm tuyệt đối tự bịa số liệu.** Mọi con số ngoài mục 2 bên dưới PHẢI đọc trực tiếp từ các file CSV/JSON thật trong `results_drive/`. Nếu file chưa có -> để placeholder `[CHỜ FILE TỪ DRIVE]`.
> 3. **Bám sát đúng cấu trúc 6 mục và bảng chỉ số** trong `docs/BTL-BC.pdf`.

---

## 1. Cấu trúc mã nguồn thật (Không được viết lệch tên file / class)
- **Bảng ánh xạ nhãn:** `configs/label_mapping.csv` (dạng CSV, không phải YAML).
- **Quy tắc gán nhãn tập tự chụp:** `docs/quy_tac_gan_nhan.md` và `docs/vn_real_manifest_template.csv`.
- **Module lõi:**
  - `src/preprocess.py`: đọc ảnh (EXIF, RGBA->nền xám, hỗ trợ `pillow-heif` đọc `.HEIC`), `RandomLetterbox` lúc train, Letterbox vuông `256×256` lúc eval/demo, thêm `downscale_for_display()` (chống nặng ảnh 12–48MP) và `assess_image_quality()` (cảnh báo thiếu sáng/lóa/mờ).
  - `src/models/net.py`: class `WasteNet` (`self.backbone` + `self.head`), hỗ trợ 3 backbone `mobilenetv3_large_100`, `convnext_tiny`, `swinv2_tiny_window8_256` và 2 chế độ `flat` | `fine_grained`.
  - `src/decision.py`: hàm `decide()` — luật `UNCERTAIN` 2 lớp (`tau_recycle = 0.75` chống ô nhiễm tái chế + `min_conf = 0.60` trên Demo chặn OOD/mơ hồ chung), nhắc tráng sạch & tách vật liệu ghép, cảnh báo rác nguy hại.
  - `src/metrics.py`: `bal_acc`, `macro_f1`, `contamination_rate`, `ECE`, `risk-coverage`, bootstrap CI95.
  - `src/engine/`: `train.py`, `benchmark.py`, `calibrate.py`, `evaluate.py`.
  - `src/explain/gradcam.py`: class `GradCAM` + `overlay()` (tự giải phóng tensor/gradient sau mỗi ảnh chống rò RAM).
  - `app/demo.py`: ứng dụng Streamlit chạy local/HF Spaces (Upload + Webcam + Ảnh mẫu dự phòng `results_drive/demo_samples/`, thanh trượt chỉnh `tau_recycle` & `min_conf`, Grad-CAM bật theo nhu cầu để giữ CPU nhanh ~150–300ms).

---

## 2. Số liệu ĐÃ KIỂM CHỨNG 100% từ checkpoint thật trên máy
*(Đọc trực tiếp từ `checkpoints/best_model.pth` và `checkpoints/temperature.json`)*

- **Run chốt cuối:** `convnext_tiny_flat_clean_advanced_weighted_ce`
- **Backbone / Mode / Input:** `convnext_tiny` / **`flat`** (3 lớp trực tiếp) / `256×256`
- **Epoch tốt nhất:** `15`
- **Validation Balanced Accuracy (`val_bal_acc`):** **`0.9833`** (`98.33%`, trên `n_val = 914` ảnh sạch)
- **Hiệu chuẩn Temperature Scaling (`temperature.json`):**
  - Nhiệt độ tối ưu `T = 1.5646`
  - `ECE`: giảm từ **`0.0135`** -> **`0.0067`**
  - `NLL`: giảm từ **`0.0803`** -> **`0.0682`**
  - Ngưỡng chống ô nhiễm tái chế `tau_recycle = 0.75` (giữ mặc định vì contamination trên val đã rất thấp `0.0125 <= 0.05`)
  - Tại `tau = 0.75` trên Val: `coverage = 98.80%`, `contamination = 0.63%`
  - Tại `tau = 0.90` trên Val (tham khảo từ sweep): `coverage = 97.05%`, `contamination = 0.00%`, `selective_bal_acc = 99.08%`

---

## 3. CÁC BẪY CẤM BỊA (ANTI-HALLUCINATION RULES)
1. **Mode cuối là `flat`, KHÔNG phải `fine_grained`:**
   - Trong `src/decision.py`, `top_fine` (Top-3 nhãn con) và cảnh báo rác nguy hại (`aerosol_cans`) **chỉ kích hoạt ở mode `fine_grained`**.
   - **CẤM viết** trong báo cáo rằng *"Demo của model cuối hiển thị Top-3 vật liệu con và tự cảnh báo bình xịt/pin"*. Hãy viết trung thực: hệ thống có thiết kế hỗ trợ `fine_grained`, nhưng thực nghiệm chọn `flat` vì đạt `val_bal_acc` cao hơn; do đó trên bản demo cuối, cơ chế chống ô nhiễm (`UNCERTAIN` + nhắc tráng sạch) hoạt động đầy đủ, còn cảnh báo pin/nguy hại là hạn chế đã biết (có ghi rõ trên giao diện Demo).
2. **Chỉ số chính là `bal_acc` (Balanced Accuracy), KHÔNG phải `Macro-F1`:**
   - Lý do kỹ thuật (theo `README.md`): tỷ lệ lớp Tái chế thay đổi mạnh giữa các tập (65% ở `test_in` vs 84% ở `GD v2`) làm Macro-F1 biến động ~0.10 dù mô hình không kém đi. `Macro-F1`, `Precision`, `Recall` vẫn báo cáo đầy đủ theo yêu cầu `BTL-BC.pdf`, nhưng dùng `bal_acc` làm tiêu chí chọn mô hình.
3. **KHÔNG train / test trên `TrashNet` và `TACO`:**
   - `configs/label_mapping.csv` có liệt kê `trashnet` và `taco` để tham chiếu quy tắc gán nhãn, **KHÔNG** có nghĩa là mô hình được huấn luyện hay đánh giá trên 2 bộ này (TrashNet trùng 90% GD v2; TACO không có ảnh sẵn).
4. **Trạng thái Tầng 4 (`VN-Real`):**
   - **CHỈ được viết số liệu `VN-Real`** nếu trong `results_drive/eval_letterbox/summary.csv` thực sự có dòng tầng `VN-Real`. Nếu chưa có -> ghi rõ ở mục Hạn chế & Hướng phát triển, **cấm tự bịa số liệu VN-Real**.
5. **Số liệu kiểm toán dữ liệu gốc (đã xác minh trước huấn luyện):**
   - Bộ 15k (`Recyclable and Household Waste Classification`): 15.000 file -> 11.263 ảnh duy nhất; 2.966 ảnh `real_world` trùng hệt `default` (~4.300 ảnh `real_world` độc lập); 280 nhóm ảnh giống hệt khác nhãn (gồm các cặp xuyên luật như `plastic_food_containers` vs `styrofoam_food_containers`).
   - `Garbage Dataset (GD) v2`: 12.259 ảnh (3 bản resize, chỉ dùng `standardized_384`); 10 lớp lệch mạnh (`clothes` 1.892 vs `trash` 453); trùng 7 ảnh với bộ 15k; lớp `trash` lẫn ảnh bãi rác nhiều vật.
   - `TrashNet`: 2.527 ảnh, 2.271/2.527 (~90%) nằm trong GD v2 -> loại bỏ.
   - `NexTech14_Waste_Dataset`: 8.996 ảnh nhưng chỉ có 1.230 ảnh gốc + ~7.766 ảnh `_aug`; rò rỉ nặng (1.296/1.352 test và 1.292/1.348 val trùng gốc với train); xếp sai `trash` vào `recyclable`; trùng ít nhất 149 ảnh với bộ 15k.

---

## 4. Nguồn file số liệu trong `results_drive/` (Bắt buộc đối chiếu khi viết Mục 5)
| Nội dung Mục 5 (`BTL-BC.pdf`) | File nguồn bắt buộc đọc trong `results_drive/` |
| :--- | :--- |
| Bảng Leakage Ablation (clean vs leaky) & flat vs fine_grained | `ablation/convnext_tiny_flat_clean_summary.json`, `ablation/convnext_tiny_flat_leaky_summary.json`, `ablation/convnext_tiny_fine_grained_clean_summary.json` |
| Bảng so sánh 3 backbone + aug + số tham số + độ trễ | `runs_table.csv` |
| Bảng đánh giá 4 tầng (Letterbox) | `eval_letterbox/summary.csv` & `eval_letterbox/results.json` |
| So sánh Letterbox vs Center-Crop | `eval_center_crop/summary.csv` |
| Phân tích Recall từng nhãn con (GD v2 / `trash`) | `eval_letterbox/per_label_recall.csv` |
| Hình Ma trận nhầm lẫn & Risk-Coverage | `eval_letterbox/confusion_matrices.png`, `eval_letterbox/risk_coverage.png` |
| Hình phân tích trường hợp làm tốt / dự đoán sai | `gradcam/*.png` |
| Ảnh minh họa Demo chạy thực tế | `demo_screenshots/*.png` (tự chụp từ `http://localhost:8501`) |

---

## 5. Thông tin hành chính nhóm BTL (Điền trước khi xuất báo cáo cuối)
- **Trường:** Đại học Giao thông Vận tải (UTC)
- **Học phần:** Thị giác máy tính (Computer Vision)
- **Đề tài:** Phân loại 3 lớp chất thải rắn sinh hoạt theo Luật Bảo vệ môi trường 2020
- **Giảng viên hướng dẫn:** `[ĐIỀN TÊN GVHD]`
- **Danh sách & phân công 5 thành viên:**
  1. `[Họ tên SV1 — MSSV]`: Dữ liệu, kiểm toán trùng lặp, `build_manifest.py`, đóng gói `waste_data_v2.zip` (Mục 1, 3).
  2. `[Họ tên SV2 — MSSV]`: Kiến trúc `WasteNet`, thực nghiệm Leakage Ablation & `flat` vs `fine_grained` (Mục 2, 4, 5).
  3. `[Họ tên SV3 — MSSV]`: So sánh 3 backbone, augmentation `basic` vs `advanced`, benchmark độ trễ, chốt `best_model.pth` (Mục 4, 5).
  4. `[Họ tên SV4 — MSSV]`: Hiệu chuẩn `temperature.json`, đánh giá đa tầng, Letterbox vs Center-Crop, phân tích lỗi Grad-CAM (Mục 5).
  5. `[Họ tên SV5 — MSSV]`: Xây dựng & kiểm thử Web Demo Streamlit `app/demo.py`, tổng hợp báo cáo & slide (Mục 4, 6, Phụ lục).
