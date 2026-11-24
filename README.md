# vn-waste-classification-cv

Phân loại ảnh **một vật** vào 3 nhóm chất thải rắn sinh hoạt theo Điều 75 Luật BVMT 2020 và hướng dẫn kỹ thuật kèm Công văn 9368/BTNMT-KSONMT:

- **0 – Tái chế**
- **1 – Thực phẩm**
- **2 – Rác sinh hoạt khác** (gồm cả rác nguy hại)

Đặc tả đầy đủ nằm trong `SPEC_v2.md` (thư mục cha). Quy tắc gán nhãn tập tự chụp nằm trong `docs/quy_tac_gan_nhan.md`.

> Trên Windows, đặt repo ở đường dẫn **không dấu**, ví dụ `D:\Projects\vn-waste-classification-cv`.

## Cấu trúc

```text
configs/label_mapping.csv   bảng nhãn (nhóm luật, vùng xám, ghi chú nguồn)
src/preprocess.py           đọc ảnh (EXIF, alpha) + transform DÙNG CHUNG train/eval/demo
src/labels.py               30 nhãn con, ma trận gộp M (30x3)
src/metrics.py              bal_acc, macro-F1, contamination, ECE, risk-coverage, bootstrap
src/decision.py             luật quyết định (UNCERTAIN, nhắc tráng sạch, cảnh báo nguy hại)
src/inference.py            suy luận 1 ảnh (demo dùng)
src/data/build_manifest.py  làm sạch + manifest.csv              [SV1, local]
src/data/make_package.py    đóng gói zip lên Drive               [SV1, local]
src/data/manifest.py        chọn split / tầng đánh giá
src/data/dataset.py         PyTorch Dataset
src/models/net.py           timm backbone, mode flat | fine_grained   [SV2]
src/engine/train.py         LP-FT, AMP, weighted CE / focal, --allow_leaky  [SV2, SV3]
src/engine/benchmark.py     bảng mọi run + số tham số + độ trễ          [SV3]
src/engine/calibrate.py     T và tau trên val sạch -> temperature.json  [SV4]
src/engine/evaluate.py      đánh giá 4 tầng                             [SV4]
src/explain/gradcam.py      Grad-CAM + bộ ảnh theo 4 nhóm lỗi           [SV4]
app/demo.py                 demo Streamlit                              [SV5]
notebooks/colab_runner.ipynb  chạy toàn bộ trên Colab
tests/test_pipeline.py      unit test
```

## Quy trình

### Bước 1 – SV1, chạy local

```bash
pip install -r requirements.txt
python -m src.data.build_manifest --root "D:/Data/waste" --out manifest.csv --report report.txt
python -m src.data.make_package --root "D:/Data/waste" --manifest manifest.csv --out waste_data_v2.zip
WASTE_MANIFEST=manifest.csv WASTE_DATA_ROOT="D:/Data/waste" pytest tests -q
```

`--root` là thư mục chứa 4 bộ dữ liệu gốc. Tên thư mục con của từng bộ khai báo trong `DATASETS` ở đầu `build_manifest.py`.

Nếu đã có `manifest.csv` và `waste_data_v2.zip` từ thư mục cha (tạo ngày 24/11/2026) thì bỏ qua bước này. `build_manifest.py` cho ra manifest giống từng ô với bản đó.

Tải `waste_data_v2.zip` lên `MyDrive/PhanLoaiRac/`.

### Bước 2–4 – trên Colab

1. Đẩy repo lên GitHub, rồi sửa `REPO_URL` trong `notebooks/colab_runner.ipynb`.
2. Mở notebook trên Colab và chọn GPU T4.
3. Chạy lần lượt từ trên xuống. Run nào đã có `summary.json` trên Drive sẽ được bỏ qua.

Các lệnh notebook gọi:

```bash
python -m src.engine.train     --manifest /content/manifest_colab.csv --data_root /content/data --out_dir RUNS \
                               --backbone convnext_tiny --mode flat [--allow_leaky] [--aug basic] [--loss focal]
python -m src.engine.benchmark --runs_dir RUNS --latency
python -m src.engine.calibrate --ckpt RUNS/<run>/best_model.pth --manifest ... --data_root ...
python -m src.engine.evaluate  --ckpt RUNS/<run>/best_model.pth --manifest ... --data_root ... \
                               [--vn_real_manifest vn_real/vn_real_manifest.csv] [--eval_mode center_crop]
python -m src.explain.gradcam  --ckpt ... --predictions RUNS/<run>/eval_letterbox/predictions.csv --data_root ...
```

Thêm `--smoke` vào bất kỳ lệnh nào để chạy thử với vài ảnh. Chế độ này chỉ để bắt lỗi code, không đo chất lượng.

### Bước 5 – SV5, demo local

```bash
# chép best_model.pth + temperature.json từ MyDrive/PhanLoaiRac/final/ vào checkpoints/
streamlit run app/demo.py
```

## Chỉ số và cách đọc kết quả

- **Chỉ số chính là `bal_acc`** (balanced accuracy, tức trung bình recall các lớp). Tỷ lệ Tái chế là 65% ở `test_in` nhưng 84% ở GD, nên macro-F1 có thể giảm khoảng 0,10 dù mô hình không kém đi. Macro-F1 vẫn được báo cáo, nhưng chỉ để tham khảo.
- **`contamination_rate`**: trong các ảnh thật thuộc nhóm 1–2 và không bị từ chối, tỷ lệ bị đoán là Tái chế.
- **Leakage ablation**: so `test_in_own_split` của run `leaky` với run `clean`. `test_shift` không có bản trùng nào trong ảnh default, nên là thước đo trung thực cho cả hai run.
- **Tầng 3 GD**: lớp `trash` chứa nhiều vật bộ train không có (khẩu trang, bàn chải, tã), nên xem thêm `T3_gd_no_trash` và `per_label_recall.csv`.
- **VN-Real** chỉ có khoảng 40 ảnh mỗi ô. Luôn đọc kèm khoảng tin cậy `*_ci95` và đừng kết luận mạnh từ một ô riêng.

## Hạn chế đã biết

- Tập train **không có ảnh pin hay thiết bị điện tử**. Cảnh báo nguy hại chỉ hoạt động với bình xịt, và chỉ ở mode `fine_grained`. Demo có ghi rõ hạn chế này.
- Phép khử trùng không bắt được ảnh bị crop hoặc lật. Lần quét độc lập còn thấy 3 cặp gần trùng giữa train và các tập khác.
- Bảng nhãn dựa trên bản tóm tắt CV 9368. Cần đối chiếu nguyên văn trước khi nộp báo cáo, nhất là túi nilon, ống hút và lá cây.

## Khác biệt nhỏ so với SPEC v2

- `RandomLetterbox` lúc train thêm viền **rộng hơn** cạnh ảnh 0–35%. Ảnh bộ 15k đều vuông 256×256, nên nếu chỉ đệm cho vuông như `LetterboxSquare` thì bước này không có tác dụng gì.
- Thiết bị tự chọn là CUDA, không có thì CPU. GPU Apple (MPS) phải bật bằng `--device mps`, vì torch 2.15 bị crash MPS trên macOS 14.
- Bảng nhãn giữ dạng CSV (`configs/label_mapping.csv`), không chuyển sang YAML.

## Môi trường đã kiểm thử (local, CPU)

Python 3.11, torch 2.15.0, torchvision 0.29.2, timm 1.0.31, pandas 3.1.1, numpy 2.4.6, streamlit 1.67.0.
