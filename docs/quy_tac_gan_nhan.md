# Quy tắc chụp và gán nhãn tập VN-Real (tầng 4)

Viết **trước khi chụp**, cả 5 người dùng chung. Mục tiêu: 150–200 ảnh, mỗi người 35–40 ảnh.

## 1. Nhóm nhãn

Theo Điều 75 Luật BVMT 2020 và hướng dẫn kỹ thuật kèm CV 9368/BTNMT-KSONMT:

| `vn_class_id` | Nhóm | Ví dụ |
|---|---|---|
| 0 | Tái chế | Giấy, bìa carton, báo; chai/hộp nhựa **sạch**; **túi nilon sạch**; lon nhôm, lon sắt; chai lọ thủy tinh; quần áo, giày dép cũ; đồ điện tử hỏng |
| 1 | Thực phẩm | Thức ăn thừa, vỏ trái cây, rau, vỏ trứng, bã cà phê, túi trà, thực phẩm hết hạn |
| 2 | Rác sinh hoạt khác | Hộp/cốc xốp; ống hút, dao nĩa nhựa dùng 1 lần; cốc giấy tráng nhựa; túi đựng rác; **đồ đã dính bẩn không làm sạch được**; khẩu trang, tã; lá cây; **pin, bình xịt** (nguy hại) |

## 2. Quy tắc cho vùng xám (bắt buộc làm theo)

1. **Dính thức ăn / dầu mỡ rõ ràng mà không tráng được** (hộp nhựa đựng cơm còn thức ăn, túi nilon dính nước canh) → **2**. Nếu chỉ dính ít, tráng được → 0, ghi `is_dirty = 1`.
2. **Ly trà sữa nhựa**: còn trà/trân châu → 2; đã rửa sạch → 0. Nắp nhựa đi theo ly.
3. **Hộp sữa giấy (Tetra Pak)** → 0 (bao bì giấy nhiều lớp, thu gom tái chế riêng).
4. **Cốc giấy** (cà phê, trà) → 2. **Hộp xốp** → 2 trong mọi trường hợp.
5. **Pin, bình xịt** → 2. Mô hình chưa học ảnh pin, nên ảnh pin chỉ dùng để báo cáo hạn chế.
6. Vật ghép nhiều chất liệu không tách được → 2.

## 3. Cách chụp

- Mỗi ảnh **một vật**, chiếm khoảng 30–80% khung hình. Chụp bằng điện thoại, giữ nguyên tỷ lệ 4:3 và **không cắt ảnh**.
- `bg_type`: `simple` (nền trơn: bàn, sàn) hoặc `cluttered` (nền lộn xộn: bếp, bãi rác). Cố gắng có khoảng 50/50.
- `is_vn_specific = 1` cho đồ đặc thù VN: hộp xốp cơm tấm, ly trà sữa, hộp sữa giấy VN, túi nilon chợ, vỏ bánh tráng…
- Đặt tên file không dấu, không khoảng trắng, ví dụ `sv2/ly_tra_sua_01.jpg`.

## 4. Gán nhãn hai lần

- Người chụp gán `vn_class_id`. Một người khác gán độc lập vào cột `annotator2_vn_class_id`, **không xem nhãn của người kia**.
- Ảnh có hai nhãn khác nhau → cả nhóm thảo luận theo mục 2 rồi chốt. Báo cáo tỷ lệ đồng ý và Cohen's kappa **trước khi** chốt.
- Lưu thành `vn_real/vn_real_manifest.csv` theo mẫu `docs/vn_real_manifest_template.csv`. Cột `path` là đường dẫn POSIX tính từ thư mục `vn_real/`.
- **Không xem dự đoán của mô hình khi gán nhãn.**
