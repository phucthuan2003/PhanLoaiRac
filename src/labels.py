"""Nhãn: 3 nhóm luật VN, 30 nhãn con của bộ rhw và ma trận gộp M (30 x 3). Không phụ thuộc torch."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAPPING_CSV = REPO / 'configs' / 'label_mapping.csv'

VN_CLASSES = ['tai_che', 'thuc_pham', 'khac']
VN_NAMES = {0: 'Tái chế', 1: 'Thực phẩm', 2: 'Rác sinh hoạt khác'}
VN_ID = {'tai_che': 0, 'thuc_pham': 1, 'khac': 2, 'nguy_hai': 2}
HAZARD_FINE = {'aerosol_cans'}            # nhãn con nguy hại có trong tập train (pin KHÔNG có)
FINE_DATASET = 'rhw'


def read_mapping(path: Path | str = MAPPING_CSV) -> dict:
    with open(path, encoding='utf-8') as fh:
        return {(r['dataset'], r['orig_label']): r for r in csv.DictReader(fh)}


def fine_labels(path: Path | str = MAPPING_CSV) -> list[str]:
    """30 nhãn con của bộ rhw, sắp theo chữ cái -> fine_id = vị trí trong danh sách."""
    return sorted(lab for (ds, lab) in read_mapping(path) if ds == FINE_DATASET)


def mapping_matrix(labels: list[str] | None = None, path: Path | str = MAPPING_CSV) -> np.ndarray:
    """M[i, k] = 1 nếu nhãn con i thuộc nhóm luật k. Mỗi hàng có đúng một số 1."""
    mp = read_mapping(path)
    labels = labels or fine_labels(path)
    M = np.zeros((len(labels), 3), dtype=np.float32)
    for i, lab in enumerate(labels):
        M[i, VN_ID[mp[(FINE_DATASET, lab)]['nhom']]] = 1.0
    return M


VN_DISPLAY_FINE = {
    'aerosol_cans': 'Bình xịt', 'aluminum_food_cans': 'Lon nhôm đựng thực phẩm', 'aluminum_soda_cans': 'Lon nước ngọt',
    'cardboard_boxes': 'Thùng carton', 'cardboard_packaging': 'Bao bì carton', 'clothing': 'Quần áo',
    'coffee_grounds': 'Bã cà phê', 'disposable_plastic_cutlery': 'Dao nĩa nhựa dùng 1 lần', 'eggshells': 'Vỏ trứng',
    'food_waste': 'Thức ăn thừa', 'glass_beverage_bottles': 'Chai thuỷ tinh', 'glass_cosmetic_containers': 'Lọ mỹ phẩm thuỷ tinh',
    'glass_food_jars': 'Hũ thuỷ tinh', 'magazines': 'Tạp chí', 'newspaper': 'Báo giấy', 'office_paper': 'Giấy văn phòng',
    'paper_cups': 'Cốc giấy', 'plastic_cup_lids': 'Nắp cốc nhựa', 'plastic_detergent_bottles': 'Chai nhựa đựng chất tẩy',
    'plastic_food_containers': 'Hộp nhựa đựng thức ăn', 'plastic_shopping_bags': 'Túi nilon',
    'plastic_soda_bottles': 'Chai nhựa nước ngọt', 'plastic_straws': 'Ống hút nhựa', 'plastic_trash_bags': 'Túi đựng rác',
    'plastic_water_bottles': 'Chai nước nhựa', 'shoes': 'Giày dép', 'steel_food_cans': 'Lon thép đựng thực phẩm',
    'styrofoam_cups': 'Cốc xốp', 'styrofoam_food_containers': 'Hộp xốp đựng thức ăn', 'tea_bags': 'Túi trà',
}
