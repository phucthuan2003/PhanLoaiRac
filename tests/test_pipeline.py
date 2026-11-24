"""Unit test (SPEC v2 bước 1 + điều kiện Done bước 5).

  pytest tests/test_pipeline.py -q
Biến môi trường (tuỳ chọn):
  WASTE_MANIFEST   đường dẫn manifest.csv / manifest_colab.csv (mặc định: ../manifest.csv nếu có)
  WASTE_DATA_ROOT  thư mục gốc ảnh tương ứng (mặc định: ..)
  WASTE_READ_ALL=1 đọc thử 100% ảnh dùng được (chậm, ~vài phút)
Test cần torch/timm sẽ tự bỏ qua nếu chưa cài.
"""
from __future__ import annotations

import io
import os
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.decision import MSG_CLEAN, MSG_HAZARD, decide
from src.labels import VN_ID, fine_labels, mapping_matrix, read_mapping
from src.metrics import accept_mask, bal_acc, contamination_rate, ece, macro_f1, risk_coverage
from src.preprocess import LetterboxSquare, RandomLetterbox, load_clean_rgb

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(os.environ.get('WASTE_MANIFEST', REPO.parent / 'manifest.csv'))
DATA_ROOT = Path(os.environ.get('WASTE_DATA_ROOT', REPO.parent))
have_manifest = pytest.mark.skipif(not MANIFEST.exists(), reason='không có manifest')


@pytest.fixture(scope='module')
def df():
    from src.data.manifest import load_manifest
    return load_manifest(MANIFEST)


# ---------------- nhãn ----------------
def test_mapping_matrix_one_hot():
    labs = fine_labels()
    M = mapping_matrix(labs)
    assert len(labs) == 30 and M.shape == (30, 3)
    assert (M.sum(1) == 1).all()
    assert labs == sorted(labs)


def test_mapping_follows_spec_v2():
    mp = read_mapping()
    g = lambda ds, lab: VN_ID[mp[(ds, lab)]['nhom']]
    assert g('rhw', 'plastic_shopping_bags') == 0        # nhựa mềm -> tái chế (CV 9368)
    assert g('rhw', 'aerosol_cans') == 2 and mp[('rhw', 'aerosol_cans')]['nhom'] == 'nguy_hai'
    assert g('nextech', 'e-waste') == 0 and g('nextech', 'battery') == 2 and g('gd', 'battery') == 2
    assert g('nextech', 'garden-waste') == 2
    counts = np.bincount([g('rhw', lab) for lab in fine_labels()], minlength=3)
    assert counts.tolist() == [19, 4, 7]


# ---------------- manifest: rò rỉ, đường dẫn ----------------
@have_manifest
def test_no_group_in_two_splits(df):
    use = df[~df['split'].isin(['dup', 'exclude'])]
    assert (use.groupby('group_id')['split'].nunique() == 1).all()
    assert (use.groupby('group_id').size() == 1).all()        # mỗi cụm chỉ 1 ảnh đại diện được dùng


@have_manifest
def test_paths_posix_relative(df):
    assert not df['path'].str.contains('\\\\', regex=False).any()
    assert not df['path'].str.match(r'^([A-Za-z]:|/)').any()


@have_manifest
def test_split_sizes_and_labels(df):
    s = df['split'].value_counts()
    tr, va, te = s['train'], s['val'], s['test_in']
    assert abs(tr / (tr + va + te) - 0.70) < 0.01
    assert set(df['vn_class_id']) <= {0, 1, 2}
    assert (df.loc[df['is_hazardous'] == 1, 'vn_class_id'] == 2).all()
    for sp in ('train', 'val', 'test_in'):
        sub = df[df['split'] == sp]
        assert ((sub['dataset'] == 'rhw') & (sub['context'] == 'default')).all()


@have_manifest
def test_test_shift_has_no_dup_in_default(df):
    """test_shift là thước đo trung thực cho cả run leaky: không cụm nào chứa ảnh rhw/default."""
    shift_groups = set(df.loc[df['split'] == 'test_shift', 'group_id'])
    default_groups = set(df.loc[(df['dataset'] == 'rhw') & (df['context'] == 'default'), 'group_id'])
    assert not shift_groups & default_groups


@have_manifest
def test_select_rules(df):
    from src.data.manifest import select
    fine_tr = select(df, 'train', 'fine_grained')
    assert (fine_tr['cross_fine'] == 0).all() and (fine_tr['fine_id'] >= 0).all()
    leaky = select(df, 'train', 'flat', allow_leaky=True)
    clean = select(df, 'train', 'flat')
    assert len(leaky) > len(clean)
    raw = df[df['split_raw'] != '']
    assert ((raw['dataset'] == 'rhw') & (raw['context'] == 'default')).all()
    # leaky: phải có cụm trùng nằm ở cả train lẫn test_in (đúng mục đích ablation)
    lt = set(leaky['group_id']) & set(select(df, 'test_in', 'flat', allow_leaky=True)['group_id'])
    assert len(lt) > 0


@have_manifest
def test_read_images_sample(df):
    use = df[~df['split'].isin(['dup', 'exclude'])]
    if os.environ.get('WASTE_READ_ALL') != '1':
        use = use.groupby('dataset', group_keys=False).apply(lambda d: d.sample(min(len(d), 40), random_state=0))
    missing = [p for p in use['path'] if not (DATA_ROOT / p).exists()]
    if len(missing) == len(use):
        pytest.skip('không thấy ảnh dưới WASTE_DATA_ROOT')
    assert not missing, missing[:5]
    for p in use['path']:
        im = load_clean_rgb(DATA_ROOT / p)
        assert im.mode == 'RGB' and min(im.size) > 0


# ---------------- tiền xử lý ----------------
def _jpeg_with_orientation(w, h, orientation):
    im = Image.new('RGB', (w, h), (200, 0, 0))
    im.paste((0, 0, 200), (0, 0, w // 4, h // 4))           # đánh dấu góc trên-trái
    ex = Image.Exif(); ex[274] = orientation
    b = io.BytesIO(); im.save(b, 'JPEG', exif=ex.tobytes()); return b.getvalue()


def test_exif_rotation():
    im = load_clean_rgb(_jpeg_with_orientation(400, 300, 6))   # 6 = xoay 90° theo chiều kim đồng hồ
    assert im.size == (300, 400)


def test_rgba_composited_on_gray():
    im = Image.new('RGBA', (10, 10), (255, 0, 0, 0))
    b = io.BytesIO(); im.save(b, 'PNG')
    out = load_clean_rgb(b.getvalue())
    assert out.mode == 'RGB' and out.getpixel((5, 5)) == (128, 128, 128)


def test_palette_transparency():
    im = Image.new('P', (8, 8), 0); im.putpalette([255, 0, 0] + [0] * 765); im.info['transparency'] = 0
    b = io.BytesIO(); im.save(b, 'PNG', transparency=0)
    assert load_clean_rgb(b.getvalue()).getpixel((0, 0)) == (128, 128, 128)


def test_letterbox():
    out = LetterboxSquare()(Image.new('RGB', (400, 300), (255, 255, 255)))
    assert out.size == (400, 400)
    assert out.getpixel((200, 10)) == (128, 128, 128) and out.getpixel((200, 200)) == (255, 255, 255)
    random.seed(0)
    sq = Image.new('RGB', (256, 256), (255, 255, 255))
    outs = [RandomLetterbox(p=1.0)(sq).size[0] for _ in range(20)]
    assert all(s >= 256 for s in outs) and max(outs) > 256     # ảnh vuông vẫn được thêm viền


def test_transforms_shape():
    pytest.importorskip('torchvision')
    from src.preprocess import get_transforms
    img = Image.new('RGB', (640, 480), (10, 200, 30))
    for tf in (get_transforms(True), get_transforms(True, aug='basic'), get_transforms(False),
               get_transforms(False, eval_mode='center_crop')):
        assert tuple(tf(img).shape) == (3, 256, 256)


# ---------------- chỉ số + luật quyết định ----------------
def test_metrics_match_sklearn():
    sk = pytest.importorskip('sklearn.metrics')
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500); p = np.where(rng.random(500) < .7, y, rng.integers(0, 3, 500))
    assert abs(bal_acc(y, p) - sk.balanced_accuracy_score(y, p)) < 1e-12
    assert abs(macro_f1(y, p) - sk.f1_score(y, p, average='macro')) < 1e-12


def test_bal_acc_ignores_prior():
    y1 = np.array([0] * 50 + [1] * 50); y2 = np.array([0] * 90 + [1] * 10)
    def pred(y):  # recall 0.8 cho mỗi lớp
        p = y.copy()
        for c in (0, 1):
            idx = np.flatnonzero(y == c); p[idx[: len(idx) // 5]] = 1 - c
        return p
    assert abs(bal_acc(y1, pred(y1)) - bal_acc(y2, pred(y2))) < 1e-9
    assert macro_f1(y1, pred(y1)) - macro_f1(y2, pred(y2)) > 0.05


def test_abstention_and_contamination():
    P = np.array([[0.6, 0.2, 0.2], [0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.5, 0.1, 0.4]])
    y = np.array([2, 0, 1, 1])
    acc = accept_mask(P, 0.75)
    assert acc.tolist() == [False, True, True, False]
    assert contamination_rate(y, P.argmax(1)) == pytest.approx(2 / 3)
    assert contamination_rate(y, P.argmax(1), acc) == 0.0
    rc = risk_coverage(P, y)
    assert rc[0]['coverage'] >= rc[-1]['coverage']
    assert 0 <= ece(P, y) <= 1


def test_decide_rules():
    labs = fine_labels()
    r = decide(np.array([0.6, 0.2, 0.2]), 0.75)
    assert r['status'] == 'UNCERTAIN'
    r = decide(np.array([0.9, 0.05, 0.05]), 0.75)
    assert r['status'] == 'OK' and MSG_CLEAN in r['messages']
    fp = np.zeros(30); fp[labs.index('aerosol_cans')] = 0.9; fp[0] += 0.1
    r = decide(np.array([0.1, 0.0, 0.9]), 0.75, fp, labs)
    assert r['hazard'] and r['messages'][0] == MSG_HAZARD and r['top_fine'][0]['label'] == 'aerosol_cans'


# ---------------- mô hình (cần torch + timm) ----------------
@pytest.mark.parametrize('mode', ['flat', 'fine_grained'])
def test_model_shapes_and_probs(mode):
    torch = pytest.importorskip('torch'); pytest.importorskip('timm')
    from src.models.net import WasteNet
    m = WasteNet('mobilenetv3_large_100', mode, pretrained=False).eval()
    with torch.no_grad():
        lg = m(torch.randn(2, 3, 256, 256))
        P = m.vn_probs(lg, 1.5)
    assert lg.shape == (2, 3 if mode == 'flat' else 30)
    assert torch.allclose(P.sum(1), torch.ones(2), atol=1e-5)


def test_demo_matches_evaluate_pipeline(tmp_path):
    """Điều kiện Done bước 5: demo (predict_one) và evaluate (DataLoader) cho xác suất lệch < 1e-4, cùng nhãn."""
    torch = pytest.importorskip('torch'); pytest.importorskip('timm')
    import pandas as pd
    from src.engine.common import predict_logits, vn_probs_np
    from src.inference import predict_one, probs_equal
    from src.models.net import WasteNet
    torch.manual_seed(0)
    m = WasteNet('mobilenetv3_large_100', 'fine_grained', pretrained=False).eval()
    rows = []
    for i, (w, h, o) in enumerate([(400, 300, 6), (300, 300, 1), (500, 260, 3)]):
        p = tmp_path / f'{i}.jpg'; p.write_bytes(_jpeg_with_orientation(w, h, o))
        rows.append(dict(path=p.name, vn_class_id=0, fine_id=-1))
    d = pd.DataFrame(rows)
    P_eval = vn_probs_np(m, predict_logits(m, d, tmp_path, torch.device('cpu'), workers=0), 1.3)
    for i, p in enumerate(d['path']):
        P_demo, _, _ = predict_one(m, (tmp_path / p).read_bytes(), 1.3)
        assert probs_equal(P_demo, P_eval[i]) and P_demo.argmax() == P_eval[i].argmax()
