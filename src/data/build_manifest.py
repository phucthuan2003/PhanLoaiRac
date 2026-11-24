"""Làm sạch dữ liệu và tạo manifest.csv (SPEC v2 mục 3.3). Chạy LOCAL.

Không di chuyển / xoá ảnh nào. Chỉ đọc ảnh và ghi ra:
  manifest.csv  - mỗi dòng một ảnh (schema đóng băng, xem SPEC_v2.md)
  report.txt    - thống kê làm sạch
  <cache_dir>/  - đặc trưng đã tính (md5, dHash, ảnh thu nhỏ) để chạy lại nhanh

Cách chạy (từ thư mục gốc repo):
  python -m src.data.build_manifest --root "D:/Data/waste" --out manifest.csv
  python -m src.data.build_manifest --root ... --max-seconds 150   # chạy từng đợt, chạy lại để tiếp tục

Thuật toán (giữ nguyên từ clean_manifest.py đã kiểm chứng):
  1. Duyệt 4 bộ: rhw (Recyclable and Household Waste), gd (GD v2/original), nextech, trashnet.
  2. Tính md5 + dHash 64 bit + ảnh xám 24x24 + ảnh màu 8x8.
  3. Cùng cụm (Union-Find) nếu: cùng md5; HOẶC gần trùng = tương quan 24x24 >= 0.99
     VÀ Hamming dHash <= 6 VÀ chênh màu 8x8 <= 12; HOẶC cùng ảnh gốc NexTech (_augN).
     (Chỉ dùng hash sẽ nối chuỗi ~30% số ảnh vào một cụm - KHÔNG được bỏ 2 điều kiện sau.)
  4. Cụm có nhãn thuộc 2 nhóm luật khác nhau -> exclude.
  5. Mỗi cụm giữ 1 ảnh đại diện, còn lại 'dup'.
  6. Chia split theo ảnh đại diện (không cụm nào nằm ở 2 split).
Mọi đường dẫn ghi ra là đường dẫn tương đối POSIX (dấu /).
"""
import argparse
import collections
import csv
import hashlib
import io
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DATASETS = collections.OrderedDict([
    ('rhw', 'Recyclable and Household Waste Classification/images/images'),
    ('gd', 'Garbage Dataset (GD) v2/original'),
    ('nextech', 'NexTech14_Waste_Dataset'),
    ('trashnet', 'dataset-original'),
])
EXT = ('.jpg', '.jpeg', '.png', '.webp')
NX_RE = re.compile(r'^(.*)-(\d+)(_aug\d+)?\.\w+$')
# thứ tự ưu tiên chọn ảnh đại diện trong một cụm trùng
PRIORITY = {('rhw', 'default'): 0, ('rhw', 'real_world'): 1, ('gd', ''): 2,
            ('nextech', 'original'): 3, ('trashnet', ''): 4, ('nextech', 'aug'): 5}
VN_ID = {'tai_che': 0, 'thuc_pham': 1, 'khac': 2, 'nguy_hai': 2}   # nguy hại là nhóm con của nhóm 3 (CV 9368)
COLUMNS = ['path', 'dataset', 'orig_label', 'context', 'orig_split', 'nhom', 'vn_class_id', 'is_hazardous',
           'mo_ho', 'split', 'split_raw', 'reason', 'cross_fine', 'group_id', 'group_size', 'is_rep',
           'md5', 'format', 'width', 'height']
SPLIT_ORDER = ['train', 'val', 'test_in', 'test_shift', 'test_cross_gd', 'test_cross_nextech',
               'trashnet_only', 'dup', 'exclude']


def posix_rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def list_images(root: Path, ds: str):
    base = root / DATASETS[ds]
    out = []
    if not base.is_dir():
        print(f'  [bỏ qua] không thấy {base}')
        return out
    for dp, dn, fs in os.walk(base):
        dn.sort()
        for f in sorted(fs):
            if f.startswith('.') or not f.lower().endswith(EXT):
                continue
            p = Path(dp) / f
            parts = p.relative_to(base).parts
            r = dict(path=posix_rel(p, root), dataset=ds, orig_label='', context='', orig_split='', nx_base='')
            if ds == 'rhw':
                r['orig_label'], r['context'] = parts[0], parts[1]
            elif ds == 'nextech':
                r['orig_split'] = parts[0]
                m = NX_RE.match(f)
                r['orig_label'] = m.group(1) if m else 'unknown'
                r['nx_base'] = f'{m.group(1)}-{m.group(2)}' if m else f
                r['context'] = 'aug' if (m and m.group(3)) else 'original'
            else:
                r['orig_label'] = parts[0]
            out.append(r)
    return out


def features(path: Path):
    b = path.read_bytes()
    md5 = hashlib.md5(b).hexdigest()
    im = Image.open(io.BytesIO(b))
    w, h, fmt = im.size[0], im.size[1], im.format or ''
    im.draft('RGB', (160, 160))          # giải mã JPEG nhanh ở độ phân giải thấp
    g = im.convert('L')
    d = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    dh = np.packbits((d[:, 1:] > d[:, :-1]).ravel()).view('>u8')[0]
    th = np.asarray(g.resize((24, 24), Image.BILINEAR), dtype=np.uint8).ravel()
    col = np.asarray(im.convert('RGB').resize((8, 8), Image.BILINEAR), dtype=np.uint8).ravel()
    return md5, fmt, w, h, int(dh), th, col


def load_cache(cache_dir: Path, ds: str):
    f = cache_dir / f'{ds}.npz'
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=False)
    if 'col' not in z.files:
        return {}
    return {str(p).replace('\\', '/'): (m, fm, int(w), int(h), int(dh), th, c) for p, m, fm, w, h, dh, th, c in
            zip(z['path'], z['md5'], z['fmt'], z['w'], z['h'], z['dh'], z['th'], z['col'])}


def save_cache(cache_dir: Path, ds: str, cache):
    cache_dir.mkdir(parents=True, exist_ok=True)
    ks = list(cache)
    tmp = cache_dir / f'{ds}.tmp.npz'
    np.savez(tmp, path=np.array(ks), md5=np.array([cache[k][0] for k in ks]),
             fmt=np.array([cache[k][1] for k in ks]), w=np.array([cache[k][2] for k in ks]),
             h=np.array([cache[k][3] for k in ks]),
             dh=np.array([cache[k][4] for k in ks], dtype=np.uint64),
             th=np.stack([cache[k][5] for k in ks]) if ks else np.zeros((0, 576), np.uint8),
             col=np.stack([cache[k][6] for k in ks]) if ks else np.zeros((0, 192), np.uint8))
    os.replace(tmp, cache_dir / f'{ds}.npz')


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def read_mapping(path: Path):
    mp = {}
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            mp[(r['dataset'], r['orig_label'])] = (r['nhom'], r['mo_ho'])
    return mp


def build(args):
    t0 = time.time()
    root = Path(args.root).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else root / '_cache'

    # ---- 1. danh sách ảnh + đặc trưng (có cache) ----
    rows = []
    for ds in DATASETS:
        items = list_images(root, ds)
        cache = load_cache(cache_dir, ds)
        todo = [r for r in items if r['path'] not in cache]
        print(f'{ds}: {len(items)} ảnh, đã có cache {len(items) - len(todo)}, cần tính {len(todo)}', flush=True)
        for i, r in enumerate(todo):
            try:
                cache[r['path']] = features(root / r['path'])
            except Exception as e:  # ảnh hỏng: đánh dấu, không dừng chương trình
                cache[r['path']] = ('ERROR', str(e)[:40], 0, 0, 0, np.zeros(576, np.uint8), np.zeros(192, np.uint8))
            if (i + 1) % 2000 == 0:
                save_cache(cache_dir, ds, cache)
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                save_cache(cache_dir, ds, cache)
                print(f'Hết thời gian ({len(todo) - i - 1} ảnh {ds} chưa tính). Chạy lại để tiếp tục.')
                sys.exit(2)
        if todo:
            save_cache(cache_dir, ds, cache)
        for r in items:
            m, fm, w, h, dh, th, c = cache[r['path']]
            r.update(md5=str(m), format=str(fm), width=int(w), height=int(h), _dh=dh, _th=th, _col=c)
        rows += items
    n = len(rows)
    if n == 0:
        sys.exit(f'Không tìm thấy ảnh nào dưới {root}')
    print(f'Tổng {n} ảnh. Tính đặc trưng xong sau {time.time() - t0:.0f}s', flush=True)

    # ---- 2. ánh xạ nhãn (báo lỗi ngay nếu có nhãn chưa khai báo) ----
    mp = read_mapping(Path(args.mapping))
    unmapped = sorted({(r['dataset'], r['orig_label']) for r in rows} - set(mp))
    if unmapped:
        sys.exit(f'LỖI: nhãn chưa có trong {args.mapping}: {unmapped}')
    for r in rows:
        r['nhom'], r['mo_ho'] = mp[(r['dataset'], r['orig_label'])]

    # ---- 3. gộp cụm trùng ----
    uf = UnionFind(n)
    reason_edge = collections.Counter()
    by_md5 = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r['md5'] != 'ERROR':
            by_md5[r['md5']].append(i)
    for v in by_md5.values():
        for j in v[1:]:
            uf.union(v[0], j); reason_edge['md5'] += 1
    by_nx = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r['dataset'] == 'nextech':
            by_nx[r['nx_base']].append(i)
    for v in by_nx.values():
        for j in v[1:]:
            uf.union(v[0], j); reason_edge['nextech_aug'] += 1

    T = np.stack([r['_th'] for r in rows]).astype(np.float32)
    T -= T.mean(1, keepdims=True)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-6)
    DH = np.array([r['_dh'] for r in rows], dtype=np.uint64)
    COL = np.stack([r['_col'] for r in rows]).astype(np.float32)

    def same_image(p, q):   # kiểm tra chặt: phải giống cả cấu trúc (dHash) lẫn màu
        if bin(int(DH[p]) ^ int(DH[q])).count('1') > args.max_ham:
            return False
        return float(np.abs(COL[p] - COL[q]).mean()) <= args.max_color

    near_pairs = 0
    for b in range(4):   # dHash chia 4 dải 16 bit làm khoá ứng viên
        key = (DH >> np.uint64(16 * b)) & np.uint64(0xFFFF)
        order = np.argsort(key, kind='stable')
        ks = key[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1
        for grp in np.split(order, cuts):
            if len(grp) < 2:
                continue
            G = T[grp]
            for s in range(0, len(grp), 2048):
                S = G[s:s + 2048] @ G.T
                ii, jj = np.nonzero(S >= args.sim)
                for x, y in zip(ii, jj):
                    x += s
                    if x < y:
                        p, q = grp[x], grp[y]
                        if uf.find(p) != uf.find(q) and same_image(p, q):
                            uf.union(p, q); near_pairs += 1
    reason_edge['gan_trung'] = near_pairs
    print(f'Gộp cụm xong sau {time.time() - t0:.0f}s: {dict(reason_edge)}', flush=True)

    groups = collections.defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    gid = {}
    for k, (root_i, mem) in enumerate(sorted(groups.items())):
        for i in mem:
            gid[i] = k

    # ---- 4-5. xung đột nhãn, chọn đại diện ----
    rng = random.Random(args.seed)
    for r in rows:
        r['split'] = ''; r['reason'] = ''; r['is_rep'] = 0
    conflict_pairs = collections.Counter()
    rep_rows = []
    for root_i, mem in groups.items():
        k = gid[mem[0]]
        for i in mem:
            rows[i]['group_id'] = k; rows[i]['group_size'] = len(mem)
        errs = [i for i in mem if rows[i]['md5'] == 'ERROR']
        for i in errs:
            rows[i]['split'], rows[i]['reason'] = 'exclude', 'loi_doc_anh'
        mem = [i for i in mem if i not in errs]
        if not mem:
            continue
        nh = {VN_ID[rows[i]['nhom']] for i in mem} if args.conflict_level == 'vn' else {rows[i]['nhom'] for i in mem}
        if len(nh) > 1:
            labs = tuple(sorted({f"{rows[i]['dataset']}:{rows[i]['orig_label']}" for i in mem}))
            conflict_pairs[labs] += 1
            names = sorted({rows[i]['nhom'] for i in mem})
            for i in mem:
                rows[i]['split'], rows[i]['reason'] = 'exclude', 'xung_dot_nhom:' + '|'.join(names)
            continue
        mem.sort(key=lambda i: (PRIORITY.get((rows[i]['dataset'], rows[i]['context']), 9), rows[i]['path']))
        rep = mem[0]
        rows[rep]['is_rep'] = 1
        for i in mem[1:]:
            rows[i]['split'], rows[i]['reason'] = 'dup', 'trung_voi:' + rows[rep]['path']
        rep_rows.append(rep)

    # ---- 6. chia split theo ảnh đại diện ----
    rhw_default = collections.defaultdict(list)
    for i in rep_rows:
        r = rows[i]
        key = (r['dataset'], r['context'])
        if key == ('rhw', 'default'):
            rhw_default[r['orig_label']].append(i)
        elif key == ('rhw', 'real_world'):
            r['split'] = 'test_shift'
        elif r['dataset'] == 'gd':
            r['split'] = 'test_cross_gd'
        elif key == ('nextech', 'original'):
            r['split'] = 'test_cross_nextech'
        elif key == ('nextech', 'aug'):
            r['split'], r['reason'] = 'exclude', 'nextech_chi_co_ban_aug'
        elif r['dataset'] == 'trashnet':
            r['split'] = 'trashnet_only'
    for lab, idx in sorted(rhw_default.items()):   # phân tầng theo nhãn con, 70/15/15
        idx.sort(key=lambda i: rows[i]['path'])
        rng.shuffle(idx)
        n_tr, n_va = round(len(idx) * 0.70), round(len(idx) * 0.15)
        for j, i in enumerate(idx):
            rows[i]['split'] = 'train' if j < n_tr else ('val' if j < n_tr + n_va else 'test_in')

    # ---- cột bổ sung ----
    for mem in groups.values():
        labs = {rows[i]['orig_label'] for i in mem}
        nhs = {rows[i]['nhom'] for i in mem}
        for i in mem:
            rows[i]['cross_fine'] = int(len(labs) > 1 and len(nhs) == 1)
    for r in rows:
        r['vn_class_id'] = VN_ID.get(r['nhom'], -1)
        r['is_hazardous'] = int(r['nhom'] == 'nguy_hai')
        r['split_raw'] = ''
    # split_raw: chia NGẪU NHIÊN bỏ qua cụm trùng, trên TOÀN BỘ ảnh rhw/default -> chỉ cho leakage ablation
    rng_raw = random.Random(args.seed + 1)
    raw_by_lab = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r['dataset'] == 'rhw' and r['context'] == 'default' and r['md5'] != 'ERROR':
            raw_by_lab[r['orig_label']].append(i)
    for lab, idx in sorted(raw_by_lab.items()):
        idx.sort(key=lambda i: rows[i]['path'])
        rng_raw.shuffle(idx)
        n_tr, n_va = round(len(idx) * 0.70), round(len(idx) * 0.15)
        for j, i in enumerate(idx):
            rows[i]['split_raw'] = 'train' if j < n_tr else ('val' if j < n_tr + n_va else 'test_in')

    # ---- ghi manifest ----
    out = Path(args.out)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- báo cáo ----
    L = [f'Manifest tạo lúc {time.strftime("%Y-%m-%d %H:%M")}, gần trùng: corr>={args.sim}, hamming<={args.max_ham}, '
         f'màu<={args.max_color}; seed={args.seed}',
         f'Tổng ảnh: {n}; số cụm trùng: {len(groups)}; cạnh gộp: {dict(reason_edge)}', '',
         '== Số ảnh theo bộ x split ==']
    ct = collections.Counter((r['dataset'], r['split']) for r in rows)
    L.append('dataset'.ljust(10) + ''.join(s[:14].rjust(15) for s in SPLIT_ORDER))
    for ds in DATASETS:
        L.append(ds.ljust(10) + ''.join(str(ct[(ds, s)]).rjust(15) for s in SPLIT_ORDER))
    L += ['', '== Số ảnh dùng được theo nhóm x split ==']
    use = SPLIT_ORDER[:7]
    cn = collections.Counter((r['nhom'], r['split']) for r in rows)
    L.append('nhom'.ljust(12) + ''.join(s[:14].rjust(15) for s in use))
    for nh in ['tai_che', 'thuc_pham', 'khac', 'nguy_hai']:
        if any(cn[(nh, s)] for s in use):
            L.append(nh.ljust(12) + ''.join(str(cn[(nh, s)]).rjust(15) for s in use))
    L += ['', '== Lý do loại (exclude) ==']
    for k, v in collections.Counter(r['reason'] for r in rows if r['split'] == 'exclude').most_common():
        L.append(f'  {v:6d}  {k}')
    L += ['', '== Cụm trùng có nhãn mâu thuẫn (top 20) ==']
    for k, v in conflict_pairs.most_common(20):
        L.append(f'  {v:5d}  {" vs ".join(k)}')
    bad = sum(1 for mem in groups.values()
              if len({rows[i]['split'] for i in mem} - {'dup', 'exclude'}) > 1)
    L += ['', f'Kiểm tra rò rỉ: số cụm xuất hiện ở >1 split = {bad} (phải bằng 0)',
          f'Số ảnh cross_fine dùng được: {sum(1 for r in rows if r["cross_fine"] and r["split"] not in ("dup", "exclude"))}']
    txt = '\n'.join(L)
    Path(args.report).write_text(txt + '\n', encoding='utf-8')
    print(txt)
    print(f'Xong sau {time.time() - t0:.0f}s -> {out}')
    if bad:
        sys.exit('LỖI: phát hiện rò rỉ giữa các split')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='thư mục chứa 4 bộ dữ liệu gốc')
    ap.add_argument('--mapping', default='configs/label_mapping.csv')
    ap.add_argument('--out', default='manifest.csv')
    ap.add_argument('--report', default='report.txt')
    ap.add_argument('--cache-dir', default=None, help='mặc định <root>/_cache')
    ap.add_argument('--max-seconds', type=float, default=0, help='giới hạn thời gian tính đặc trưng (0 = không giới hạn)')
    ap.add_argument('--sim', type=float, default=0.99, help='ngưỡng tương quan ảnh xám 24x24')
    ap.add_argument('--max-ham', type=int, default=6, help='Hamming dHash tối đa')
    ap.add_argument('--max-color', type=float, default=12.0, help='chênh lệch màu trung bình (8x8 RGB) tối đa')
    ap.add_argument('--conflict-level', choices=['nhom', 'vn'], default='nhom',
                    help="'nhom': nguy_hai khác khac là xung đột (mặc định, giống bản đã kiểm chứng); 'vn': chỉ xét 3 nhóm luật")
    ap.add_argument('--seed', type=int, default=42)
    build(ap.parse_args(argv))


if __name__ == '__main__':
    main()
