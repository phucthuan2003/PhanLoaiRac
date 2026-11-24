"""Đóng gói ảnh theo manifest.csv thành 1 file zip để tải lên Google Drive. Chạy LOCAL.

  python -m src.data.make_package --root "D:/Data/waste" --manifest manifest.csv --out waste_data_v2.zip

Nội dung zip:
  data/<path>          ảnh (TrashNet: xoay theo EXIF rồi thu về cạnh dài 512 -> <tên>_512.jpg)
  manifest_colab.csv   manifest.csv + cột in_zip=1, đường dẫn đã đổi cho TrashNet
  label_mapping.csv, report.txt (nếu có)
Gồm: mọi ảnh dùng được + toàn bộ bản trùng/loại của rhw và nextech (cần cho leakage ablation).
Chạy lại được nhiều lần: file đã có trong zip được bỏ qua (dùng --max-seconds để chạy từng đợt).
"""
import argparse
import csv
import io
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--manifest', default='manifest.csv')
    ap.add_argument('--mapping', default='configs/label_mapping.csv')
    ap.add_argument('--report', default='report.txt')
    ap.add_argument('--out', default='waste_data_v2.zip')
    ap.add_argument('--max-seconds', type=float, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    root = Path(a.root)

    rows = []
    with open(a.manifest, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            usable = r['split'] not in ('dup', 'exclude')
            ablation = r['dataset'] in ('rhw', 'nextech')
            if not (usable or ablation) or r['md5'] == 'ERROR':
                continue
            if r['dataset'] == 'trashnet':
                r['_src'] = r['path']
                r['path'] = str(Path(r['path']).with_suffix('')).replace('\\', '/') + '_512.jpg'
            r['in_zip'] = 1
            rows.append(r)

    done = set()
    if Path(a.out).exists():
        with zipfile.ZipFile(a.out) as z:
            done = set(z.namelist())
    with zipfile.ZipFile(a.out, 'a', compression=zipfile.ZIP_STORED) as z:
        for i, r in enumerate(rows):
            arc = 'data/' + r['path']
            if arc in done:
                continue
            if r['dataset'] == 'trashnet':
                with Image.open(root / r['_src']) as im:
                    im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((512, 512))
                b = io.BytesIO(); im.save(b, 'JPEG', quality=95)
                z.writestr(arc, b.getvalue())
            else:
                z.write(root / r['path'], arc)
            done.add(arc)
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                print('tạm dừng, chạy lại để tiếp tục', i); sys.exit(2)
        if 'manifest_colab.csv' not in done:
            cols = [c for c in rows[0].keys() if not c.startswith('_')]
            s = io.StringIO(); w = csv.DictWriter(s, fieldnames=cols, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)
            z.writestr('manifest_colab.csv', s.getvalue())
            for extra in (a.mapping, a.report):
                if Path(extra).exists():
                    z.write(extra, Path(extra).name)
    print('xong', len(rows), 'ảnh,', round(Path(a.out).stat().st_size / 1e6), 'MB,', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
