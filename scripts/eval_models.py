"""Bagian evaluasi yang butuh torch (Anggota 4): prediksi YOLOv8 dan Faster R-CNN dalam SATU format,
jumlah parameter, dan latensi (batch 1, fp32, pemanasan, GPU dicatat).

Format keluaran = format bersama scripts/eval_shared.py (list dict COCO, bbox xywh piksel absolut citra asli,
category_id = id YOLO + 1). Jalankan dari akar repo; scripts/ harus ada di sys.path.

Hal yang DIVERIFIKASI saat dijalankan (bukan diasumsikan): nama kelas model YOLO, urutan/nama berkas hasil
prediksi terhadap GT, ukuran citra asli terhadap JSON.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_shared import GT, yolo_to_coco_records

CLASSES0 = {0: "WBC", 1: "RBC", 2: "Platelets"}


def device_info() -> dict:
    cuda = torch.cuda.is_available()
    return {"gpu": torch.cuda.get_device_name(0) if cuda else "CPU", "torch": torch.__version__,
            "cuda": torch.version.cuda if cuda else None, "python": platform.python_version()}


def torch_load(path, device):
    """torch.load aman: coba weights_only=True dulu; bila gagal, muat penuh (berkas milik tim sendiri)."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def load_images(paths, bgr: bool = False) -> list[np.ndarray]:
    """Muat citra ke memori sebagai array uint8 HxWx3. bgr=True meniru cv2.imread (konvensi input array Ultralytics)."""
    out = []
    for p in paths:
        with Image.open(p) as im:
            a = np.array(im.convert("RGB"))
        out.append(np.ascontiguousarray(a[..., ::-1]) if bgr else a)
    return out


# ---------------------------------------------------------------------------------------- YOLO
def load_yolo(weights):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    names = {int(k): str(v) for k, v in m.names.items()}
    assert names == CLASSES0, f"nama kelas model tidak sesuai {CLASSES0}: {names}"
    return m


def predict_yolo(model, gt: GT, data_root, imgsz: int, conf: float = 0.001, iou: float = 0.7,
                 max_det: int = 300, chunk: int = 16, device=None) -> list[dict]:
    """Prediksi YOLO untuk semua citra GT -> format bersama. conf=0.001 (lantai rendah) agar AP penuh dan ambang
    apa pun >= 0.001 bisa diterapkan sesudahnya. iou/max_det: nilai yang sama dengan model.val() bawaan Ultralytics
    (CETAK DEFAULT_CFG di notebook untuk memverifikasi pada versi terpasang)."""
    dev = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    paths = [str(Path(data_root) / im["file_name"]) for im in gt.images]
    out: list[dict] = []
    for s in range(0, len(paths), chunk):
        part, metas = paths[s:s + chunk], gt.images[s:s + chunk]
        res = model.predict(source=part, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=dev,
                            verbose=False)       # half tidak diset: default fp32 (argumen 'half' sudah deprecated)
        assert len(res) == len(part), f"jumlah hasil {len(res)} != jumlah citra {len(part)}"
        for r, im in zip(res, metas):
            assert Path(r.path).name == Path(im["file_name"]).name, (r.path, im["file_name"])
            H, W = r.orig_shape
            assert (int(W), int(H)) == (im["width"], im["height"]), f"ukuran citra {(W, H)} != JSON {(im['width'], im['height'])}"
            b = r.boxes
            out += yolo_to_coco_records(im["id"], b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                        b.cls.cpu().numpy().astype(int))
    return out


# ---------------------------------------------------------------------------------- Faster R-CNN
def load_frcnn(run_dir, device=None):
    """Bangun model dari config.json run (min_size/max_size/anchor), muat best.pt. Memakai build_model milik
    Anggota 3 (scripts/frcnn_lib.py) agar arsitektur identik dengan saat pelatihan."""
    from frcnn_lib import DEFAULT_ANCHOR_SIZES, build_model
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    anchors = tuple(int(s[0]) for s in cfg["anchor_sizes"])
    model = build_model(min_size=int(cfg["min_size"][-1]), max_size=int(cfg["max_size"]),
                        anchor_sizes=None if anchors == tuple(DEFAULT_ANCHOR_SIZES) else anchors,
                        trainable_backbone_layers=int(cfg["args"].get("trainable_layers", 3)))
    ck = torch_load(run_dir / "best.pt", device)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, ck


def predict_frcnn(model, gt: GT, data_root, device=None, score_thresh: float = 0.001, batch_size: int = 4,
                  num_workers: int = 2) -> list[dict]:
    """Prediksi Faster R-CNN untuk semua citra GT memakai predict() milik Anggota 3 (format COCO xywh piksel asli).
    score_thresh=0.001 sama dengan evaluasi Anggota 3 (--eval-score-thresh)."""
    from frcnn_lib import CocoDetDataset, predict
    device = device or next(model.parameters()).device
    ds = CocoDetDataset(gt.ann_file, img_root=data_root, train=False)
    assert ds.image_ids() == gt.img_ids, "urutan citra dataset != GT"
    old = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = score_thresh
    try:
        res = predict(model, ds, device, batch_size=batch_size, num_workers=num_workers)
    finally:
        model.roi_heads.score_thresh = old
    return res


# --------------------------------------------------------------------------- parameter dan latensi
def count_params(module: torch.nn.Module) -> dict:
    """Jumlah parameter model yang dimuat. Untuk YOLO ini bisa sedikit berbeda dari tabel ringkasan Ultralytics
    (mis. model sudah/belum difusi Conv+BN); sebutkan di laporan bahwa angka ini 'sebagaimana dimuat'."""
    tot = sum(p.numel() for p in module.parameters())
    return {"params": int(tot), "params_m": round(tot / 1e6, 3)}


def measure_latency(fn, images: list, n: int = 100, warmup: int = 10, repeats: int = 3, min_images: int = 50) -> dict:
    """Latensi per citra, BATCH 1. fn(img) harus melakukan seluruh pipeline untuk satu citra dan memindahkan
    hasil ke CPU. Waktu per citra diukur dengan torch.cuda.synchronize() sebelum dan sesudah. Pemanasan `warmup`
    citra tidak dihitung. Dilaporkan: rerata, SD, median, p95, FPS, rerata per pengulangan, GPU, versi."""
    if n < min_images:
        raise ValueError(f"n={n} < {min_images}: ukur minimal {min_images} citra")
    if len(images) == 0:
        raise ValueError("tidak ada citra")
    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    seq = [images[i % len(images)] for i in range(n)]
    for k in range(warmup):
        fn(images[k % len(images)])
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    ms, rep_means = [], []
    for _ in range(repeats):
        cur = []
        for img in seq:
            sync()
            t0 = time.perf_counter()
            fn(img)
            sync()
            cur.append((time.perf_counter() - t0) * 1000.0)
        rep_means.append(float(np.mean(cur)))
        ms += cur
    a = np.asarray(ms)
    out = {"n_images": n, "unique_images": min(n, len(images)), "repeats": repeats, "warmup": warmup, "batch": 1,
           "precision": "fp32", "mean_ms": float(a.mean()), "std_ms": float(a.std(ddof=1)),
           "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
           "fps": float(1000.0 / a.mean()), "mean_ms_per_repeat": rep_means,
           "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else None}
    out.update(device_info())
    if not cuda:
        out["peringatan"] = "TANPA GPU: latensi CPU tidak sebanding dengan angka GPU"
    return out


def make_yolo_fn(model, imgsz: int, conf: float, iou: float = 0.7, max_det: int = 300):
    """Pipeline YOLO per citra: array BGR di memori -> (letterbox, forward, NMS, skala balik) -> kotak di CPU."""
    dev = 0 if torch.cuda.is_available() else "cpu"
    speeds: list[dict] = []

    def fn(img_bgr):
        r = model.predict(img_bgr, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=dev,
                          verbose=False)[0]      # fp32 (default)
        _ = r.boxes.xyxy.cpu().numpy()
        speeds.append(dict(r.speed))

    fn.speeds = speeds
    return fn


def make_frcnn_fn(model, score_thresh: float, device=None):
    """Pipeline Faster R-CNN per citra: array RGB di memori -> tensor -> GPU -> model (resize, backbone, RPN,
    ROI, NMS) -> kotak di CPU. Cakupan setara make_yolo_fn (tanpa baca disk)."""
    device = device or next(model.parameters()).device
    model.eval()
    model.roi_heads.score_thresh = score_thresh

    @torch.no_grad()
    def fn(img_rgb):
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float().div_(255).to(device)
        out = model([x])[0]
        _ = out["boxes"].cpu().numpy()

    return fn


def bench_yolo(model, gt: GT, data_root, imgsz: int, conf: float, n: int = 100, **kw) -> dict:
    paths = [str(Path(data_root) / im["file_name"]) for im in gt.images[:n]]
    imgs = load_images(paths, bgr=True)
    fn = make_yolo_fn(model, imgsz, conf)
    res = measure_latency(fn, imgs, n=n, **kw)
    tot = res["n_images"] * res["repeats"]
    sp = fn.speeds[-tot:]
    if sp:
        res["ultralytics_speed_ms_mean"] = {k: float(np.mean([s[k] for s in sp])) for k in sp[0]}
    res.update(count_params(model.model))
    res["conf_inferensi"] = conf
    res["imgsz"] = imgsz
    res["cakupan"] = "array BGR di memori -> kotak di CPU (preprocess+forward+NMS+skala balik); tanpa baca disk"
    return res


def bench_frcnn(model, gt: GT, data_root, score_thresh: float, n: int = 100, **kw) -> dict:
    paths = [str(Path(data_root) / im["file_name"]) for im in gt.images[:n]]
    imgs = load_images(paths, bgr=False)
    old = model.roi_heads.score_thresh
    try:
        fn = make_frcnn_fn(model, score_thresh)
        res = measure_latency(fn, imgs, n=n, **kw)
    finally:
        model.roi_heads.score_thresh = old
    res.update(count_params(model))
    res["conf_inferensi"] = score_thresh
    res["ukuran_masukan"] = [list(model.transform.min_size), model.transform.max_size]
    res["cakupan"] = "array RGB di memori -> tensor GPU -> model -> kotak di CPU; tanpa baca disk"
    return res
