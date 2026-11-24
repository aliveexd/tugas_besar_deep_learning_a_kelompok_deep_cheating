"""Grad-CAM untuk detektor (Anggota 4): implementasi sendiri dengan hook PyTorch (tanpa pustaka grad-cam),
supaya setiap langkahnya bisa diperiksa. Grad-CAM = ReLU( sum_k alpha_k * A_k ), alpha_k = rerata spasial
dari d(skor)/d(A_k), A = peta fitur pada lapisan target.

Untuk detektor, Grad-CAM butuh SATU skalar target. Pilihan di sini (harus ditulis di laporan):
  * Faster R-CNN: logit kelas target dari satu proposal ROI (proposal yang kotak hasil regresinya paling
    cocok dengan kotak target dan peluang kelasnya tertinggi). Lapisan = tingkat FPN tempat ROI itu dipool.
  * YOLOv8: logit(skor kelas target) pada satu anchor (anchor yang kotaknya IoU >= 0.5 dengan kotak target
    dengan skor kelas tertinggi). Lapisan = keluaran neck pada tingkat anchor itu (P3/P4/P5).
  * Kasus FN: kotak target = kotak GT; yang dijelaskan adalah skor kelas yang SUDAH rendah pada kotak itu,
    BUKAN 'mengapa model tidak mendeteksi'.
`approx=True` berarti tidak ada kandidat dengan IoU >= 0.5 sehingga dipakai kandidat IoU tertinggi; jangan
menafsirkan peta itu sebagai penjelasan kotak target.

KETERBATASAN (ringkas; uraian lengkap ada di docs/anggota4_evaluasi.md):
  1. Per deteksi, bukan per citra; bergantung pada aturan pemilihan target di atas.
  2. Tidak menjelaskan regresi kotak, NMS, atau keputusan RPN (Faster R-CNN) / penugasan anchor (YOLO).
  3. Faster R-CNN: gradien hanya mengalir lewat RoIAlign di dalam proposal, jadi peta SELALU terkurung di
     sekitar proposal secara konstruksi; tidak bisa membuktikan model memakai atau mengabaikan konteks.
  4. Resolusi kasar (stride 4/8/16/32); platelet berukuran puluhan piksel hanya beberapa sel fitur.
  5. Bukan penjelasan kausal; saturasi gradien (dipakai logit untuk mengurangi), ReLU membuang bukti negatif,
     8 kasus = anekdot. Peta kedua model TIDAK sebanding (lapisan, stride, skalar target berbeda).
"""
from __future__ import annotations

import copy
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from error_analysis import crop_window
from eval_shared import CLASS_IDS


def _minmax01(cam: torch.Tensor) -> torch.Tensor:
    cam = cam - cam.min()
    m = cam.max()
    return cam / m if float(m) > 0 else cam


def _resize(cam: torch.Tensor, size_hw) -> torch.Tensor:
    return F.interpolate(cam[None, None], size=tuple(int(s) for s in size_hw), mode="bilinear", align_corners=False)[0, 0]


def _xyxy_from_xywh(b):
    x, y, w, h = [float(v) for v in b]
    return [x, y, x + w, y + h]


# ------------------------------------------------------------------------------- Faster R-CNN
def gradcam_frcnn(model, img_rgb: np.ndarray, target_xywh, target_cat: int, device=None) -> dict:
    """target_cat = category_id COCO (1..3) = indeks kelas pada box_predictor (0 = latar belakang)."""
    from torchvision.ops import box_iou
    device = device or next(model.parameters()).device
    model.eval()
    H, W = img_rgb.shape[:2]
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float().div(255).to(device)
    x1, y1, x2, y2 = _xyxy_from_xywh(target_xywh)
    with torch.enable_grad():
        images, _ = model.transform([x])
        with torch.no_grad():
            feats0 = model.backbone(images.tensors)
        feats = OrderedDict((k, v.detach().clone().requires_grad_(True)) for k, v in feats0.items())
        with torch.no_grad():
            proposals, _ = model.rpn(images, feats)
        rh = model.roi_heads
        bf = rh.box_roi_pool(feats, proposals, images.image_sizes)
        bf = rh.box_head(bf)
        logits, reg = rh.box_predictor(bf)
        dec = rh.box_coder.decode(reg, proposals)                               # [N, C, 4] (koordinat citra ter-resize)
        hr, wr = images.image_sizes[0]
        sy, sx = hr / H, wr / W
        tb = torch.tensor([[x1 * sx, y1 * sy, x2 * sx, y2 * sy]], dtype=torch.float32, device=device)
        iou = box_iou(dec[:, target_cat, :].detach(), tb)[:, 0]
        prob = torch.softmax(logits.detach(), dim=-1)[:, target_cat]
        ok = iou >= 0.5
        j = int(torch.where(ok, prob, torch.full_like(prob, -1.0)).argmax()) if bool(ok.any()) else int(iou.argmax())
        score = logits[j, target_cat]
        model.zero_grad(set_to_none=True)
        score.backward()
    best_k, best_mag = None, 0.0
    for k in ("0", "1", "2", "3"):
        g = feats[k].grad
        if g is not None:
            mag = float(g.abs().sum())
            if mag > best_mag:
                best_k, best_mag = k, mag
    if best_k is None:
        raise RuntimeError("gradien nol pada semua tingkat FPN (proposal tak terpool?); periksa target")
    g, a = feats[best_k].grad, feats[best_k].detach()
    cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))[0]
    Hp, Wp = images.tensors.shape[-2:]
    cam = _resize(cam, (Hp, Wp))[:hr, :wr]
    cam = _minmax01(_resize(cam, (H, W))).detach().cpu().numpy()
    pb = dec[j, target_cat].detach().cpu().numpy() / np.array([sx, sy, sx, sy])
    return {"cam": cam, "level": f"FPN-{best_k}", "iou_target": float(iou[j]), "prob": float(prob[j]),
            "logit": float(score.detach()), "approx": not bool(ok.any()), "proposal_xyxy": pb.tolist()}


# ----------------------------------------------------------------------------------------- YOLO
def gradcam_yolo(yolo, img_rgb: np.ndarray, target_xywh, target_cat0: int, imgsz: int, device=None) -> dict:
    """target_cat0 = id YOLO (0=WBC, 1=RBC, 2=Platelets). Bekerja pada salinan model agar cache anchor/strides
    milik objek YOLO asli tidak tersentuh. ASUMSI yang diperiksa saat dijalankan: keluaran eval Detect berbentuk
    [1, 4+nc, A] dengan kotak xywh (piksel masukan) lalu skor kelas (sudah sigmoid)."""
    from torchvision.ops import box_iou
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    net = copy.deepcopy(yolo.model).to(device).float().eval()
    H, W = img_rgb.shape[:2]
    r = min(imgsz / H, imgsz / W)
    nh, nw = int(round(H * r)), int(round(W * r))
    stride = int(net.stride.max()) if hasattr(net, "stride") else 32
    Hp, Wp = int(math.ceil(nh / stride) * stride), int(math.ceil(nw / stride) * stride)
    canvas = np.full((Hp, Wp, 3), 114, np.uint8)
    canvas[:nh, :nw] = np.asarray(Image.fromarray(img_rgb).resize((nw, nh), Image.BILINEAR))
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None].float().div(255).to(device).requires_grad_(True)

    det = net.model[-1]
    idxs = list(det.f) if isinstance(det.f, (list, tuple)) else [det.f]
    if hasattr(det, "shape"):
        det.shape = None                                    # paksa hitung ulang anchor di luar inference_mode
    store: dict[int, torch.Tensor] = {}

    def _hook(i):
        def h(_m, _inp, out):
            store[i] = out
            if out.requires_grad:
                out.retain_grad()
        return h

    handles = [net.model[i].register_forward_hook(_hook(i)) for i in idxs]
    x1, y1, x2, y2 = [v * r for v in _xyxy_from_xywh(target_xywh)]
    try:
        with torch.inference_mode(False), torch.enable_grad():
            out = net(x)
            y = out[0] if isinstance(out, (list, tuple)) else out
            if not (torch.is_tensor(y) and y.ndim == 3 and y.shape[1] > 4):
                raise RuntimeError(f"keluaran Detect tak terduga: {type(out)} {getattr(y, 'shape', None)}")
            nc = y.shape[1] - 4
            if not (0 <= target_cat0 < nc):
                raise ValueError(f"kelas {target_cat0} di luar {nc} kelas")
            b = y[0, :4, :].T
            xyxy = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
            tb = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=device)
            xyc = xyxy.detach().clone()                          # Ultralytics memotong kotak ke batas citra
            xyc[:, 0::2] = xyc[:, 0::2].clamp(0, Wp)
            xyc[:, 1::2] = xyc[:, 1::2].clamp(0, Hp)
            iou = box_iou(xyc, tb)[:, 0]
            prob = y[0, 4 + target_cat0, :]
            ok = iou >= 0.5
            a = int(torch.where(ok, prob.detach(), torch.full_like(prob, -1.0)).argmax()) if bool(ok.any()) else int(iou.argmax())
            score = torch.logit(prob[a].clamp(1e-6, 1 - 1e-6))
            net.zero_grad(set_to_none=True)
            score.backward()
        sizes = [store[i].shape[2] * store[i].shape[3] for i in idxs]
        if sum(sizes) != y.shape[2]:
            raise RuntimeError(f"jumlah anchor {y.shape[2]} != jumlah sel fitur {sum(sizes)}: urutan anchor tak sesuai asumsi")
        cum = np.cumsum(sizes)
        lvl = int(np.searchsorted(cum, a, side="right"))
        feat = store[idxs[lvl]]
        if feat.grad is None:
            raise RuntimeError("gradien tidak tersedia pada lapisan target")
        cam = F.relu((feat.grad.mean(dim=(2, 3), keepdim=True) * feat.detach()).sum(1))[0]
        cam = _resize(cam, (Hp, Wp))[:nh, :nw]
        cam = _minmax01(_resize(cam, (H, W))).detach().cpu().numpy()
        return {"cam": cam, "level": f"layer{idxs[lvl]}(stride {stride // (2 ** (len(idxs) - 1 - lvl))})",
                "iou_target": float(iou[a]), "prob": float(prob[a].detach()), "logit": float(score.detach()),
                "approx": not bool(ok.any())}
    finally:
        for h in handles:
            h.remove()
        del net


# ----------------------------------------------------------------------------- penjalanan kasus
def run_gradcam_cases(kind: str, model, cases, gt, data_root, out_png, imgsz: int = 640, device=None,
                      ncols: int = 4, title: str = ""):
    """cases: DataFrame dari error_analysis.select_gradcam_cases. kind: 'yolo' | 'frcnn'.
    Mengembalikan (path_png, DataFrame info per kasus)."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.patches import Rectangle

    n = len(cases)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    info = []
    for ax, (_, r) in zip(axes.ravel(), cases.iterrows()):
        with Image.open(Path(data_root) / r["file_name"]) as im:
            img = np.array(im.convert("RGB"))
        H, W = img.shape[:2]
        tgt = (r["x"], r["y"], r["w"], r["h"])
        cat = int(r["cat"])
        if kind == "frcnn":
            res = gradcam_frcnn(model, img, tgt, cat, device)
        elif kind == "yolo":
            res = gradcam_yolo(model, img, tgt, cat - 1, imgsz, device)
        else:
            raise ValueError(kind)
        x0, y0, side = crop_window(W, H, tgt, ctx=4.0, min_side=96)
        crop = img[y0:y0 + side, x0:x0 + side]
        cam = res["cam"][y0:y0 + side, x0:x0 + side]
        ax.axis("on")
        ax.imshow(crop, extent=(0, side, side, 0), interpolation="bilinear")
        ax.imshow(colormaps["jet"](cam)[..., :3], extent=(0, side, side, 0), alpha=0.45, interpolation="bilinear")
        for a in gt.anns:
            if gt.pos[a["image_id"]] == r["img"]:
                bx, by, bw, bh = a["bbox"]
                if bx < x0 + side and bx + bw > x0 and by < y0 + side and by + bh > y0:
                    ax.add_patch(Rectangle((bx - x0, by - y0), bw, bh, fill=False, ec="lime", lw=1.0))
        ax.add_patch(Rectangle((tgt[0] - x0, tgt[1] - y0), tgt[2], tgt[3], fill=False, ec="white", lw=1.2, ls="--"))
        ax.set_xlim(0, side)
        ax.set_ylim(side, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        sc = "" if np.isnan(r["score"]) else f" s={r['score']:.2f}"
        flag = " (approx)" if res["approx"] else ""
        ax.set_title(f"{r['grup']}: {r['kind']}/{r['subtype']}\n{r['kelas']}{sc} | {r['sumber']} | {res['level']}{flag}",
                     fontsize=7, color=("darkgreen" if r["grup"] == "benar" else "firebrick"), pad=3)
        info.append({"image_id": int(r["image_id"]), "grup": r["grup"], "kind": r["kind"], "subtype": r["subtype"],
                     "sumber": r["sumber"], "level": res["level"], "iou_target": res["iou_target"],
                     "prob_atau_skor": res["prob"], "logit": res["logit"], "approx": res["approx"],
                     "cam_massa_dalam_kotak": float(res["cam"][int(max(tgt[1], 0)):int(tgt[1] + tgt[3]) + 1,
                                                               int(max(tgt[0], 0)):int(tgt[0] + tgt[2]) + 1].sum()
                                                    / max(res["cam"].sum(), 1e-9))})
    fig.suptitle((title + "\n" if title else "") + "Grad-CAM (jet) pada target tunggal; hijau = GT, putih putus-putus = kotak target",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    import pandas as pd
    return str(out_png), pd.DataFrame(info)
