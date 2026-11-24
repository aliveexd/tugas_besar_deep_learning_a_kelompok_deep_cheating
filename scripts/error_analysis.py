"""Analisis kesalahan detektor (Anggota 4): taksonomi FP/FN sederhana + galeri contoh salah.

Tanpa torch. Memakai GT dan ModelEval dari scripts/eval_shared.py.

Taksonomi (ala TIDE tetapi SEDERHANA dan buatan sendiri; BUKAN toolbox TIDE, Bolya dkk. 2020), pada ambang
confidence `thr` dan IoU 0.5 (pencocokan serakah per kelas menurut skor, seperti COCO):
  TP                      : prediksi cocok dengan GT sekelas (IoU >= 0.5).
  FP / duplikat           : IoU >= 0.5 dengan GT sekelas yang sudah dipakai prediksi lain.
  FP / salah_kelas        : IoU >= 0.5 dengan GT kelas lain.
  FP / lokalisasi         : IoU 0.1..0.5 dengan GT sekelas (kotak ada, tetapi kurang pas).
  FP / lainnya            : IoU 0.1..0.5 dengan GT kelas lain saja.
  FP / latar_belakang     : IoU < 0.1 dengan semua GT (bisa saja platelet asli yang tidak dianotasi:
                            periksa visual, terutama di sumber 575x575).
  FN / skor_rendah        : ada prediksi sekelas IoU >= 0.5 tetapi skornya < thr (korban ambang).
  FN / salah_kelas        : ada prediksi kelas lain (skor >= thr) IoU >= 0.5.
  FN / tumpang_tindih     : prediksi sekelas IoU >= 0.5 ada, tetapi sudah dipakai GT lain (sel bertumpuk).
  FN / lokalisasi         : prediksi sekelas (skor >= thr) IoU 0.1..0.5.
  FN / terlewat           : tidak ada prediksi yang relevan.
Urutan pemeriksaan = urutan daftar di atas (kondisi pertama yang terpenuhi menang).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from eval_shared import CLASS_IDS, PLATELET_ID, GT, ModelEval


def _xyxy(b) -> np.ndarray:
    b = np.asarray(b, float).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-12)


def size_bin(area: float) -> str:
    """Definisi COCO pada piksel citra asli: small < 32^2, medium < 96^2, large selebihnya."""
    return "kecil" if area < 32 ** 2 else ("sedang" if area < 96 ** 2 else "besar")


def classify_errors(me: ModelEval, thr: float, iou_thr: float = 0.5, bg_iou: float = 0.1) -> pd.DataFrame:
    """Satu baris per TP/FP/FN. Kotak fokus (x,y,w,h): prediksi untuk TP/FP, GT untuk FN."""
    gt, tab = me.gt, me.table
    by_img_g: dict[int, list] = {}
    for a in gt.anns:
        by_img_g.setdefault(gt.pos[a["image_id"]], []).append(a)
    tab_sorted = tab.sort_values("score", ascending=False, kind="mergesort")
    by_img_d = {i: g for i, g in tab_sorted.groupby("img")}
    rows = []
    for i in range(len(gt)):
        G = by_img_g.get(i, [])
        Gb = _xyxy([a["bbox"] for a in G]) if G else np.zeros((0, 4))
        Gc = np.array([a["category_id"] for a in G], int)
        Dt = by_img_d.get(i)
        if Dt is None:
            Dt = tab_sorted.iloc[0:0]
        Db = _xyxy(Dt[["x", "y", "w", "h"]].to_numpy()) if len(Dt) else np.zeros((0, 4))
        Dc, Ds = Dt["cat"].to_numpy(int), Dt["score"].to_numpy(float)
        hi = Ds >= thr
        iou = _iou(Db, Gb)                                                   # [nd, ng]
        gmatch = np.full(len(G), -1)
        dmatch = np.full(len(Dt), -1)
        for d in np.flatnonzero(hi):
            if len(G) == 0:
                break
            cand = (Gc == Dc[d]) & (gmatch < 0)
            if cand.any():
                v = np.where(cand, iou[d], -1.0)
                j = int(v.argmax())
                if v[j] >= iou_thr:
                    gmatch[j], dmatch[d] = d, j
        base = {"img": i, "image_id": gt.img_ids[i], "file_name": gt.images[i]["file_name"], "sumber": gt.src[i]}

        def _row(kind, sub, cat, score, box, gt_box=None, iou_v=np.nan):
            x, y, w, h = box
            r = dict(base, kind=kind, subtype=sub, cat=int(cat), kelas=CLASS_IDS[int(cat)], score=score,
                     x=float(x), y=float(y), w=float(w), h=float(h), ukuran=size_bin(w * h), iou=float(iou_v))
            if gt_box is not None:
                r.update(gx=float(gt_box[0]), gy=float(gt_box[1]), gw=float(gt_box[2]), gh=float(gt_box[3]))
            return r

        for d in np.flatnonzero(hi):
            box = Dt[["x", "y", "w", "h"]].to_numpy()[d]
            if dmatch[d] >= 0:
                j = dmatch[d]
                rows.append(_row("TP", "benar", Dc[d], Ds[d], box, G[j]["bbox"], iou[d, j]))
                continue
            io = iou[d] if len(G) else np.zeros(0)
            same = Gc == Dc[d]
            if len(G) and (io[same] >= iou_thr).any():
                sub = "duplikat"
            elif len(G) and (io[~same] >= iou_thr).any():
                sub = "salah_kelas"
            elif len(G) and (io[same] >= bg_iou).any():
                sub = "lokalisasi"
            elif len(G) and (io >= bg_iou).any():
                sub = "lainnya"
            else:
                sub = "latar_belakang"
            rows.append(_row("FP", sub, Dc[d], Ds[d], box, iou_v=(io.max() if len(io) else np.nan)))
        for j in np.flatnonzero(gmatch < 0):
            io = iou[:, j] if len(Dt) else np.zeros(0)
            same = Dc == Gc[j]
            if len(Dt) and (same & ~hi & (io >= iou_thr)).any():
                sub = "skor_rendah"
            elif len(Dt) and (~same & hi & (io >= iou_thr)).any():
                sub = "salah_kelas"
            elif len(Dt) and (same & hi & (io >= iou_thr)).any():
                sub = "tumpang_tindih"
            elif len(Dt) and (same & hi & (io >= bg_iou)).any():
                sub = "lokalisasi"
            else:
                sub = "terlewat"
            rows.append(_row("FN", sub, Gc[j], np.nan, G[j]["bbox"], iou_v=(io.max() if len(io) else np.nan)))
    return pd.DataFrame(rows)


def error_tables(errs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Ringkasan hitungan: per kelas, platelet per sumber, platelet per ukuran kotak."""
    out = {}
    out["per_kelas"] = (errs.groupby(["kind", "subtype", "kelas"]).size().unstack("kelas", fill_value=0)
                        .reindex(columns=list(CLASS_IDS.values()), fill_value=0))
    p = errs[errs.cat == PLATELET_ID]
    out["platelet_per_sumber"] = p.groupby(["sumber", "kind", "subtype"]).size().unstack(["kind", "subtype"], fill_value=0)
    out["platelet_per_ukuran"] = p.groupby(["ukuran", "kind"]).size().unstack("kind", fill_value=0)
    return out


def select_gallery(errs: pd.DataFrame, cats=(PLATELET_ID,), min_total: int = 8, per_group: int = 2,
                   seed: int = 0) -> pd.DataFrame:
    """Pilih contoh salah (FP/FN) untuk galeri, DETERMINISTIK (seed tetap), tanpa memilih yang 'bagus':
    FP = skor tertinggi per kelompok (kesalahan paling yakin), FN = acak bertingkat per sumber.
    Tiap kelompok (kind, subtype) diwakili maksimal `per_group` contoh; bila total < min_total, ditambah
    dari kelompok terbesar bergiliran. Bila kesalahan yang tersedia kurang dari min_total, semuanya diambil."""
    e = errs[(errs.kind != "TP") & (errs.cat.isin(cats))].copy()
    if e.empty:
        return e
    rng = np.random.default_rng(seed)
    cand = {}
    for (kind, sub), g in e.groupby(["kind", "subtype"]):
        if kind == "FP":
            g = g.sort_values("score", ascending=False, kind="mergesort")
            order = list(g.index)
        else:
            order = list(g.index[rng.permutation(len(g))])
        # selang-seling antar sumber agar contoh beragam
        by_src: dict[str, list] = {}
        for ix in order:
            by_src.setdefault(g.loc[ix, "sumber"], []).append(ix)
        mixed, lists = [], [by_src[k] for k in sorted(by_src)]
        while any(lists):
            for lst in lists:
                if lst:
                    mixed.append(lst.pop(0))
        cand[(kind, sub)] = mixed
    keys = sorted(cand, key=lambda k: -len(cand[k]))
    chosen: list = []
    taken = {k: 0 for k in keys}
    for k in keys:
        for ix in cand[k][:per_group]:
            chosen.append(ix)
        taken[k] = min(per_group, len(cand[k]))
    while len(chosen) < min_total and any(taken[k] < len(cand[k]) for k in keys):
        for k in keys:
            if len(chosen) >= min_total:
                break
            if taken[k] < len(cand[k]):
                chosen.append(cand[k][taken[k]])
                taken[k] += 1
    sel = e.loc[chosen].copy()
    sel["grup"] = sel["kind"] + "/" + sel["subtype"]
    sel = sel.sort_values(["kind", "subtype", "sumber"], kind="mergesort")
    sel.attrs["n_available"] = int(len(e))
    return sel


def crop_window(W: int, H: int, box_xywh, ctx: float = 3.0, min_side: int = 96):
    """Jendela persegi di sekitar kotak (konteks ctx x ukuran kotak, minimal min_side piksel), di dalam citra."""
    x, y, w, h = box_xywh
    cx, cy = x + w / 2, y + h / 2
    side = int(round(min(max(min_side, ctx * max(w, h)), W, H)))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    return x0, y0, side


def draw_gallery(me: ModelEval, sel: pd.DataFrame, thr: float, data_root, out_png, ncols: int = 4,
                 ctx: float = 3.0, min_side: int = 96, title: str = ""):
    """Galeri berkelompok. Hijau = GT, merah = prediksi (skor >= thr), putus-putus putih = kotak fokus.
    Menyimpan PNG dan mengembalikan (path, DataFrame contoh)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    gt, tab = me.gt, me.table
    n = len(sel)
    if n == 0:
        raise ValueError("tidak ada contoh salah untuk digambar")
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), sel.iterrows()):
        im = Image.open(Path(data_root) / r.file_name).convert("RGB")
        W, H = im.size
        x0, y0, side = crop_window(W, H, (r.x, r.y, r.w, r.h), ctx, min_side)
        ax.imshow(np.asarray(im.crop((x0, y0, x0 + side, y0 + side))), extent=(0, side, side, 0), interpolation="bilinear")
        def _inside(bx, by, bw, bh):          # hanya gambar kotak yang beririsan dengan jendela
            return bx < x0 + side and bx + bw > x0 and by < y0 + side and by + bh > y0

        for a in gt.anns:
            if gt.pos[a["image_id"]] == r.img and _inside(*a["bbox"]):
                bx, by, bw, bh = a["bbox"]
                ax.add_patch(Rectangle((bx - x0, by - y0), bw, bh, fill=False, ec="lime", lw=1.2))
                ax.text(bx - x0 + 1, by - y0 + 1, CLASS_IDS[a["category_id"]][0], color="lime", fontsize=6,
                        va="top", clip_on=True)
        d = tab[(tab.img == r.img) & (tab.score >= thr)]
        for _, p in d.iterrows():
            if not _inside(p["x"], p["y"], p["w"], p["h"]):
                continue
            ax.add_patch(Rectangle((p["x"] - x0, p["y"] - y0), p["w"], p["h"], fill=False, ec="red", lw=1.2))
            ax.text(p["x"] - x0 + 1, p["y"] - y0 + p["h"] - 1, f'{CLASS_IDS[int(p["cat"])][0]}{p["score"]:.2f}',
                    color="red", fontsize=6, va="bottom", clip_on=True)
        ax.add_patch(Rectangle((r.x - x0 - 2, r.y - y0 - 2), r.w + 4, r.h + 4, fill=False, ec="white", lw=1.0, ls="--"))
        ax.set_xlim(0, side)
        ax.set_ylim(side, 0)
        sc = "" if np.isnan(r.score) else f" s={r.score:.2f}"
        ax.set_title(f"{r.kind}/{r.subtype}\n{r.kelas}{sc} | {r.sumber} | id={r.image_id}", fontsize=7,
                     color=("firebrick" if r.kind == "FP" else "navy"), pad=3)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle((title + "\n" if title else "") + f"Hijau = GT, merah = prediksi (skor >= {thr:g}), putih putus-putus = kotak fokus",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    return str(out_png), sel.reset_index(drop=True)


def select_gradcam_cases(errs: pd.DataFrame, cat: int = PLATELET_ID, n_correct: int = 4, n_wrong: int = 4,
                         seed: int = 0) -> pd.DataFrame:
    """Pilih kasus untuk Grad-CAM, deterministik dan TANPA memilih yang 'cantik':
      benar : n_correct TP kelas `cat` yang diambil acak, bergilir antar sumber citra (bukan skor tertinggi).
      salah : separuh FP (skor tertinggi = paling yakin salah), separuh FN (acak bergilir antar sumber);
              bila salah satu jenis kurang, diisi dari jenis lain. Bila total kasus salah < n_wrong, semuanya diambil.
    Kolom 'grup' = 'benar' / 'salah'."""
    rng = np.random.default_rng(seed)
    e = errs[errs.cat == cat]

    def interleave(g: pd.DataFrame) -> list:
        order = list(g.index[rng.permutation(len(g))])
        by: dict[str, list] = {}
        for ix in order:
            by.setdefault(g.loc[ix, "sumber"], []).append(ix)
        mixed, lists = [], [by[k] for k in sorted(by)]
        while any(lists):
            for lst in lists:
                if lst:
                    mixed.append(lst.pop(0))
        return mixed

    tp = interleave(e[e.kind == "TP"])[:n_correct]
    fp = list(e[e.kind == "FP"].sort_values("score", ascending=False, kind="mergesort").index)
    fn = interleave(e[e.kind == "FN"])
    n_fp, n_fn = n_wrong // 2, n_wrong - n_wrong // 2
    pick_fp, pick_fn = fp[:n_fp], fn[:n_fn]
    short = n_wrong - len(pick_fp) - len(pick_fn)
    if short > 0:
        pick_fp += fp[len(pick_fp):len(pick_fp) + short]
        short = n_wrong - len(pick_fp) - len(pick_fn)
    if short > 0:
        pick_fn += fn[len(pick_fn):len(pick_fn) + short]
    out = pd.concat([e.loc[tp].assign(grup="benar"), e.loc[pick_fp + pick_fn].assign(grup="salah")])
    return out.reset_index(drop=True)


def add_data_attributes(errs: pd.DataFrame, gt: GT, overlap_thr: float = 0.3, border_px: int = 2) -> pd.DataFrame:
    """Atribut data yang BISA dihitung otomatis, untuk frekuensi kategori kesalahan (rubrik K6):
      bertumpuk : kotak fokus beririsan >= overlap_thr (irisan / luas kotak yang lebih kecil) dengan GT lain
                  pada citra yang sama (kelas apa pun, selain dirinya sendiri).
      kecil     : luas kotak fokus < 32^2 px (definisi COCO small, piksel citra asli).
      tepi      : kotak fokus menyentuh tepi citra (<= border_px piksel): sel terpotong.
    Kategori yang butuh mata manusia (tepi buram, artefak pewarnaan, platelet tak dianotasi) TIDAK diisi
    otomatis: kolom 'kategori_manual' dikosongkan untuk diisi tim setelah memeriksa galeri."""
    by_img: dict[int, list] = {}
    for a in gt.anns:
        by_img.setdefault(gt.pos[a["image_id"]], []).append(a["bbox"])
    ber, tep = [], []
    for r in errs.itertuples():
        W, H = gt.size[r.img]
        box = np.array([[r.x, r.y, r.x + r.w, r.y + r.h]], float)
        others = by_img.get(r.img, [])
        flag = False
        if others:
            ob = _xyxy(others)
            ix = np.clip(np.minimum(box[0, 2], ob[:, 2]) - np.maximum(box[0, 0], ob[:, 0]), 0, None)
            iy = np.clip(np.minimum(box[0, 3], ob[:, 3]) - np.maximum(box[0, 1], ob[:, 1]), 0, None)
            inter = ix * iy
            amin = np.minimum(r.w * r.h, (ob[:, 2] - ob[:, 0]) * (ob[:, 3] - ob[:, 1]))
            frac = inter / np.maximum(amin, 1e-9)
            same = (np.abs(ob[:, 0] - box[0, 0]) < 1e-6) & (np.abs(ob[:, 1] - box[0, 1]) < 1e-6) & \
                   (np.abs(ob[:, 2] - box[0, 2]) < 1e-6) & (np.abs(ob[:, 3] - box[0, 3]) < 1e-6)
            flag = bool(((frac >= overlap_thr) & ~same).any())
        ber.append(flag)
        tep.append(bool(r.x <= border_px or r.y <= border_px or r.x + r.w >= W - border_px or r.y + r.h >= H - border_px))
    out = errs.copy()
    out["bertumpuk"], out["tepi"] = ber, tep
    out["kecil"] = out["ukuran"] == "kecil"
    out["kategori_manual"] = ""
    return out


def category_frequency(errs: pd.DataFrame, cat: int | None = PLATELET_ID) -> pd.DataFrame:
    """Frekuensi tiap atribut data pada FP dan FN (tabel yang diminta rubrik K6)."""
    e = errs if cat is None else errs[errs.cat == cat]
    rows = []
    for kind in ("FP", "FN"):
        k = e[e.kind == kind]
        row = {"jenis": kind, "n": int(len(k))}
        for col in ("kecil", "bertumpuk", "tepi"):
            row[col] = int(k[col].sum())
        row["tanpa_atribut_otomatis"] = int((~(k["kecil"] | k["bertumpuk"] | k["tepi"])).sum())
        rows.append(row)
    return pd.DataFrame(rows)
