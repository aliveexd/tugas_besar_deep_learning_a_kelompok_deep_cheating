"""Evaluasi bersama YOLOv8 dan Faster R-CNN (Anggota 4), TXL-PBC, topik T1-B.

Format prediksi tunggal untuk KEDUA model (daftar dict, sama dengan format hasil COCO):
    {"image_id": int, "category_id": 1|2|3, "bbox": [x, y, w, h], "score": float}
bbox = piksel absolut pada citra ASLI. category_id COCO = id YOLO + 1 (1=WBC, 2=RBC, 3=Platelets).
Konversi prediksi YOLO: yolo_to_coco_records(image_id, xyxy, conf, cls0)  (cls0 berbasis 0).

Modul ini sengaja TIDAK memakai torch (bisa diuji di CPU/laptop). Isi:
  - GT: pembaca anotasi COCO (+ buang baris duplikat persis, sama seperti Ultralytics)
  - ModelEval: mAP@0.5, mAP@[.5:.95], AP/recall per kelas, galat hitung (MAE per kelas per citra),
    presisi/recall pada ambang tertentu, hasil per sumber citra
  - bootstrap(): interval kepercayaan persentil atas citra (berpasangan antar model)
  - threshold_curve()/select_threshold()/freeze_thresholds(): ablasi ambang confidence (dipilih di val)
  - append_final_test_row(): tulis ke logs/final_test_results.csv (tanpa menimpa, tanpa duplikat)

Aturan keras tim: kembangkan semua alat pada prediksi VAL. Test/test_unflagged hanya untuk
konfigurasi final, sekali per konfigurasi (CLI menolak split test tanpa --final).

Catatan keterbandingan: AP dihitung dengan pycocotools (interpolasi 101 titik), BUKAN dengan kode
metrik internal Ultralytics, jadi angka YOLO di sini bisa sedikit berbeda dari `model.val()`.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_IDS = {1: "WBC", 2: "RBC", 3: "Platelets"}
CAT_IDS = [1, 2, 3]
PLATELET_ID = 3
KNOWN_SIZES = [(256, 256), (360, 363), (575, 575), (640, 480)]
AREA_LABELS = ["all", "small", "medium", "large"]      # urutan areaRng bawaan pycocotools
REC_THRS = np.linspace(0.0, 1.0, 101)                  # = recThrs bawaan pycocotools
FINAL_TEST_COLS = ["run_id", "test_split", "map50", "map50_95", "ap50_platelet", "mae_wbc", "mae_rbc",
                   "mae_platelet", "bootstrap_ci_platelet", "date", "notes"]


# ------------------------------------------------------------------------------ util kecil
def source_label(w: int, h: int) -> str:
    """Kelompok sumber citra = ukuran citra (sesuai kesepakatan tim)."""
    return f"{int(w)}x{int(h)}" if (int(w), int(h)) in KNOWN_SIZES else "lainnya"


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def fmt_ci(lo: float, hi: float, digits: int = 3) -> str:
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)) or (isinstance(hi, float) and math.isnan(hi)):
        return "NA"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


# ---------------------------------------------------------------- konversi format prediksi
def to_coco_records(image_id, xyxy, scores, category_ids) -> list[dict]:
    """xyxy piksel absolut (x1,y1,x2,y2) -> daftar dict format COCO (bbox xywh)."""
    out = []
    for (x1, y1, x2, y2), s, c in zip(xyxy, scores, category_ids):
        out.append({"image_id": int(image_id), "category_id": int(c),
                    "bbox": [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)],
                    "score": float(s)})
    return out


def yolo_to_coco_records(image_id, xyxy, conf, cls0) -> list[dict]:
    """Prediksi YOLO (kelas berbasis 0) -> format COCO: category_id = cls0 + 1, bbox xywh piksel absolut."""
    return to_coco_records(image_id, xyxy, conf, [int(c) + 1 for c in cls0])


# ------------------------------------------------------------------------------------- GT
class GT:
    """Anotasi COCO (category_id 1-3). dedup=True membuang baris duplikat persis (citra, kelas, kotak),
    seperti yang dilakukan Ultralytics saat memuat label (val: 15 duplikat RBC). Jumlah yang dibuang dicatat."""

    def __init__(self, ann_file, dedup: bool = True):
        from pycocotools.coco import COCO
        self.ann_file = str(ann_file)
        self.name = Path(ann_file).stem
        coco = json.loads(Path(ann_file).read_text())
        self.images = sorted(coco["images"], key=lambda d: d["id"])
        self.img_ids = [im["id"] for im in self.images]
        self.pos = {i: n for n, i in enumerate(self.img_ids)}
        names = {c["id"]: c["name"] for c in coco["categories"]}
        assert names == CLASS_IDS, f"kategori tidak sesuai harapan {CLASS_IDS}: {names}"

        keep, seen = [], set()
        self.n_raw_by_cat = Counter()
        self.n_dup_by_cat = Counter()
        self.n_crowd = 0
        for a in coco["annotations"]:
            assert a["image_id"] in self.pos, f"anotasi merujuk citra tak dikenal: {a['image_id']}"
            if a.get("iscrowd", 0):
                self.n_crowd += 1
                continue
            self.n_raw_by_cat[a["category_id"]] += 1
            key = (a["image_id"], a["category_id"], tuple(round(float(v), 2) for v in a["bbox"]))
            if dedup and key in seen:
                self.n_dup_by_cat[a["category_id"]] += 1
                continue
            seen.add(key)
            keep.append(a)
        self.dedup = dedup
        self.anns = keep
        self.n_dup_removed = int(sum(self.n_dup_by_cat.values()))

        d = {"images": self.images, "annotations": keep, "categories": coco["categories"]}
        with contextlib.redirect_stdout(io.StringIO()):
            c = COCO()
            c.dataset = d
            c.createIndex()
        self.coco = c

        I = len(self.images)
        self.size = np.array([[im["width"], im["height"]] for im in self.images])
        self.src = np.array([source_label(w, h) for w, h in self.size])
        self.n_gt = {cid: np.zeros(I, dtype=int) for cid in CAT_IDS}
        for a in keep:
            self.n_gt[a["category_id"]][self.pos[a["image_id"]]] += 1

    def __len__(self):
        return len(self.images)

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in ["semua"] + sorted(set(self.src)):
            m = np.ones(len(self), bool) if s == "semua" else (self.src == s)
            rows.append({"sumber": s, "n_citra": int(m.sum()),
                         **{f"n_gt_{CLASS_IDS[c]}": int(self.n_gt[c][m].sum()) for c in CAT_IDS},
                         "citra_tanpa_platelet": int((self.n_gt[PLATELET_ID][m] == 0).sum()),
                         "MAE_selalu_0_platelet": float(self.n_gt[PLATELET_ID][m].mean()) if m.any() else float("nan")})
        return pd.DataFrame(rows)


def always_zero_mae(gt: GT, cat: int = PLATELET_ID, idx=None) -> float:
    """MAE hitung per citra bila model selalu memprediksi 0 objek (pembanding yang jujur)."""
    n = gt.n_gt[cat] if idx is None else gt.n_gt[cat][np.asarray(idx)]
    return float(np.mean(n))


# ------------------------------------------------------------------------- validasi prediksi
def clean_predictions(gt: GT, dets: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Periksa format prediksi. Kesalahan format (kunci hilang, image_id/kategori asing, NaN, skor di luar
    [0,1]) -> ValueError. Kotak lebar/tinggi <= 0 dibuang dan dihitung di laporan."""
    if not isinstance(dets, list):
        raise ValueError("prediksi harus berupa list dict")
    if len(dets) == 0:
        raise ValueError("prediksi kosong")
    need = {"image_id", "category_id", "bbox", "score"}
    for d in dets[:1000]:
        miss = need - set(d)
        if miss:
            raise ValueError(f"kunci hilang pada prediksi: {miss}")
    df = pd.DataFrame({"image_id": [d["image_id"] for d in dets],
                       "cat": [d["category_id"] for d in dets],
                       "score": [d["score"] for d in dets]})
    bb = np.asarray([d["bbox"] for d in dets], dtype=float)
    if bb.ndim != 2 or bb.shape[1] != 4:
        raise ValueError(f"bbox harus [x,y,w,h]; bentuk {bb.shape}")
    if not np.isfinite(bb).all() or not np.isfinite(df["score"].to_numpy(float)).all():
        raise ValueError("ada NaN/inf pada bbox atau skor")
    unknown_img = set(df["image_id"]) - set(gt.pos)
    if unknown_img:
        raise ValueError(f"image_id tidak ada di GT {gt.name}: {sorted(unknown_img)[:5]} ...")
    unknown_cat = set(df["cat"]) - set(CAT_IDS)
    if unknown_cat:
        raise ValueError(f"category_id harus 1..3 (YOLO id + 1); ditemukan {sorted(unknown_cat)}")
    if (df["score"] < 0).any() or (df["score"] > 1).any():
        raise ValueError("skor di luar [0,1]")
    df[["x", "y", "w", "h"]] = bb
    bad = (df["w"] <= 0) | (df["h"] <= 0)
    report = {"n_pred_raw": int(len(df)), "n_dropped_nonpositive_box": int(bad.sum())}
    df = df[~bad].copy()
    df["img"] = df["image_id"].map(gt.pos).astype(int)
    df["cat"] = df["cat"].astype(int)
    df = df.reset_index(drop=True)
    report["n_pred"] = int(len(df))
    if len(df) == 0:
        raise ValueError("semua prediksi tidak valid")
    return df, report


# ---------------------------------------------------------------------- AP dari entri per citra
def _accumulate(sc, tp, ig, npig):
    """Salinan logika COCOeval.accumulate untuk satu kelas/area/maxDet. tp, ig: [T, N] bool.
    Mengembalikan (AP per IoU [T], recall per IoU [T]); NaN bila tidak ada GT."""
    T = tp.shape[0]
    if npig == 0:
        return np.full(T, np.nan), np.full(T, np.nan)
    order = np.argsort(-sc, kind="mergesort")
    tp, ig = tp[:, order], ig[:, order]
    tps = np.logical_and(tp, np.logical_not(ig))
    fps = np.logical_and(np.logical_not(tp), np.logical_not(ig))
    tp_sum = np.cumsum(tps, axis=1).astype(float)
    fp_sum = np.cumsum(fps, axis=1).astype(float)
    ap, rec = np.zeros(T), np.zeros(T)
    nd = tp.shape[1]
    for t in range(T):
        tpv, fpv = tp_sum[t], fp_sum[t]
        rc = tpv / npig
        pr = tpv / (fpv + tpv + np.spacing(1))
        rec[t] = rc[-1] if nd else 0.0
        q = np.zeros(len(REC_THRS))
        if nd:
            pr = np.maximum.accumulate(pr[::-1])[::-1]          # envelope presisi (setara loop pycocotools)
            inds = np.searchsorted(rc, REC_THRS, side="left")
            ok = inds < nd
            q[ok] = pr[inds[ok]]
        ap[t] = q.mean()
    return ap, rec


class ModelEval:
    """Satu set prediksi (satu model/seed) terhadap satu GT. Pencocokan COCO (per citra, kelas, IoU)
    dihitung SEKALI oleh pycocotools lalu dipakai ulang untuk subset citra dan bootstrap (pencocokan
    bersifat per citra, jadi resampel citra cukup menyambung ulang entri)."""

    def __init__(self, gt: GT, dets: list[dict], name: str = "model", max_det: int = 100, check: bool = True):
        from pycocotools.cocoeval import COCOeval
        self.gt, self.name, self.max_det = gt, name, max_det
        self.table, self.report = clean_predictions(gt, dets)
        t = self.table
        recs = [{"image_id": int(a), "category_id": int(c), "bbox": [float(x), float(y), float(w), float(h)],
                 "score": float(s)}
                for a, c, x, y, w, h, s in zip(t.image_id, t.cat, t.x, t.y, t.w, t.h, t.score)]
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.coco.loadRes(recs)
            ev = COCOeval(gt.coco, dt, "bbox")
            ev.params.imgIds = list(gt.img_ids)
            ev.params.catIds = list(CAT_IDS)
            ev.params.maxDets = [1, 10, max_det]
            ev.evaluate()
        p = ev.params
        K, A, I = len(p.catIds), len(p.areaRng), len(p.imgIds)
        assert list(p.imgIds) == list(gt.img_ids) and list(p.catIds) == CAT_IDS, "urutan id berubah di pycocotools"
        assert len(ev.evalImgs) == K * A * I, "struktur evalImgs pycocotools berbeda dari yang diasumsikan"
        self.T = len(p.iouThrs)
        self.iou_thrs = np.asarray(p.iouThrs)
        self.I = I
        T = self.T
        self.E = []                                        # E[a][k][i] = (skor, tp[T,n], abaikan[T,n], npig)
        for a in range(A):
            Ea = []
            for k in range(K):
                Ek = []
                for i in range(I):
                    e = ev.evalImgs[k * A * I + a * I + i]
                    if e is None:
                        Ek.append((np.empty(0), np.zeros((T, 0), bool), np.zeros((T, 0), bool), 0))
                    else:
                        sc = np.asarray(e["dtScores"], float)[:max_det]
                        n = len(sc)
                        tp = (np.asarray(e["dtMatches"]).reshape(T, -1)[:, :n] > 0)
                        ig = np.asarray(e["dtIgnore"], bool).reshape(T, -1)[:, :n]
                        npig = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
                        Ek.append((sc, tp, ig, npig))
                Ea.append(Ek)
            self.E.append(Ea)
        # susunan datar (area 'all') untuk presisi/recall pada ambang tertentu
        self._flat = {}
        for k in range(K):
            Ek = self.E[0][k]
            sc = np.concatenate([e[0] for e in Ek]) if I else np.empty(0)
            tp50 = np.concatenate([e[1][0] for e in Ek]) if I else np.empty(0, bool)
            img = np.concatenate([np.full(len(e[0]), i) for i, e in enumerate(Ek)]) if I else np.empty(0, int)
            self._flat[CAT_IDS[k]] = (sc, tp50, img)
        if check:
            self._selfcheck(ev)

    # -- verifikasi terhadap COCOeval.accumulate (menjaga dari perbedaan versi pycocotools)
    def _selfcheck(self, ev):
        with contextlib.redirect_stdout(io.StringIO()):
            ev.accumulate()
        prec = ev.eval["precision"]          # [T, R, K, A, M]
        full = np.arange(self.I)
        for k, cid in enumerate(CAT_IDS):
            ap, _ = self._ap_rec(0, k, full)
            for t in range(self.T):
                ref = prec[t, :, k, 0, -1]
                ref = float(ref[ref > -1].mean()) if (ref > -1).any() else float("nan")
                got = ap[t]
                same = (math.isnan(ref) and math.isnan(got)) or abs(ref - got) < 1e-9
                assert same, f"AP tidak cocok dengan pycocotools (kelas {cid}, IoU idx {t}): {got} vs {ref}"

    def _ap_rec(self, a, k, idx):
        E = self.E[a][k]
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0:
            return np.full(self.T, np.nan), np.full(self.T, np.nan)
        sc = np.concatenate([E[i][0] for i in idx])
        tp = np.concatenate([E[i][1] for i in idx], axis=1)
        ig = np.concatenate([E[i][2] for i in idx], axis=1)
        npig = int(sum(E[i][3] for i in idx))
        return _accumulate(sc, tp, ig, npig)

    # -------------------------------------------------------------------------- metrik AP
    def ap_metrics(self, idx=None, area: str = "all", classes=CAT_IDS) -> dict:
        idx = np.arange(self.I) if idx is None else np.asarray(idx, dtype=int)
        a = AREA_LABELS.index(area)
        out, ap50s, aps = {}, [], []
        for cid in classes:
            ap, rec = self._ap_rec(a, CAT_IDS.index(cid), idx)
            n = CLASS_IDS[cid]
            out[f"ap50_{n}"], out[f"ap_{n}"], out[f"recall50_{n}"] = float(ap[0]), _nanmean(ap), float(rec[0])
            ap50s.append(ap[0])
            aps.append(_nanmean(ap))
        out["map50"], out["map50_95"] = _nanmean(ap50s), _nanmean(aps)
        return out

    # ----------------------------------------------------------------------- metrik hitung
    def npred(self, cat: int, thr: float, idx=None) -> np.ndarray:
        """Jumlah prediksi kelas `cat` dengan skor >= thr per citra (urutan citra = gt.img_ids)."""
        t = self.table
        m = (t.cat.to_numpy() == cat) & (t.score.to_numpy() >= thr)
        n = np.bincount(t.img.to_numpy()[m], minlength=self.I)
        return n if idx is None else n[np.asarray(idx, dtype=int)]

    def npred_matrix(self, cat: int, grid) -> np.ndarray:
        """[G, I]: jumlah prediksi per citra untuk tiap ambang pada grid."""
        t = self.table
        c = t.cat.to_numpy() == cat
        sc, im = t.score.to_numpy()[c], t.img.to_numpy()[c]
        return np.stack([np.bincount(im[sc >= g], minlength=self.I) for g in grid])

    def count_metrics(self, thr, idx=None) -> dict:
        """MAE hitung per citra (|prediksi - GT|) per kelas; thr bisa skalar atau {kelas: ambang}."""
        out = {}
        for cid in CAT_IDS:
            th = thr[cid] if isinstance(thr, dict) else thr
            ngt = self.gt.n_gt[cid] if idx is None else self.gt.n_gt[cid][np.asarray(idx, dtype=int)]
            npd = self.npred(cid, th, idx)
            n = CLASS_IDS[cid]
            if len(ngt) == 0:
                out[f"mae_{n}"] = out[f"bias_{n}"] = out[f"exact_{n}"] = float("nan")
                continue
            err = np.abs(npd - ngt)
            pos = ngt > 0
            out[f"mae_{n}"] = float(np.mean(err))
            out[f"bias_{n}"] = float(np.mean(npd - ngt))           # >0: kelebihan prediksi (FP > FN)
            out[f"exact_{n}"] = float(np.mean(npd == ngt))         # proporsi citra dengan hitungan tepat
            # MAPE per citra TIDAK terdefinisi bila GT = 0: dihitung hanya pada citra dengan GT > 0.
            out[f"mape_{n}"] = float(np.mean(err[pos] / ngt[pos]) * 100) if pos.any() else float("nan")
            out[f"n_img_gt_{n}"] = int(pos.sum())                  # jumlah citra yang masuk MAPE
            # WAPE = sum|galat| / sum GT (agregat; tetap terdefinisi walau banyak citra tanpa objek)
            out[f"wape_{n}"] = float(err.sum() / ngt.sum() * 100) if ngt.sum() > 0 else float("nan")
        return out

    def recall_img(self, cat: int, thr: float, idx=None) -> tuple[float, float]:
        """Recall per citra (TP/GT pada IoU 0.5, skor >= thr) dirata-ratakan atas citra dengan GT > 0, dan
        proporsi citra yang SEMUA objeknya ditemukan. (nan, nan) bila tak ada citra dengan GT > 0."""
        sc, tp50, img = self._flat[cat]
        m = sc >= thr
        tpc = np.bincount(img[m][tp50[m]], minlength=self.I)
        ids = np.arange(self.I) if idx is None else np.asarray(idx, dtype=int)
        ng, tp = self.gt.n_gt[cat][ids], tpc[ids]
        pos = ng > 0
        if not pos.any():
            return float("nan"), float("nan")
        return float(np.mean(tp[pos] / ng[pos])), float(np.mean(tp[pos] == ng[pos]))

    def prf(self, cat: int, thr: float, idx=None) -> dict:
        """Presisi/recall/F1 pada IoU 0.5 untuk prediksi dengan skor >= thr (dari pencocokan COCO)."""
        sc, tp50, img = self._flat[cat]
        w = np.ones(self.I) if idx is None else np.bincount(np.asarray(idx, dtype=int), minlength=self.I).astype(float)
        m = sc >= thr
        wt = w[img[m]]
        tp = float(np.sum(wt * tp50[m]))
        fp = float(np.sum(wt * ~tp50[m]))
        ngt = float(np.sum(w * self.gt.n_gt[cat]))
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / ngt if ngt > 0 else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and (prec + rec) > 0) else float("nan")
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "n_gt": ngt}

    def summary(self, thr, idx=None) -> dict:
        """Satu baris ringkas: AP total/per kelas, AP platelet kecil (COCO small), galat hitung, P/R platelet."""
        out = {"n_citra": self.I if idx is None else int(len(idx)), "thr": thr if not isinstance(thr, dict) else json.dumps(thr)}
        out.update(self.ap_metrics(idx))
        sm = self.ap_metrics(idx, area="small", classes=[PLATELET_ID])
        out["ap_small_Platelets_cocoSmall"] = sm["ap_Platelets"]     # small COCO = area < 32^2 px pada citra asli
        for ar in ("small", "medium", "large"):
            am = self.ap_metrics(idx, area=ar)                       # AP[.5:.95] rerata kelas yang punya GT di rentang itu
            out[f"AP_{ar}_semua_kelas"] = am["map50_95"]
            out[f"AP_{ar}_Platelets"] = am["ap_Platelets"]
        out.update(self.count_metrics(thr, idx))
        th = thr[PLATELET_ID] if isinstance(thr, dict) else thr
        rm, rf = self.recall_img(PLATELET_ID, th, idx)
        out["recall_img_Platelets"], out["recall_img_penuh_Platelets"] = rm, rf
        pr = self.prf(PLATELET_ID, th, idx)
        out.update({"prec_Platelets": pr["precision"], "rec_Platelets": pr["recall"], "f1_Platelets": pr["f1"],
                    "n_pred_Platelets": int(self.npred(PLATELET_ID, th, idx).sum()),
                    "n_gt_Platelets": int((self.gt.n_gt[PLATELET_ID] if idx is None else self.gt.n_gt[PLATELET_ID][np.asarray(idx, dtype=int)]).sum()),
                    "mae_Platelets_selalu0": always_zero_mae(self.gt, PLATELET_ID, idx)})
        return out

    def per_source(self, thr, min_gt_platelet: int = 10) -> pd.DataFrame:
        """Hasil per sumber citra (= ukuran citra). `andal=False` bila platelet GT di sumber itu < min_gt_platelet."""
        rows = []
        th = thr[PLATELET_ID] if isinstance(thr, dict) else thr
        for s in sorted(set(self.gt.src)):
            idx = np.flatnonzero(self.gt.src == s)
            r = {"sumber": s, **self.summary(thr, idx)}
            ngt = self.gt.n_gt[PLATELET_ID][idx]
            npd = self.npred(PLATELET_ID, th, idx)
            r["n_pred_Platelets_di_citra_tanpa_GT_platelet"] = int(npd[ngt == 0].sum())
            r["andal"] = bool(r["n_gt_Platelets"] >= min_gt_platelet)
            rows.append(r)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- bootstrap
_BOOT_DEFAULT = ("ap50_Platelets", "ap_Platelets", "ap_small_Platelets", "mae_Platelets", "mape_Platelets",
                 "wape_Platelets", "recall_img_Platelets", "map50", "map50_95")


def _boot_draws(gt: GT, B: int, seed: int, strata: str | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    I = len(gt)
    if strata is None:
        return rng.integers(0, I, size=(B, I))
    if strata != "source":
        raise ValueError("strata harus None atau 'source'")
    groups = [np.flatnonzero(gt.src == s) for s in sorted(set(gt.src))]
    return np.stack([np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups]) for _ in range(B)])


def _one_replicate(me: ModelEval, idx, thr, names) -> dict:
    """Nama metrik yang didukung: map50, map50_95, ap50_/ap_/recall50_<Kelas>, ap_small_Platelets,
    mae_/mape_/wape_<Kelas>, recall_img_Platelets."""
    out = {}
    ap_names = [n for n in names if n.split("_")[0] in ("map50", "map50", "ap50", "recall50") or n in ("map50_95",)
                or (n.startswith("ap_") and not n.startswith("ap_small"))]
    if ap_names:
        need_all = any(n in ("map50", "map50_95") for n in ap_names)
        ap = me.ap_metrics(idx, classes=CAT_IDS if need_all else [PLATELET_ID])
        for n in ap_names:
            out[n] = ap[n]
    for n in names:
        if n == "ap_small_Platelets":
            out[n] = me.ap_metrics(idx, area="small", classes=[PLATELET_ID])["ap_Platelets"]
        elif n == "recall_img_Platelets":
            th = thr[PLATELET_ID] if isinstance(thr, dict) else thr
            out[n] = me.recall_img(PLATELET_ID, th, idx)[0]
        elif n.split("_")[0] in ("mae", "mape", "wape"):
            kind, cname = n.split("_", 1)
            cid = next(c for c, nm in CLASS_IDS.items() if nm == cname)
            th = thr[cid] if isinstance(thr, dict) else thr
            cm = me.count_metrics({c: (th if c == cid else 0.5) for c in CAT_IDS}, idx)
            out[n] = cm[n]
        elif n not in out:
            raise ValueError(f"metrik bootstrap tidak dikenal: {n}")
    return out


def bootstrap(models: dict, gt: GT, thr, B: int = 1000, seed: int = 2024, strata: str | None = None,
              metrics=_BOOT_DEFAULT, alpha: float = 0.05, pairs: bool = True):
    """Bootstrap persentil atas CITRA (dengan pengembalian). `models`: {nama: ModelEval | [ModelEval, ...]}.
    Bila satu model punya beberapa seed, metrik tiap resampel = rerata antar seed (CI untuk rerata seed,
    hanya mengukur ketidakpastian pemilihan citra uji). Citra yang sama dipakai semua model pada tiap resampel,
    sehingga selisih antar model (berpasangan) valid.

    Mengembalikan (ringkasan: DataFrame, sampel: DataFrame [B x (model|metrik)])."""
    models = {k: (v if isinstance(v, (list, tuple)) else [v]) for k, v in models.items()}
    metrics = tuple(metrics)
    # thr: skalar, {id_kelas: ambang}, ATAU {nama_model: ambang_atau_dict_kelas} (ambang beda per model)
    per_model = isinstance(thr, dict) and len(thr) > 0 and all(isinstance(k, str) for k in thr)
    if per_model:
        missing = set(models) - set(thr)
        if missing:
            raise ValueError(f"ambang tidak ada untuk model: {missing}")
    thr_of = (lambda k: thr[k]) if per_model else (lambda k: thr)
    I = len(gt)
    draws = _boot_draws(gt, B, seed, strata)
    full = np.arange(I)

    def agg(mlist, idx, k):
        reps = [_one_replicate(me, idx, thr_of(k), metrics) for me in mlist]
        return {n: _nanmean([r[n] for r in reps]) if not all(math.isnan(r[n]) for r in reps) else float("nan") for n in metrics}

    point = {k: agg(v, full, k) for k, v in models.items()}
    samples = {f"{k}|{n}": np.full(B, np.nan) for k in models for n in metrics}
    for b in range(B):
        idx = draws[b]
        for k, v in models.items():
            r = agg(v, idx, k)
            for n in metrics:
                samples[f"{k}|{n}"][b] = r[n]
    samp = pd.DataFrame(samples)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []
    for k in models:
        for n in metrics:
            x = samp[f"{k}|{n}"].to_numpy()
            xv = x[~np.isnan(x)]
            rows.append({"model": k, "metrik": n, "n_seed": len(models[k]), "titik": point[k][n],
                         "boot_mean": float(xv.mean()) if xv.size else float("nan"),
                         "ci_lo": float(np.percentile(xv, lo_q)) if xv.size else float("nan"),
                         "ci_hi": float(np.percentile(xv, hi_q)) if xv.size else float("nan"),
                         "n_valid": int(xv.size), "B": B})
    if pairs and len(models) > 1:
        for ka, kb in combinations(models, 2):
            for n in metrics:
                d = (samp[f"{ka}|{n}"] - samp[f"{kb}|{n}"]).to_numpy()
                dv = d[~np.isnan(d)]
                rows.append({"model": f"{ka} - {kb}", "metrik": n, "n_seed": np.nan,
                             "titik": point[ka][n] - point[kb][n],
                             "boot_mean": float(dv.mean()) if dv.size else float("nan"),
                             "ci_lo": float(np.percentile(dv, lo_q)) if dv.size else float("nan"),
                             "ci_hi": float(np.percentile(dv, hi_q)) if dv.size else float("nan"),
                             "n_valid": int(dv.size), "B": B,
                             "frac_selisih_positif": float(np.mean(dv > 0)) if dv.size else float("nan")})
    return pd.DataFrame(rows), samp


# ------------------------------------------------------------------- ablasi ambang confidence
def default_grid() -> list[float]:
    """Grid ambang (ditetapkan SEBELUM melihat kurva): 0.01, 0.02, 0.05, lalu 0.10 s.d. 0.95 langkah 0.05."""
    g = [0.01, 0.02, 0.05] + [round(float(x), 2) for x in np.arange(0.10, 0.951, 0.05)]
    return sorted(set(g))


def threshold_curve(seed_evals: list[ModelEval], grid=None) -> pd.DataFrame:
    """Kurva galat hitung vs ambang (SATU ambang global untuk semua kelas), rerata antar seed.
    AP/mAP tidak bergantung pada ambang (dihitung dari seluruh peringkat skor), jadi tidak ada di sini."""
    grid = default_grid() if grid is None else list(grid)
    rows = []
    for g in grid:
        per_seed = [me.count_metrics(g) for me in seed_evals]
        prf = [me.prf(PLATELET_ID, g) for me in seed_evals]
        r = {"thr": g}
        for n in CLASS_IDS.values():
            r[f"mae_{n}"] = _nanmean([p[f"mae_{n}"] for p in per_seed])
            r[f"bias_{n}"] = _nanmean([p[f"bias_{n}"] for p in per_seed])
        r["mae_Platelets_sd_antar_seed"] = float(np.std([p["mae_Platelets"] for p in per_seed], ddof=1)) if len(per_seed) > 1 else float("nan")
        r["prec_Platelets"] = _nanmean([p["precision"] for p in prf])
        r["rec_Platelets"] = _nanmean([p["recall"] for p in prf])
        r["f1_Platelets"] = _nanmean([p["f1"] for p in prf])
        r["n_pred_Platelets"] = float(np.mean([me.npred(PLATELET_ID, g).sum() for me in seed_evals]))
        rows.append(r)
    return pd.DataFrame(rows)


def select_threshold(seed_evals: list[ModelEval], grid=None, B: int = 1000, seed: int = 0,
                     cat: int = PLATELET_ID, strata: str | None = None) -> dict:
    """Pilih ambang pada VAL dengan aturan yang ditetapkan di muka ('plateau pusat, 1-SE'):
      1. MAE_hitung(thr) kelas `cat` = rerata antar seed dari rerata |prediksi - GT| per citra val.
      2. SE = simpangan baku MAE di ambang minimum, dari bootstrap atas citra val.
      3. Plateau = rentang ambang bersebelahan dengan MAE <= MAE_min + SE yang memuat ambang minimum.
      4. thr* = ambang di TENGAH plateau (median indeks; meredam derau pilihan argmin).
    Juga dilaporkan argmin murni dan frekuensi argmin pada resampel (stabilitas pilihan)."""
    grid = default_grid() if grid is None else list(grid)
    gt = seed_evals[0].gt
    ngt = gt.n_gt[cat]
    err = np.mean([np.abs(me.npred_matrix(cat, grid) - ngt[None, :]) for me in seed_evals], axis=0)   # [G, I]
    mae = err.mean(axis=1)
    mins = np.flatnonzero(mae <= mae.min() + 1e-12)
    i_min = int(mins[(len(mins) - 1) // 2])
    draws = _boot_draws(gt, B, seed, strata)
    boot = np.stack([err[:, d].mean(axis=1) for d in draws])                  # [B, G]
    se = float(boot[:, i_min].std(ddof=1)) if B > 1 else float("nan")
    ok = mae <= mae[i_min] + se
    lo = hi = i_min
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    while hi + 1 < len(grid) and ok[hi + 1]:
        hi += 1
    run = list(range(lo, hi + 1))
    i_star = run[(len(run) - 1) // 2]
    argmins = boot.argmin(axis=1)
    freq = {grid[g]: float(np.mean(argmins == g)) for g in range(len(grid)) if np.any(argmins == g)}
    return {"rule": "plateau_pusat_1SE_pada_val", "cat": cat, "grid": grid, "n_seed": len(seed_evals),
            "thr_star": float(grid[i_star]), "thr_argmin": float(grid[i_min]),
            "plateau": [float(grid[g]) for g in run], "se_at_argmin": se,
            "mae_val": [float(x) for x in mae],
            "ci_lo": [float(x) for x in np.percentile(boot, 2.5, axis=0)],
            "ci_hi": [float(x) for x in np.percentile(boot, 97.5, axis=0)],
            "mae_val_at_thr_star": float(mae[i_star]), "mae_val_selalu0": always_zero_mae(gt, cat),
            "argmin_freq_bootstrap": freq, "B": B, "seed": seed, "strata": strata}


def freeze_thresholds(path, key: str, result: dict, meta: dict | None = None, force: bool = False) -> dict:
    """Bekukan ambang terpilih (dari val) ke JSON. Menolak menimpa kunci yang sudah ada kecuali force=True.
    Berkas ini adalah satu-satunya sumber ambang untuk evaluasi test."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    if key in data and not force:
        raise FileExistsError(f"ambang '{key}' sudah dibekukan di {path}; gunakan force=True hanya bila tim setuju")
    data[key] = {"thr": result["thr_star"], "thr_argmin": result["thr_argmin"], "rule": result["rule"],
                 "plateau": result["plateau"], "grid": result["grid"], "cat": result["cat"],
                 "n_seed": result["n_seed"], "mae_val_at_thr": result["mae_val_at_thr_star"],
                 "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(meta or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data[key]


def load_frozen(path, key: str) -> float:
    data = json.loads(Path(path).read_text())
    if key not in data:
        raise KeyError(f"ambang '{key}' belum dibekukan di {path}. Jalankan ablasi di VAL dulu.")
    return float(data[key]["thr"])


# ------------------------------------------------------------- pencatatan hasil akhir (test)
def append_final_test_row(csv_path, row: dict) -> None:
    """Tambah satu baris ke logs/final_test_results.csv. Menolak duplikat (run_id, test_split) agar test
    benar-benar dipakai sekali per konfigurasi. Aman terhadap berkas yang barisnya tidak diakhiri newline."""
    csv_path = Path(csv_path)
    unknown = set(row) - set(FINAL_TEST_COLS)
    assert not unknown, f"kolom tidak dikenal: {unknown}"
    row = {c: row.get(c, "") for c in FINAL_TEST_COLS}
    if csv_path.exists() and csv_path.stat().st_size > 0:
        raw = csv_path.read_bytes().decode("utf-8")
        header = next(csv.reader(io.StringIO(raw)))
        assert header == FINAL_TEST_COLS, f"header berbeda dari yang diharapkan: {header}"
        for r in csv.DictReader(io.StringIO(raw)):
            if r["run_id"] == row["run_id"] and r["test_split"] == row["test_split"]:
                raise ValueError(f"({row['run_id']}, {row['test_split']}) sudah ada di {csv_path}: test hanya sekali per konfigurasi")
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            if not raw.endswith("\n"):
                f.write("\n")
            csv.DictWriter(f, fieldnames=FINAL_TEST_COLS, lineterminator="\n").writerow(row)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FINAL_TEST_COLS, lineterminator="\n")
            w.writeheader()
            w.writerow(row)


def final_test_already_done(csv_path, run_id: str, test_split: str) -> bool:
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p, newline="", encoding="utf-8") as f:
        return any(r["run_id"] == run_id and r["test_split"] == test_split for r in csv.DictReader(f))


# ----------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Evaluasi bersama; prediksi dalam format COCO (lihat docstring modul).")
    ap.add_argument("--gt", required=True, help="mis. data/processed/coco_clean/val.json")
    ap.add_argument("--pred", nargs="+", required=True, help="satu/lebih berkas prediksi JSON")
    ap.add_argument("--names", default=None, help="nama model dipisah koma, urut sesuai --pred")
    ap.add_argument("--thr", type=float, default=0.5, help="ambang confidence untuk galat hitung dan P/R")
    ap.add_argument("--bootstrap", type=int, default=1000, help="jumlah resampel (0 = lewati)")
    ap.add_argument("--no-dedup", action="store_true", help="jangan buang baris GT duplikat persis")
    ap.add_argument("--out", default=None, help="folder keluaran CSV")
    ap.add_argument("--final", action="store_true", help="WAJIB untuk split test/test_unflagged (penggunaan sekali)")
    a = ap.parse_args()

    stem = Path(a.gt).stem
    if stem.startswith("test") and not a.final:
        raise SystemExit("Split test/test_unflagged hanya untuk konfigurasi final: tambahkan --final bila memang itu.")
    gt = GT(a.gt, dedup=not a.no_dedup)
    print(f"GT {gt.name}: {len(gt)} citra; duplikat dibuang: {dict(gt.n_dup_by_cat)}; per kelas: "
          f"{ {CLASS_IDS[c]: int(gt.n_gt[c].sum()) for c in CAT_IDS} }")
    print(gt.summary().to_string(index=False))
    names = a.names.split(",") if a.names else [Path(p).stem for p in a.pred]
    assert len(names) == len(a.pred), "jumlah --names harus sama dengan --pred"
    evals = {n: ModelEval(gt, json.loads(Path(p).read_text()), name=n) for n, p in zip(names, a.pred)}
    summ = pd.DataFrame([{"model": n, **me.summary(a.thr)} for n, me in evals.items()])
    print(summ.round(4).T.to_string())
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out / "summary.csv", index=False)
    for n, me in evals.items():
        ps = me.per_source(a.thr)
        print(f"\nPer sumber citra: {n}")
        print(ps[["sumber", "n_citra", "n_gt_Platelets", "ap50_Platelets", "mae_Platelets", "mae_Platelets_selalu0",
                  "n_pred_Platelets_di_citra_tanpa_GT_platelet", "andal"]].round(4).to_string(index=False))
        if out:
            ps.to_csv(out / f"per_source_{n}.csv", index=False)
    if a.bootstrap > 0:
        res, _ = bootstrap(evals, gt, a.thr, B=a.bootstrap)
        print("\nBootstrap atas citra (CI 95% persentil):")
        print(res.round(4).to_string(index=False))
        if out:
            res.to_csv(out / "bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
