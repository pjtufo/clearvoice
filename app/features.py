"""离线音频特征分析：MFCC 说话人/相似度特征、哔哔声检测、破音检测、静音检测。

全部基于 numpy/scipy 的本地 DSP，无需联网，不依赖大模型。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-10


# ---------------------------------------------------------------- MFCC 特征

def _mel_filterbank(n_filters: int, n_fft: int, sr: int, fmin: float = 50.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    def hz2mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_filters + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c == l: c = l + 1
        if r == c: r = c + 1
        fb[i, l:c] = np.linspace(0, 1, c - l)
        fb[i, c:r] = np.linspace(1, 0, r - c)
    return fb


def logmel_sequence(y: np.ndarray, sr: int, frame: float = 0.5, hop: float = 0.25,
                    n_mels: int = 40) -> np.ndarray:
    """逐帧对数梅尔谱 (n_frames, n_mels)，用作说话人/相似度指纹。"""
    fl = int(frame * sr); hp = int(hop * sr)
    if len(y) < fl:
        fl = len(y)
    frames_n = 1 + max(0, (len(y) - fl) // hp)
    idx = np.arange(fl)[None, :] + hp * np.arange(frames_n)[:, None]
    frames = y[np.minimum(idx, len(y) - 1)] * np.hamming(fl)
    n_fft = 1 << int(np.ceil(np.log2(fl)))
    spec = np.abs(np.fft.rfft(frames, n_fft, axis=1)) ** 2
    fb = _mel_filterbank(n_mels, n_fft, sr)
    return np.log(fb @ spec.T + 1e-8).T


def similarity_curve(y: np.ndarray, sr: int, ref_audio: np.ndarray, ref_sr: int,
                     hop: float = 0.25, win: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """计算 y 每个滑窗与参考音频的相似度曲线。

    归一化统一使用目标 y 的各频点时间均值（CMN），保证参考与目标在同一特征空间。
    返回 (times秒, sims)。
    """
    m = logmel_sequence(y, sr, frame=win, hop=hop)
    if m.shape[0] == 0:
        return np.array([]), np.array([])
    mu = m.mean(axis=0, keepdims=True)               # y 的频点均值
    e = m.max(axis=1)
    m2 = m - mu

    # 参考窗长：参考太短时缩短窗长，保证 ≥4 个窗口（避免均值退化）
    ref_dur = len(ref_audio) / max(1, ref_sr)
    win_ref = win if ref_dur >= win * 3 else max(0.12, min(win, ref_dur / 4))
    ref_m = logmel_sequence(ref_audio, ref_sr, frame=win_ref, hop=max(0.05, win_ref / 2))
    if ref_m.shape[0] == 0:
        return np.arange(m.shape[0]) * hop, np.full(m.shape[0], -1.0)
    ref_e = ref_m.max(axis=1)
    thr = np.percentile(ref_e, 40)
    sel = ref_m[ref_e >= thr]
    sel = sel if len(sel) else ref_m
    ref_emb = (sel - mu).mean(axis=0)                # 与 y 同空间

    norms = np.linalg.norm(m2, axis=1) + EPS
    sims = (m2 @ ref_emb) / (norms * (np.linalg.norm(ref_emb) + EPS))
    if e.max() > 0:
        sims[e < e.max() - 6.0] = -1.0               # 静音帧不计
    return np.arange(len(sims)) * hop, sims


def find_similar_segments(y: np.ndarray, sr: int, ref_audio: np.ndarray, ref_sr: int,
                          hop: float = 0.25, win: float = 0.5,
                          threshold: float = 0.78, min_len: float = 0.3) -> list[tuple[float, float]]:
    """在 y 中查找与 ref_audio 相似的片段。返回 [(start,end)...] 秒。"""
    times, sims = similarity_curve(y, sr, ref_audio, ref_sr, hop=hop, win=win)
    if len(sims) == 0:
        return []
    hits = sims >= threshold
    segs: list[tuple[float, float]] = []
    i = 0
    while i < len(hits):
        if hits[i]:
            j = i
            while j + 1 < len(hits) and hits[j + 1]:
                j += 1
            if (j - i + 1) * hop >= min_len:
                segs.append((i * hop, (j + 1) * hop + win))
            i = j + 1
        else:
            i += 1
    return segs


# ---------------------------------------------------------------- 哔哔声检测

def detect_beep_segments(y: np.ndarray, sr: int, min_dur: float = 0.4,
                         fmin: float = 500, fmax: float = 6000) -> tuple[list[tuple[float, float]], list[int]]:
    """检测持续性窄带音（哔哔/嘀嘀）。返回 (时间段, 频点bin列表)。

    原理：STFT 每个频点随时间的方差极小且能量显著 => 稳定纯音。
    """
    f, t, Z = signal.stft(y, sr, nperseg=4096, noverlap=3072)
    mag = np.abs(Z)
    band = (f >= fmin) & (f <= fmax)
    mag_b = mag[band]; f_b = f[band]
    if mag_b.size == 0:
        return [], []
    med = np.median(mag_b, axis=1) + EPS
    # 峰值显著度：某频点能量相对邻域频点
    k = np.maximum(med / np.convolve(med, np.ones(21) / 21, mode="same") - 1.0, 0)
    cand = np.where(k > np.percentile(k, 97))[0]
    time_on: np.ndarray | None = None
    bins: list[int] = []
    dur_total = t[-1] if len(t) else 1
    for i in cand:
        col = mag_b[i] / (np.median(mag_b, axis=0) + EPS)
        onset = col > 4.0
        # 至少持续 min_dur
        run = onset.astype(int)
        ok = False
        cnt = 0
        for v in run:
            cnt = cnt + 1 if v else 0
            if cnt * (t[1] - t[0] if len(t) > 1 else 1) >= min_dur:
                ok = True
                break
        if ok:
            bins.append(i)
            time_on = onset if time_on is None else (time_on | onset)
    segs: list[tuple[float, float]] = []
    if time_on is not None and len(t) > 1:
        dt = t[1] - t[0]
        i = 0
        while i < len(time_on):
            if time_on[i]:
                j = i
                while j + 1 < len(time_on) and time_on[j + 1]:
                    j += 1
                if (j - i) * dt >= min_dur:
                    segs.append((float(t[i]), float(t[min(j + 1, len(t) - 1)])))
                i = j + 1
            else:
                i += 1
    return segs, bins


def remove_beeps(y: np.ndarray, sr: int) -> np.ndarray:
    """对检测到的哔哔频点做时频掩蔽。"""
    if len(y) < 8192:
        return y
    segs, bins = detect_beep_segments(y, sr)
    if not bins:
        return y
    f, t, Z = signal.stft(y, sr, nperseg=4096, noverlap=3072)
    mask = np.ones_like(np.abs(Z))
    if segs:
        for s, e in segs:
            ts = (t >= s - 0.05) & (t <= e + 0.05)
            mask[:, ts] = 0.05
    else:
        for i in bins:
            mask[i - 1:i + 2, :] = 0.02
    _, rec = signal.istft(Z * mask, sr, nperseg=4096, noverlap=3072)
    out = rec[: len(y)]
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------- 破音修复

def detect_clipped_ratio(y: np.ndarray, thresh: float = 0.985) -> float:
    return float(np.mean(np.abs(y) >= thresh))


def declip(y: np.ndarray, thresh: float = 0.985) -> np.ndarray:
    """修复削波破音：识别连续削波段并用边界样条插值重建。"""
    x = y.copy()
    clip = np.abs(x) >= thresh
    if not clip.any():
        return x
    # 找连续 clip run（含前后各1样本）
    d = np.diff(clip.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if clip[0]: starts = np.r_[0, starts]
    if clip[-1]: ends = np.r_[ends, len(x)]
    for s, e in zip(starts, ends):
        if e - s > max(64, len(x) // 20):
            continue  # 太长的不处理
        a, b = max(0, s - 2), min(len(x), e + 2)
        if b - a < 4:
            continue
        xs = np.arange(a, b)
        seg = x[a:b]
        good = np.abs(seg) < thresh
        if good.sum() < 2:
            continue
        # 线性+样条混合重建
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(xs[good], seg[good])
            x[s:e] = cs(np.arange(s, e))
        except Exception:
            x[s:e] = np.interp(np.arange(s, e), xs[good], seg[good])
    return np.clip(x, -1, 1)


# ---------------------------------------------------------------- 静音/能量

def detect_silence(y: np.ndarray, sr: int, top_db: float = 35, min_len: float = 0.3) -> list[tuple[float, float]]:
    """基于能量阈值的静音段检测（用于按特征分割）。"""
    fl = int(0.05 * sr); hp = fl // 2
    frames_n = 1 + max(0, (len(y) - fl) // hp)
    idx = np.arange(fl)[None, :] + hp * np.arange(frames_n)[:, None]
    rms = np.sqrt(np.mean(y[np.minimum(idx, len(y) - 1)] ** 2, axis=1))
    db = 20 * np.log10(rms + EPS)
    ref = db.max()
    quiet = db < (ref - top_db)
    segs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j + 1 < len(quiet) and quiet[j + 1]:
                j += 1
            if (j - i) * (hp / sr) >= min_len:
                segs.append((i * hp / sr, (j + 1) * hp / sr))
            i = j + 1
        else:
            i += 1
    return segs


def merge_overlapping(ranges: list[tuple[float, float]], gap: float = 0.05) -> list[tuple[float, float]]:
    """合并重叠/相邻的特征区间。"""
    if not ranges:
        return []
    rs = sorted(ranges)
    out = [list(rs[0])]
    for s, e in rs[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(max(0, s), e) for s, e in out]


def segment_boundaries_from_ranges(ranges: list[tuple[float, float]], total: float) -> list[float]:
    """由特征区间生成分割点（每个区间首尾作为切点）。"""
    pts = {0.0, total}
    for s, e in ranges:
        pts.add(max(0, s)); pts.add(min(total, e))
    return sorted(pts)
