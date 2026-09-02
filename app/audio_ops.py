"""各类杂音消除的 DSP 实现（全部离线，本地处理）。

支持的消除类型（可多选、按 REMOVAL_TYPES 固定顺序执行）:
  noise      杂音/噪声         —— 谱减法谱门限（noisereduce 平稳模式）
  steady     背景音/沙沙声     —— 非平稳谱减 + 9kHz 低通 + 平滑
  babble     嘈杂声            —— 非平稳 + 平稳双重谱减
  beep       哔哔声/嘀嘀       —— STFT 频点显著性检测 + 时频掩蔽
  declip     破音              —— 削波段样条插值重建
  bgvoice    背景人声          —— 300~3400Hz 人声带谱减 + 中频压制
  speaker    指定说话人        —— log-mel 指纹（CMN 归一化）相似度匹配，命中段静音

入口：
  load_wav()   任意媒体 → 16k 单声道 float64 波形（经 ffmpeg 解码）
  save_wav()   波形写回 wav
  process_removals()  总管线：按所选类型依次消除，返回 (音频, 报告)
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from . import features

REMOVAL_TYPES = [
    ("noise", "杂音 / 噪声"),
    ("steady", "背景音 / 沙沙声"),
    ("babble", "嘈杂声"),
    ("beep", "哔哔声 / 嘀嘀"),
    ("declip", "破音"),
    ("bgvoice", "背景人声"),
    ("speaker", "指定说话人的说话声"),
]


def load_wav(path: str, sr: int = 16000, mono: bool = True) -> tuple[np.ndarray, int]:
    """任意音/视频解码为波形（临时 wav 经 ffmpeg 提取后读取并删除）。

    mono=True 返回 (N,) 单声道；mono=False 保留声道，立体声返回 (N,2)。
    """
    import subprocess, os
    from . import ffmpeg_tools as ft
    tmp = path + f"._{sr}.wav"
    ft.extract_audio(path, tmp, sr=sr, mono=mono)
    y, rate = sf.read(tmp, dtype="float32")
    try:
        os.remove(tmp)
    except OSError:
        pass
    if mono and y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float64), rate


def save_wav(path: str, y: np.ndarray, sr: int) -> str:
    """波形写回 wav（float32 PCM）。"""
    sf.write(path, y.astype(np.float32), sr)
    return path


# ---------------------------------------------------------------- 各算法

def reduce_noise(y: np.ndarray, sr: int, strength: float = 0.85, stationary: bool = True) -> np.ndarray:
    """谱减法降噪。stationary=True 适合稳态噪声，False 适合非平稳背景。"""
    import noisereduce as nr
    return nr.reduce_noise(y=y, sr=sr, stationary=stationary, prop_decrease=strength)


def reduce_steady_hiss(y: np.ndarray, sr: int, strength: float = 0.8) -> np.ndarray:
    """背景音/沙沙声：非平稳谱减 + 9kHz 低通（压制高频嘶声）+ 平滑。"""
    out = reduce_noise(y, sr, strength=strength, stationary=False)
    from scipy.signal import butter, sosfilt
    sos = butter(2, 9000, "lowpass", fs=sr, output="sos")
    out = sosfilt(sos, out)
    return out


def reduce_babble(y: np.ndarray, sr: int, strength: float = 0.9) -> np.ndarray:
    """嘈杂声：时频 mask 双重处理（对前一结果再减一次）。"""
    a = reduce_noise(y, sr, strength=strength, stationary=False)
    b = reduce_noise(a, sr, strength=strength * 0.7, stationary=True)
    return b


def suppress_bg_voice(y: np.ndarray, sr: int, strength: float = 0.75) -> np.ndarray:
    """背景人声近似消除：300~3400Hz 人声带内做谱减 + 中频动态压制。"""
    out = reduce_noise(y, sr, strength=strength, stationary=False)
    if len(out) < 4096:
        return out
    from scipy.signal import butter, sosfilt, stft, istft
    _, _, Z = stft(out, sr, nperseg=2048, noverlap=1536)
    mag = np.abs(Z)
    band = np.zeros_like(mag)
    f = np.linspace(0, sr / 2, mag.shape[0])
    band[(f >= 300) & (f <= 3400)] = 1.0
    med = np.median(mag, axis=1, keepdims=True)
    suppress = band * (med / (mag + 1e-10))
    Z2 = Z * (1 - strength * np.clip(suppress, 0, 1))
    _, rec = istft(Z2, sr, nperseg=2048, noverlap=1536)
    return np.asarray(rec[: len(out)])


def apply_speaker_mask(y: np.ndarray, sr: int, ref_audio: np.ndarray, ref_sr: int,
                       threshold: float = 0.80, mode: str = "mute") -> tuple[np.ndarray, list[tuple[float, float]]]:
    """把与参考说话人相似的语音段消除。

    特征：0.5s 窗 log-mel 谱 + CMN 归一化（features.find_similar_segments）。
    mode: mute=静音, attenuate=-18dB；命中段边缘各加 20ms 淡出防爆音。
    返回 (处理后音频, 命中段列表 [(起, 止)…])。
    """
    segs = features.find_similar_segments(y, sr, ref_audio, ref_sr, hop=0.25, win=0.5,
                                          threshold=threshold, min_len=0.3)
    out = y.copy()
    g = 0.0 if mode == "mute" else 0.125
    fade = int(0.02 * sr)
    for s, e in segs:
        i0, i1 = int(s * sr), min(len(out), int(e * sr))
        if i1 <= i0:
            continue
        out[i0:i1] = g
        for k in range(fade):
            if i0 - k - 1 >= 0:
                out[i0 - k - 1] *= (1 - (fade - k) / fade)
            if i1 + k < len(out):
                out[i1 + k] *= (1 - (fade - k) / fade)
    return out, segs


def apply_beep_removal(y: np.ndarray, sr: int) -> np.ndarray:
    """哔哔声消除（features.remove_beeps：STFT 频点显著性检测 + 时频掩蔽）。"""
    return features.remove_beeps(y, sr)


def apply_declip(y: np.ndarray, sr: int) -> np.ndarray:
    """破音修复（features.declip：|x|≥0.985 削波段样条插值重建）。"""
    return features.declip(y, 0.985)


# ---------------------------------------------------------------- 总管线

def process_removals(y: np.ndarray, sr: int, selected: list[str], strength: float = 0.85,
                     speaker_ref: np.ndarray | None = None, speaker_threshold: float = 0.80,
                     progress=None) -> tuple[np.ndarray, dict]:
    """总消除管线：按 REMOVAL_TYPES 固定顺序执行所选类型，每步后限幅 [-1,1]。

    selected: 类型 key 列表（见模块头）；strength: 降噪强度 0.1~1.0；
    speaker_ref: 说话人参考波形（未提供则跳过 speaker 并写入报告）；
    speaker_threshold: 相似度阈值；progress: (百分比, 文本) 回调。
    返回 (处理后音频, 报告 dict)。
    """
    report: dict = {}
    order = [t for t, _ in REMOVAL_TYPES if t in selected]
    total = max(1, len(order))
    for i, kind in enumerate(order):
        if progress:
            progress(int(i / total * 100), f"正在处理: {dict(REMOVAL_TYPES)[kind]}")
        if kind == "noise":
            y = reduce_noise(y, sr, strength=strength)
        elif kind == "steady":
            y = reduce_steady_hiss(y, sr, strength=strength)
        elif kind == "babble":
            y = reduce_babble(y, sr, strength=strength)
        elif kind == "beep":
            y = apply_beep_removal(y, sr)
        elif kind == "declip":
            y = apply_declip(y, sr)
        elif kind == "bgvoice":
            y = suppress_bg_voice(y, sr, strength=strength)
        elif kind == "speaker":
            if speaker_ref is None:
                report["speaker"] = "未设置说话人参考片段，已跳过"
                continue
            y, segs = apply_speaker_mask(y, sr, speaker_ref, sr, threshold=speaker_threshold)
            report["speaker"] = f"检测并消除 {len(segs)} 段说话人语音"
        y = np.clip(y, -1, 1)
    if progress:
        progress(100, "消除完成")
    return y, report
