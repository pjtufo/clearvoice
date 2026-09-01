"""人声/伴奏分离（去背景音乐）：torchaudio 内置 HDemucs。

权重为 MUSDB18-HQ 高质量版（4 音轨：drums/bass/other/vocals），首次运行自动下载到
~/.cache/torch/hub/torchaudio/models/，之后完全离线。也可在「设置」页指定权重目录
（放入 hdemucs_high*.pt 即可，例如从魔搭 Demucs-Repackage 转换后的权重）。
"""
from __future__ import annotations

import glob
import os

SAMPLE_RATE = 44100
SOURCES = ("drums", "bass", "other", "vocals")
VOCALS_IDX = SOURCES.index("vocals")

_SEGMENT_S = 10.0    # 分块推理窗长（控制内存）
_OVERLAP = 0.25      # 窗间交叠比例


def available() -> bool:
    try:
        import torchaudio  # noqa: F401
        return True
    except ImportError:
        return False


def _candidate_dirs() -> list[str]:
    from . import config
    dirs = []
    d = config.load_config().get("demucs_model_dir", "")
    if d:
        dirs.append(d)
    dirs.append(os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "torchaudio", "models"))
    dirs.append(os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints"))
    return dirs


def weights_path() -> str | None:
    """已存在的权重文件（配置目录优先，其次 torch 缓存）。"""
    for d in _candidate_dirs():
        if not os.path.isdir(d):
            continue
        for pat in ("hdemucs_high*.pt", "*.pt", "*.pth"):
            hits = sorted(glob.glob(os.path.join(d, pat)))
            if hits:
                return hits[0]
    return None


def get_model():
    """加载 HDemucs（权重缓存命中则离线，否则 torchaudio 自动下载）。"""
    import torchaudio
    pipeline = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = pipeline.get_model()
    model.eval()
    return model, pipeline.sample_rate


def _chunked_separate(model, wav, progress_cb=None):
    """分块 + 线性交叠融合推理。wav: (2, n)@44100 → (4, 2, n) 四音轨。"""
    import torch
    seg = int(_SEGMENT_S * SAMPLE_RATE)
    hop = int(seg * (1 - _OVERLAP))
    edge = seg - hop
    n = wav.shape[1]
    if n <= seg:
        with torch.no_grad():
            return model(wav[None])[0]
    ramp = torch.linspace(0.0, 1.0, edge)
    starts = list(range(0, n - seg + 1, hop))
    if starts[-1] != n - seg:
        starts.append(n - seg)
    out = torch.zeros(4, 2, n)
    wsum = torch.zeros(1, 1, n)
    total = len(starts)
    for k, s in enumerate(starts):
        e = s + seg
        with torch.no_grad():
            est = model(wav[None, :, s:e])[0]          # (4, 2, seg)
        w = torch.ones(seg)
        if k > 0:
            w[:edge] = ramp
        if k < total - 1:
            w[-edge:] = 1.0 - ramp
        ww = w[None, None, :]
        out[:, :, s:e] += est * ww
        wsum[:, :, s:e] += ww
        if progress_cb:
            progress_cb(int(100 * (k + 1) / total))
    return out / wsum.clamp_min(1e-8)


def separate(wav_in: str, out_vocals: str, out_acc: str | None = None,
             keep_ratio: float = 0.0, progress_cb=None) -> tuple[str, str | None]:
    """分离人声/伴奏。输出 = 人声 + keep_ratio*伴奏；返回 (人声文件, 伴奏文件|None)。

    读写使用 soundfile（torchaudio 2.9+ 的 load/save 需要 torchcodec，不依赖）。
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    progress_cb(3, "加载分离模型（首次运行自动下载权重）…")
    model, sr = get_model()
    progress_cb(8, "解码音频…")
    data, orig = sf.read(wav_in, dtype="float32", always_2d=True)   # (n, ch)
    wav = torch.from_numpy(data.T)                                   # (ch, n)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if orig != sr:
        wav = torchaudio.functional.resample(wav, orig, sr)
    wav = wav.float()

    def cb(p):
        progress_cb(10 + int(80 * p / 100))

    progress_cb(10, "人声/伴奏分离推理中…")
    stems = _chunked_separate(model, wav, cb)
    vocals = stems[VOCALS_IDX]
    acc = stems[:3].sum(dim=0)
    result = vocals if keep_ratio <= 0 else vocals + keep_ratio * acc
    peak = result.abs().max()
    if peak > 1.0:
        result = result / peak
    sf.write(out_vocals, result.clamp(-1, 1).numpy().T, sr, subtype="PCM_16")
    if out_acc:
        peak = acc.abs().max()
        if peak > 1.0:
            acc = acc / peak
        sf.write(out_acc, acc.clamp(-1, 1).numpy().T, sr, subtype="PCM_16")
    progress_cb(95, "分离完成")
    return out_vocals, out_acc
