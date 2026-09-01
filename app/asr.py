"""魔塔(ModelScope)小模型集成：语音识别（转写/导出/文字特征剔除）与语音增强。

模型优先从「设置」中配置的本地目录加载（完全离线）；
本地没有对应模型时才回退到模型 ID（funasr 自动下载到默认缓存）。
"""
from __future__ import annotations

import os
import re

ASR_MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
ENHANCE_MODEL_ID = "iic/speech_zipenhancer_ans_multiloss_16k_base"

# 标点/空白（用于文本清洗，保证关键词匹配不受标点影响）
_PUNC_RE = re.compile(r"[，。！？、；：…—·“”‘’（）《》〈〉【】,.!?;:()\[\]{}<>\"'\s]+")

_asr_pipeline = None
_enhance_pipeline = None


def _local(model_id: str) -> str | None:
    """在配置的模型根目录下查找本地模型，找到返回路径，否则 None。"""
    from . import config
    base = config.load_config().get("funasr_model_dir", "")
    if not base or not os.path.isdir(base):
        return None
    name = model_id.split("/")[-1]
    for cand in (os.path.join(base, name),
                 os.path.join(base, "hub", "iic", name),
                 os.path.join(base, "iic", name)):
        if os.path.isfile(os.path.join(cand, "config.yaml")):
            return cand
    return None


def asr_available() -> bool:
    try:
        import funasr  # noqa: F401
        return True
    except ImportError:
        return False


def enhance_available() -> bool:
    try:
        import modelscope  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def get_asr():
    """加载魔塔 Paraformer 语音识别（带时间戳、VAD 与标点）。

    优先使用设置页配置的本地模型目录（完全离线），否则用模型 ID 自动下载。
    """
    global _asr_pipeline
    if _asr_pipeline is not None:
        return _asr_pipeline
    if not asr_available():
        raise RuntimeError(
            "未安装魔塔语音识别依赖。请运行: uv sync --extra modelscope "
            f"(模型: {ASR_MODEL_ID})")
    from funasr import AutoModel
    from . import config
    cfg = config.load_config()
    kwargs = dict(
        model=_local(ASR_MODEL_ID) or ASR_MODEL_ID,
        vad_model=_local(VAD_MODEL_ID) or VAD_MODEL_ID,
        disable_update=True,
        device=cfg.get("asr_device", "cpu"))
    punc = _local(PUNC_MODEL_ID)
    if punc:
        kwargs["punc_model"] = punc
    try:
        _asr_pipeline = AutoModel(**kwargs)
    except Exception:
        if "punc_model" in kwargs:      # 标点模型异常时回退到无标点
            kwargs.pop("punc_model")
            _asr_pipeline = AutoModel(**kwargs)
        else:
            raise
    return _asr_pipeline


def reset_pipeline():
    """设置变更后清空已加载管线，下次识别时按新配置重新加载。"""
    global _asr_pipeline
    _asr_pipeline = None


def transcribe_with_timestamps(wav_path: str) -> list[dict]:
    """返回 [{'text':..., 'start':秒, 'end':秒}, ...]"""
    model = get_asr()
    res = model.generate(input=wav_path, batch_size_s=60)
    out: list[dict] = []
    for item in res:
        sent = item.get("sentence_info") or []
        if sent:
            cur = None
            for w in sent:
                t = w.get("text", "")
                if cur is None:
                    cur = {"text": t, "start": w["start"] / 1000.0, "end": w["end"] / 1000.0}
                else:
                    cur["text"] += t
                    cur["end"] = w["end"] / 1000.0
            if cur:
                out.append(cur)
        else:
            out.append({"text": item.get("text", ""), "start": 0.0,
                        "end": item.get("end_time", 0) / 1000.0 if item.get("end_time") else 0.0})
    return out


def transcribe_words(wav_path: str) -> list[dict]:
    """字级时间戳转写: [{'text':原始token, 'start':秒, 'end':秒}, ...]（滤除纯标点）。"""
    model = get_asr()
    res = model.generate(input=wav_path, batch_size_s=60)
    words: list[dict] = []
    for item in res:
        for w in item.get("sentence_info") or []:
            t = str(w.get("text", ""))
            start, end = w.get("start"), w.get("end")
            if start is None or end is None or not t:
                continue
            if not _PUNC_RE.sub("", t):
                continue          # 纯标点 token 无定位意义
            words.append({"text": t, "start": start / 1000.0, "end": end / 1000.0})
    return words


def find_occurrences(words: list[dict], keyword: str | None = None,
                     pattern: str | None = None) -> list[tuple[float, float, str]]:
    """在字级序列中查找关键词/正则，返回 [(start秒, end秒, 匹配文本), ...]。

    匹配基于去除标点后的连接文本，跨标点也能命中；结果按时间排序。
    """
    parts = [_PUNC_RE.sub("", w["text"]) for w in words]
    text = "".join(parts)
    if not text:
        return []
    idx: list[int] = []                     # 字符位置 -> 词索引
    for wi, p in enumerate(parts):
        idx.extend([wi] * len(p))
    spans: list[tuple[int, int]] = []
    if keyword:
        start = 0
        while True:
            p = text.find(keyword, start)
            if p < 0:
                break
            spans.append((p, p + len(keyword)))
            start = p + len(keyword)
    elif pattern:
        for m in re.finditer(pattern, text):
            if m.end() > m.start():
                spans.append((m.start(), m.end()))
    out: list[tuple[float, float, str]] = []
    for a, b in spans:
        out.append((words[idx[a]]["start"], words[idx[b - 1]]["end"], text[a:b]))
    out.sort(key=lambda x: x[0])
    return out


def compute_split_segments(occ: list[tuple[float, float, str]], total: float,
                           kind: str, pad_before: float = 0.1,
                           pad_after: float = 0.1) -> list[tuple[float, float]]:
    """根据匹配位置计算分割段 [(start,end)...]。

    kind:
      head  以关键字开头分割：段止于匹配开始前 pad_before，匹配本身不进入任何段
      tail  以关键字结束分割：段止于匹配结束后 pad_after，段连续覆盖全文件
      erase 去掉关键字分割：匹配及其后 pad_after 均不进入任何段
    """
    segs: list[tuple[float, float]] = []
    cur = 0.0
    for s, e, _t in occ:
        if kind == "tail":
            end = min(total, e + pad_after)
            if end - cur > 0.05:
                segs.append((cur, end))
            cur = end
        elif kind == "erase":
            end = s - pad_before
            if end - cur > 0.05:
                segs.append((cur, end))
            cur = max(cur, e + pad_after)
        else:  # head
            end = s - pad_before
            if end - cur > 0.05:
                segs.append((cur, end))
            cur = max(cur, e)
    if total - cur > 0.05:
        segs.append((cur, total))
    return segs


def find_keyword_segments(wav_path: str, keywords: list[str]) -> tuple[list[tuple[float, float]], list[str]]:
    """转写并查找包含关键词的句子时间段。"""
    sents = transcribe_with_timestamps(wav_path)
    hits: list[tuple[float, float]] = []
    matched: list[str] = []
    for s in sents:
        text = s.get("text", "")
        for kw in keywords:
            if kw and kw in text:
                hits.append((float(s.get("start", 0)), float(s.get("end", 0))))
                matched.append(text)
                break
    return hits, matched


# ---------------------------------------------------------------- 导出文件

def _fmt_srt(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_lrc(t: float) -> str:
    t = max(0.0, t)
    m = int(t // 60)
    return f"[{m:02d}:{t - m * 60:05.2f}]"


def export_txt(sents: list[dict], path: str) -> str:
    """纯文本文档，每句一行。"""
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            f.write(s.get("text", "").strip() + "\n")
    return path


def export_srt(sents: list[dict], path: str) -> str:
    """SRT 字幕文件。"""
    out: list[str] = []
    for i, s in enumerate(sents, 1):
        out.append(str(i))
        out.append(f"{_fmt_srt(s.get('start', 0))} --> {_fmt_srt(s.get('end', 0))}")
        out.append(s.get("text", "").strip())
        out.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


def export_lrc(sents: list[dict], path: str, title: str | None = None) -> str:
    """LRC 歌词文件，使用句子开始时间戳。"""
    lines: list[str] = []
    if title:
        lines.append(f"[ti:{title}]")
    lines.append("[ar:ClearVoice ASR]")
    for s in sents:
        text = s.get("text", "").strip()
        if text:
            lines.append(f"{_fmt_lrc(s.get('start', 0))}{text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def modelscope_enhance(wav_path: str, out_wav: str) -> str:
    """魔塔 ZipEnhancer 语音增强（消除背景噪声/杂音，离线小模型）。"""
    global _enhance_pipeline
    if not enhance_available():
        raise RuntimeError(
            "未安装魔塔语音增强依赖。请运行: uv sync --extra modelscope "
            f"(模型: {ENHANCE_MODEL_ID})")
    if _enhance_pipeline is None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        _enhance_pipeline = pipeline(Tasks.speech_enhancement, model=ENHANCE_MODEL_ID, device="cpu")
    result = _enhance_pipeline(input=wav_path)
    import soundfile as sf
    key = "output_waveform" if "output_waveform" in result else "output"
    sf.write(out_wav, result[key].astype("float32"), 16000)
    return out_wav
