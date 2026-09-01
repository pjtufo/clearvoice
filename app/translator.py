"""文本翻译：本地 M2M100 小模型 / OpenAI 兼容大模型 API（本地 Ollama 或线上均可）。

语言对：英译中 / 中译英 / 日译中。含双语 TXT / SRT / LRC 导出。
本地小模型权重解析顺序：设置页「翻译模型目录」→ HF 镜像自动下载（facebook/m2m100_418M）。
"""
from __future__ import annotations

import os

PAIRS = {"英译中": ("en", "zh"), "中译英": ("zh", "en"), "日译中": ("ja", "zh")}
LANG_NAMES = {"zh": "中文", "en": "英语", "ja": "日语"}

_MT_MODEL_ID = "facebook/m2m100_418M"
_MT = None          # (tokenizer, model)


def mt_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_model_dir() -> str | None:
    from . import config
    d = config.load_config().get("translate_model_dir", "")
    if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json")):
        return d
    return None


def get_mt():
    """加载 M2M100 多语言翻译模型（懒加载）。"""
    global _MT
    if _MT is not None:
        return _MT
    if not mt_available():
        raise RuntimeError("未安装 transformers。请运行: uv sync --extra modelscope")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内镜像
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    src = _resolve_model_dir() or _MT_MODEL_ID
    tok = AutoTokenizer.from_pretrained(src)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(src)
    mdl.eval()
    _MT = (tok, mdl)
    return _MT


# ---------------------------------------------------------------- 本地小模型

def translate_texts_local(texts: list[str], src_lang: str, tgt_lang: str,
                          progress_cb=None, batch: int = 8) -> list[str]:
    import torch
    tok, mdl = get_mt()
    tok.src_lang = src_lang
    forced_bos = tok.get_lang_id(tgt_lang)
    outs: list[str] = []
    total = len(texts)
    for i in range(0, total, batch):
        chunk = [t if t.strip() else "。" for t in texts[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
        with torch.no_grad():
            gen = mdl.generate(**enc, forced_bos_token_id=forced_bos,
                               num_beams=4, max_length=256)
        outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        if progress_cb:
            progress_cb(min(100, int(100 * (i + len(chunk)) / total)))
    return outs


# ---------------------------------------------------------------- 大模型 API

def translate_texts_llm(texts: list[str], src_lang: str, tgt_lang: str,
                        progress_cb=None) -> list[str]:
    """OpenAI 兼容 chat API 逐句翻译。base_url 指向本地（如 Ollama）或线上均可。"""
    import json as _json

    import requests
    from . import config
    cfg = config.load_config()
    base = (cfg.get("api_base") or "http://localhost:11434/v1").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    sys_prompt = (f"你是专业字幕翻译。把用户给出的{LANG_NAMES[src_lang]}句子翻译成"
                  f"{LANG_NAMES[tgt_lang]}。只输出译文本身，不要解释、不要引号、不要标注序号。")
    outs: list[str] = []
    total = len(texts)
    for i, t in enumerate(texts):
        payload = {
            "model": cfg.get("api_model") or "qwen2.5:7b",
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": t if t.strip() else "…"}],
            "temperature": 0.2, "stream": False,
        }
        r = requests.post(url, headers=headers, data=_json.dumps(payload), timeout=180)
        r.raise_for_status()
        outs.append(r.json()["choices"][0]["message"]["content"].strip())
        if progress_cb:
            progress_cb(int(100 * (i + 1) / total))
    return outs


def translate(texts: list[str], src_lang: str, tgt_lang: str,
              progress_cb=None) -> list[str]:
    """按设置页选择的翻译后端执行。"""
    from . import config
    if config.load_config().get("translate_backend", "local") == "api":
        return translate_texts_llm(texts, src_lang, tgt_lang, progress_cb)
    return translate_texts_local(texts, src_lang, tgt_lang, progress_cb)


# ---------------------------------------------------------------- 双语导出

def export_txt_bilingual(sents: list[dict], path: str) -> str:
    """双语文本文档：每句两行（原文 / 译文），句间空行。"""
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            f.write(s.get("text", "").strip() + "\n")
            f.write(s.get("trans", "").strip() + "\n\n")
    return path


def export_srt_bilingual(sents: list[dict], path: str) -> str:
    """双语 SRT：每条字幕两行（原文在上、译文在下）。"""
    from .asr import _fmt_srt
    out: list[str] = []
    for i, s in enumerate(sents, 1):
        out.append(str(i))
        out.append(f"{_fmt_srt(s.get('start', 0))} --> {_fmt_srt(s.get('end', 0))}")
        out.append(s.get("text", "").strip())
        out.append(s.get("trans", "").strip())
        out.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


def export_lrc_bilingual(sents: list[dict], path: str, title: str | None = None) -> str:
    """双语 LRC：同一时间戳一行，原文与译文用「 / 」分隔。"""
    from .asr import _fmt_lrc
    lines: list[str] = []
    if title:
        lines.append(f"[ti:{title}]")
    lines.append("[ar:ClearVoice ASR+MT]")
    for s in sents:
        text = s.get("text", "").strip()
        trans = s.get("trans", "").strip()
        if not text and not trans:
            continue
        lines.append(f"{_fmt_lrc(s.get('start', 0))}{' / '.join(x for x in (text, trans) if x)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
