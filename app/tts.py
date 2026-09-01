"""文字转语音：edge-tts（线上免费，微软服务）/ OpenAI 兼容 TTS API（本地或线上）。

支持中文（普通话）、英语、粤语。输出 MP3。
"""
from __future__ import annotations

# (显示名, edge-tts 音色) —— 音色名同时用作 OpenAI 兼容 API 的 voice 参数
VOICES = [
    ("中文（女·晓晓）", "zh-CN-XiaoxiaoNeural"),
    ("中文（男·云希）", "zh-CN-YunxiNeural"),
    ("英语（女·Aria）", "en-US-AriaNeural"),
    ("英语（男·Guy）", "en-US-GuyNeural"),
    ("粤语（女·曉曼）", "zh-HK-HiuMaanNeural"),
    ("粤语（男·雲龍）", "zh-HK-WanLungNeural"),
]


def edge_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


def _synth_edge(text: str, out: str, voice: str) -> None:
    import asyncio

    import edge_tts

    async def _run():
        com = edge_tts.Communicate(text, voice)
        await com.save(out)

    asyncio.run(_run())


def _synth_api(text: str, out: str, voice: str) -> None:
    """OpenAI 兼容 /v1/audio/speech 接口（本地 TTS 服务或线上）。"""
    import requests
    from . import config
    cfg = config.load_config()
    base = (cfg.get("api_base") or "http://localhost:11434/v1").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    payload = {
        "model": cfg.get("tts_api_model") or "tts-1",
        "voice": voice,
        "input": text,
        "response_format": "mp3",
    }
    r = requests.post(f"{base}/audio/speech", headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    with open(out, "wb") as f:
        f.write(r.content)


def synth(text: str, out: str, voice: str, progress_cb=None) -> str:
    """按设置页选择的 TTS 后端合成。返回输出文件路径。"""
    from . import config
    if not text.strip():
        raise ValueError("合成文本为空")
    if progress_cb:
        progress_cb(10, f"文字转语音（{voice}）…")
    if config.load_config().get("tts_backend", "edge") == "api":
        _synth_api(text, out, voice)
    else:
        if not edge_available():
            raise RuntimeError("未安装 edge-tts。请运行: uv sync --extra modelscope")
        _synth_edge(text, out, voice)
    if progress_cb:
        progress_cb(95, "合成完成")
    return out
