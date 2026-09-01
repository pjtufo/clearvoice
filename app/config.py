"""ClearVoice 简单配置持久化（JSON，位于用户主目录）。"""
from __future__ import annotations

import json
import os

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".clearvoice_config.json")

DEFAULTS = {
    "funasr_model_dir": r"D:\funasrModel",
    "asr_device": "cpu",
    "demucs_model_dir": "",
    # 翻译
    "translate_backend": "local",   # local=M2M100 小模型 / api=OpenAI兼容大模型
    "translate_model_dir": "",      # M2M100 权重目录（留空自动下载，HF 镜像）
    # 文字转语音
    "tts_backend": "edge",          # edge=edge-tts(线上免费) / api=OpenAI兼容TTS
    "tts_api_model": "tts-1",
    # OpenAI 兼容服务（本地大模型如 Ollama，或线上 API 均可）
    "api_base": "http://localhost:11434/v1",
    "api_key": "",
    "api_model": "qwen2.5:7b",
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: dict) -> None:
    data = {k: cfg.get(k, DEFAULTS[k]) for k in DEFAULTS}
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
