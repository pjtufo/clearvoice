"""文件/目录批处理工具。

- 目录递归扫描（文件/目录混合输入 → 媒体文件列表）
- 批量输出时保持源目录相对结构
- 文件名主干长度限制（超长截断，冲突自动加序号）
- 批量重命名：对齐 xrename.bat 功能（字符串替换 / keep 保留子串 /
  lcut 左切除 / cut 中间切除），带空名、同名、目标已存在保护

字符位置约定：0 基字符索引（非字节），支持负数（从右数），与 Python 切片一致。
"""
from __future__ import annotations

import os

from .ffmpeg_tools import VIDEO_EXTS, AUDIO_EXTS

MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS


# ---------------------------------------------------------------- 扫描

def scan_inputs(paths: list[str], exts: set[str] | None = None,
                recursive: bool = True) -> list[str]:
    """文件/目录混合列表 → 文件绝对路径列表。

    目录按扩展名过滤递归穷举子目录及文件；文件直接保留（可选扩展名校验）。
    结果去重并排序。
    """
    exts = {e.lower() for e in exts} if exts else None
    found: set[str] = set()
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            if exts is None or os.path.splitext(p)[1].lower() in exts:
                found.add(p)
        elif os.path.isdir(p):
            if recursive:
                for root, _dirs, files in os.walk(p):
                    for f in files:
                        fp = os.path.join(root, f)
                        if exts is None or os.path.splitext(f)[1].lower() in exts:
                            found.add(os.path.abspath(fp))
            else:
                for f in os.listdir(p):
                    fp = os.path.join(p, f)
                    if os.path.isfile(fp) and (
                            exts is None or os.path.splitext(f)[1].lower() in exts):
                        found.add(os.path.abspath(fp))
    return sorted(found)


def source_root(paths: list[str]) -> str:
    """推断源根目录：输入中第一个目录；全是文件时取其共同目录。"""
    dirs = [os.path.abspath(p) for p in paths if os.path.isdir(p)]
    if dirs:
        return dirs[0]
    files = [os.path.abspath(p) for p in paths if os.path.isfile(p)]
    if files:
        return os.path.dirname(os.path.commonpath(files)) if len(files) > 1 \
            else os.path.dirname(files[0])
    return ""


# ---------------------------------------------------------------- 输出路径

def truncate_stem(stem: str, max_len: int) -> str:
    """文件名主干截断到 max_len 个字符（0=不截断）。"""
    if max_len and len(stem) > max_len:
        return stem[:max_len]
    return stem


def unique_path(path: str) -> str:
    """目标已存在时在主干后加 _1/_2… 序号，返回不冲突路径。"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def plan_output(src_file: str, src_root: str, out_root: str,
                suffix: str = "", ext: str | None = None,
                keep_structure: bool = True, max_name_len: int = 0) -> str:
    """计算批量处理输出路径。

    out_root 为空 → 输出到源文件同目录；
    keep_structure 且文件位于 src_root 之下 → 在 out_root 中重建相对目录；
    suffix 追加到主干后；ext 覆盖扩展名（含点，如 ".mp3"）；
    max_name_len>0 时主干超长截断；重名自动加序号。
    """
    src_file = os.path.abspath(src_file)
    src_dir, name = os.path.split(src_file)
    stem, src_ext = os.path.splitext(name)
    out_ext = ext if ext is not None else src_ext
    stem = truncate_stem(stem, max_name_len)
    out_name = f"{stem}{suffix}{out_ext}"

    rel_dir = ""
    if keep_structure and src_root:
        root = os.path.abspath(src_root)
        if src_file.startswith(root + os.sep):
            rel_dir = os.path.relpath(os.path.dirname(src_file), root)

    target_dir = os.path.join(out_root, rel_dir) if out_root else (
        os.path.join(src_dir, rel_dir) if rel_dir else src_dir)
    os.makedirs(target_dir, exist_ok=True)
    return unique_path(os.path.join(target_dir, out_name))


# ---------------------------------------------------------------- 批量重命名

def transform_stem(stem: str, mode: str, n: int = 0, m: int = 0,
                   find: str = "", repl: str = "") -> str:
    """文件名主干变换（对齐 xrename.bat 四种模式，位置为 0 基字符索引）。

    - replace: 把 find 替换为 repl；
    - keep:    从第 n 位起保留——m>0 保留 m 个字符，m<0 保留到倒数 |m| 位；
    - lcut:    去掉前 m 位；
    - cut:     删除第 n 位到第 m 位之间的字符（保留 n 之前与 m 之后）。
    """
    if mode == "replace":
        return stem.replace(find, repl) if find else stem
    if mode == "keep":
        return stem[n:n + m] if m > 0 else stem[n:m]
    if mode == "lcut":
        return stem[m:]
    if mode == "cut":
        return stem[:n] + stem[m:]
    raise ValueError(f"未知重命名模式: {mode}")


def plan_rename(files: list[str], mode: str, n: int = 0, m: int = 0,
                find: str = "", repl: str = "", max_name_len: int = 0,
                auto_index: bool = True) -> list[dict]:
    """生成重命名计划（不执行）。

    返回 [{old, new, status, note}]；status: ok / skip（空名、未变化、
    目标已存在）。auto_index=True 时目标重名自动加序号。
    """
    plan: list[dict] = []
    used: set[str] = set()
    for old in files:
        d, name = os.path.split(old)
        stem, ext = os.path.splitext(name)
        try:
            new_stem = transform_stem(stem, mode, n, m, find, repl)
        except Exception as e:  # 位置参数异常等
            plan.append({"old": old, "new": "", "status": "skip", "note": f"变换失败: {e}"})
            continue
        new_stem = truncate_stem(new_stem, max_name_len)
        if not new_stem:
            plan.append({"old": old, "new": "", "status": "skip", "note": "结果为空"})
            continue
        new = os.path.join(d, new_stem + ext)
        if os.path.abspath(new) == os.path.abspath(old):
            plan.append({"old": old, "new": new, "status": "skip", "note": "名称未变化"})
            continue
        if os.path.exists(new) or new in used:
            if auto_index:
                base = os.path.join(d, new_stem)
                i = 1
                while os.path.exists(f"{base}_{i}{ext}") or f"{base}_{i}{ext}" in used:
                    i += 1
                new = f"{base}_{i}{ext}"
            else:
                plan.append({"old": old, "new": new, "status": "skip", "note": "目标已存在"})
                continue
        used.add(new)
        plan.append({"old": old, "new": new, "status": "ok", "note": ""})
    return plan


def apply_rename(plan: list[dict], progress_cb=None) -> tuple[int, list[dict]]:
    """执行重命名计划，返回 (成功数, 失败/跳过列表)。"""
    done = 0
    failed = []
    total = len(plan)
    for i, item in enumerate(plan):
        if item["status"] != "ok":
            failed.append(item)
            continue
        try:
            os.rename(item["old"], item["new"])
            done += 1
        except OSError as e:
            item["note"] = f"重命名失败: {e}"
            failed.append(item)
        if progress_cb:
            progress_cb(int(95 * (i + 1) / max(1, total)),
                        f"重命名 {i + 1}/{total}: {os.path.basename(item['old'])}")
    return done, failed
