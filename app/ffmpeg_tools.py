"""ffmpeg / ffprobe 命令行封装。"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# 视频常见扩展名 / 音频常见扩展名
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts", ".m4v", ".wmv"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".opus", ".amr"}


class FFmpegError(RuntimeError):
    pass


def run(args: list[str], timeout: int = 3600) -> str:
    """执行 ffmpeg/ffprobe，失败抛出异常，返回 stderr 文本。"""
    cmd = list(map(str, args))
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    err = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(f"命令失败: {' '.join(cmd)}\n{err[-3000:]}")
    return err


def probe(path: str) -> dict:
    """ffprobe 读取媒体信息（json）。"""
    out = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return json.loads(out.stdout.decode("utf-8", errors="replace"))


@dataclass
class MediaInfo:
    path: str = ""
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    width: int = 0
    height: int = 0
    sample_rate: int = 0
    channels: int = 0
    format_name: str = ""
    streams: list = field(default_factory=list)


def media_info(path: str) -> MediaInfo:
    data = probe(path)
    info = MediaInfo(path=path)
    fmt = data.get("format", {})
    info.duration = float(fmt.get("duration", 0) or 0)
    info.format_name = fmt.get("format_name", "")
    for st in data.get("streams", []):
        info.streams.append(st)
        if st.get("codec_type") == "video":
            info.has_video = True
            info.width = int(st.get("width", 0) or 0)
            info.height = int(st.get("height", 0) or 0)
            if info.duration <= 0:
                info.duration = float(st.get("duration", 0) or 0)
        elif st.get("codec_type") == "audio":
            info.has_audio = True
            info.sample_rate = int(st.get("sample_rate", 0) or 0)
            info.channels = int(st.get("channels", 0) or 0)
            if info.duration <= 0:
                info.duration = float(st.get("duration", 0) or 0)
    return info


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def is_audio(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS


def extract_audio(src: str, dst_wav: str, sr: int = 16000, mono: bool = True, start: float | None = None, end: float | None = None) -> str:
    """从任意媒体提取 wav（供 DSP 处理）。"""
    args = [FFMPEG, "-y", "-v", "error"]
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", src]
    if end is not None:
        if start is not None:
            args += ["-t", f"{max(0.0, end - start):.3f}"]
        else:
            args += ["-t", f"{end:.3f}"]
    args += ["-vn", "-ar", str(sr), "-ac", "1" if mono else "2", "-c:a", "pcm_s16le", dst_wav]
    run(args)
    return dst_wav


def mux_replace_audio(video: str, wav: str, out: str, align: str = "shortest", audio_offset: float = 0.0) -> str:
    """把处理后的音频与原视频合成。video 流直接 copy，音频编码 aac。

    align: "shortest"=对齐到较短流; "full"=保留完整长度; offset 为音频提前/延后秒数（正=延后）。
    """
    args = [FFMPEG, "-y", "-v", "error"]
    if audio_offset:
        args += ["-itsoffset", f"{audio_offset:.3f}"]
    args += ["-i", wav, "-i", video]
    args += ["-map", "0:a", "-map", "1:v", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if align == "shortest":
        args += ["-shortest"]
    args += [out]
    run(args)
    return out


def mute_range(src: str, out: str, start: float, end: float, inplace_style: bool = True) -> str:
    """把 [start,end] 的音频替换为静音，视频流 copy。"""
    args = [
        FFMPEG, "-y", "-v", "error", "-i", src,
        "-filter_complex",
        f"[0:a]volume=enable='between(t,{start},{end})':volume=0[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out,
    ]
    run(args)
    return out


def _reencode_args(path: str) -> list[str]:
    """按输出扩展名返回重编码参数。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in AUDIO_EXTS and ext not in (".m4a",):
        if ext == ".mp3":
            return ["-vn", "-c:a", "libmp3lame", "-q:a", "2"]
        if ext in (".aac", ".m4a"):
            return ["-vn", "-c:a", "aac", "-b:a", "192k"]
        return ["-vn", "-c:a", "pcm_s16le"]  # wav/flac等
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"]


def cut_range(src: str, out: str, start: float, end: float) -> str:
    """精确截取片段（重编码保证帧精度）。输出扩展名与 src 一致。"""
    if not os.path.splitext(out)[1]:
        out += os.path.splitext(src)[1]
    run([FFMPEG, "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
         *_reencode_args(out), out])
    return out


def remove_ranges(src: str, out: str, ranges: list[tuple[float, float]]) -> str:
    """剪切掉多个时间段后拼接（re-encode）。"""
    tmpdir = os.path.dirname(os.path.abspath(out))
    ext = os.path.splitext(src)[1].lower() or ".mp4"
    parts: list[str] = []
    cursor = 0.0
    import uuid
    reenc = _reencode_args(".mp4" if is_video(src) else ".wav")
    for s, e in sorted(ranges):
        if s - cursor > 0.01:
            p = os.path.join(tmpdir, f"_part_{uuid.uuid4().hex[:8]}{ext}")
            run([FFMPEG, "-y", "-v", "error", "-ss", f"{cursor:.3f}", "-to", f"{s:.3f}", "-i", src, *reenc, p])
            parts.append(p)
        cursor = max(cursor, e)
    if cursor < (media_info(src).duration - 0.01):
        p = os.path.join(tmpdir, f"_part_{uuid.uuid4().hex[:8]}{ext}")
        run([FFMPEG, "-y", "-v", "error", "-ss", f"{cursor:.3f}", "-i", src, *reenc, p])
        parts.append(p)
    if not parts:
        raise FFmpegError("没有可保留的内容")
    concat_list = os.path.join(tmpdir, f"_concat_{uuid.uuid4().hex[:8]}.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", out])
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass
    os.remove(concat_list)
    return out


def mute_ranges(src: str, out: str, ranges: list[tuple[float, float]]) -> str:
    """把多个时间段的音频静音（视频流 copy）。"""
    from .features import merge_overlapping
    ranges = merge_overlapping(ranges)
    conds = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in ranges)
    args = [FFMPEG, "-y", "-v", "error", "-i", src,
            "-filter_complex", f"[0:a]volume=enable='{conds}':volume=0[a]",
            "-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    if is_video(src):
        args = args[:-4] + ["-map", "0:v", "-c:v", "copy", "-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    args.append(out)
    run(args)
    return out


def split_fixed(src: str, out_dir: str, seg_seconds: float, base: str = "part") -> list[str]:
    """定长分割（流拷贝，速度快，切点在关键帧）。"""
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(src)[1].lower() or ".mp4"
    pattern = os.path.join(out_dir, f"{base}_%04d{ext}")
    run([FFMPEG, "-y", "-v", "error", "-i", src, "-f", "segment", "-segment_time", f"{seg_seconds:.3f}",
         "-reset_timestamps", "1", "-c", "copy", pattern])
    import glob
    return sorted(glob.glob(os.path.join(out_dir, f"{base}_*{ext}")))


def merge_av(video: str, audio: str, out: str, audio_offset: float = 0.0, use_shortest: bool = True) -> str:
    """视频 + 音频合并，时间轴对齐。"""
    args = [FFMPEG, "-y", "-v", "error"]
    if audio_offset:
        args += ["-itsoffset", f"{audio_offset:.3f}"]
    args += ["-i", audio, "-i", video,
             "-map", "1:v", "-map", "0:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if use_shortest:
        args += ["-shortest"]
    args += [out]
    run(args)
    return out


def change_speed(src: str, out: str, speed: float) -> str:
    """调整时间轴速度（视频 setpts + 音频 atempo 链），v/a 同步；纯音频只变速音频。"""
    if speed <= 0:
        raise ValueError("速度必须大于0")
    # atempo 有效范围 0.5~2.0，超出需要串联
    tempos: list[float] = []
    s = speed
    while s > 2.0:
        tempos.append(2.0)
        s /= 2.0
    while s < 0.5:
        tempos.append(0.5)
        s /= 0.5
    tempos.append(s)
    atempo = "".join(f"atempo={t:.6f}," for t in tempos)[:-1]
    args = [FFMPEG, "-y", "-v", "error", "-i", src]
    if is_video(src):
        pts = f"setpts=PTS/{speed:.6f}"
        fc = f"[0:v]{pts}[v];[0:a]{atempo}[a]"
        args += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]"]
    else:
        args += ["-filter:a", atempo, "-vn"]
    args += [out]
    run(args)
    return out


def encode_wav_to_media(wav: str, out: str, has_video_in_out: bool = False) -> str:
    """处理后的 wav 写回音频文件（mp3/m4a/wav 按扩展名）。"""
    args = [FFMPEG, "-y", "-v", "error", "-i", wav]
    ext = os.path.splitext(out)[1].lower()
    if ext in (".m4a", ".aac", ".mp4"):
        args += ["-c:a", "aac", "-b:a", "192k"]
    elif ext == ".mp3":
        args += ["-c:a", "libmp3lame", "-q:a", "2"]
    else:
        args += ["-c:a", "pcm_s16le"]
    args += [out]
    run(args)
    return out
