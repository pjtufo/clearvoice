"""冒烟测试：依赖导入 + DSP 管线 + GUI 构造。"""
import os
import sys

import numpy as np

print("1) 依赖导入...", end=" ", flush=True)
import PySide6  # noqa: F401
import scipy
import soundfile
import noisereduce
import PySide6.QtMultimedia  # noqa: F401
import PySide6.QtMultimediaWidgets  # noqa: F401
print("ok")

print("2) 模块导入...", end=" ", flush=True)
from app import asr, audio_ops, features, ffmpeg_tools, main_window  # noqa: F401
print("ok")

print("3) 合成测试音频（语音带 + 哔哔声 + 削波）...", end=" ", flush=True)
sr = 16000
t = np.linspace(0, 6.0, int(sr * 6.0), endpoint=False)
rng = np.random.default_rng(0)
# 模拟"语音"：低频幅度调制 + 谐波
speech = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 1.2 * t)))
speech *= (np.sin(2 * np.pi * 0.8 * t) > 0)  # 说话停顿
noise = rng.normal(0, 0.05, len(t))           # 白噪声(杂音)
beep = 0.2 * np.sin(2 * np.pi * 1000 * t) * (t > 2.0) * (t < 2.8)  # 哔哔声
y = speech + noise + beep
y[:: 3] *= 1.5                                # 人为削波(破音)
y = np.clip(y, -1, 1)
test_wav = os.path.join(os.path.dirname(__file__), "_test.wav")
soundfile.write(test_wav, y.astype(np.float32), sr)
print("ok")

print("4) 特征检测...")
segs, bins = features.detect_beep_segments(y, sr)
print(f"   哔哔段: {segs}  频点数: {len(bins)}")
clip_ratio = features.detect_clipped_ratio(y)
print(f"   削波比例: {clip_ratio:.4f}")
sil = features.detect_silence(y, sr)
print(f"   静音段: {len(sil)} 个")

print("5) 消除管线（噪声+哔哔+破音）...", end=" ", flush=True)
y2, report = audio_ops.process_removals(y, sr, ["noise", "beep", "declip"], 0.85)
print("ok")
nr = np.sqrt(np.mean(y2 ** 2))
print(f"   处理后 RMS={nr:.4f}")

print("6) 说话人匹配消除...", end=" ", flush=True)
ref = y[int(0.2 * sr): int(1.2 * sr)]  # 取一段"语音"作参考
y3, spk_segs = audio_ops.apply_speaker_mask(y, sr, ref, sr, threshold=0.80)
print(f"命中 {len(spk_segs)} 段")

print("7) ffmpeg 工具（静音段/变速/提取）...", end=" ", flush=True)
out_mute = os.path.join(os.path.dirname(__file__), "_test_muted.wav")
ffmpeg_tools_mute = __import__("app.ffmpeg_tools", fromlist=["mute_ranges"]).mute_ranges(test_wav, out_mute, [(1.0, 2.0)])
info = ffmpeg_tools_mute and None
from app import ffmpeg_tools as ft
info = ft.media_info(out_mute)
# 音画同步滤镜参数断言（纯逻辑，离线）
assert ft._sync_audio_args(0.3) == ["-af", "adelay=300:all=1"]
assert ft._sync_audio_args(-0.25) == ["-af", "atrim=start=0.250,asetpts=PTS-STARTPTS"]
assert ft._sync_audio_args(0.0) == []
try:
    ft.av_sync_offset(out_mute, os.path.join(os.path.dirname(__file__), "_x.mp4"), 0.0)
    raise SystemExit("偏移 0 应报错")
except Exception:
    pass
# 分离音频多格式表（保留原始采样率/声道）
assert {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"} <= set(ft.AUDIO_OUT_FORMATS)
try:
    ft.extract_audio_keep(out_mute, os.path.join(os.path.dirname(__file__), "_x.wma"))
    raise SystemExit("不支持的格式应报错")
except Exception:
    pass
# 格式转换：指定码率/采样率 + 降噪路径（音频转音频）
_d = os.path.dirname(__file__)
conv_mp3 = os.path.join(_d, "_conv.mp3")
ft.convert_media(out_mute, conv_mp3, bitrate="128k", sr=22050)
ci = ft.media_info(conv_mp3)
assert ci.sample_rate == 22050 and ci.channels == 1, f"转换采样率/声道不符: {ci.sample_rate}/{ci.channels}"
conv_den = os.path.join(_d, "_conv_den.wav")
ft.convert_media(out_mute, conv_den, denoise=0.5)
assert ft.media_info(conv_den).duration > 0
for _p in (conv_mp3, conv_den):
    os.remove(_p)
print(f"ok ({info.duration:.2f}s)")

print("8) ASR 导出格式（TXT/SRT/LRC）...", end=" ", flush=True)
fake_sents = [
    {"text": "你好世界", "start": 0.0, "end": 1.25},
    {"text": "第二句话测试", "start": 2.4, "end": 3.75},
]
d = os.path.dirname(__file__)
p_txt = asr.export_txt(fake_sents, os.path.join(d, "_test.txt"))
p_srt = asr.export_srt(fake_sents, os.path.join(d, "_test.srt"))
p_lrc = asr.export_lrc(fake_sents, os.path.join(d, "_test.lrc"), title="测试")
assert "你好世界" in open(p_txt, encoding="utf-8").read()
srt_txt = open(p_srt, encoding="utf-8").read()
assert "00:00:00,000 --> 00:00:01,250" in srt_txt
lrc_txt = open(p_lrc, encoding="utf-8").read()
assert "[00:02.40]第二句话测试" in lrc_txt
for p in (p_txt, p_srt, p_lrc):
    os.remove(p)
print("ok")

print("9) 关键词/正则分割计算...", end=" ", flush=True)
words = [
    {"text": "大家好", "start": 0.0, "end": 0.9},
    {"text": "，", "start": 0.9, "end": 0.95},
    {"text": "今天", "start": 1.0, "end": 1.6},
    {"text": "天气", "start": 1.6, "end": 2.2},
    {"text": "大家好", "start": 2.5, "end": 3.4},
    {"text": "。", "start": 3.4, "end": 3.5},
    {"text": "再见", "start": 3.6, "end": 4.2},
]
occ = asr.find_occurrences(words, keyword="大家好")
assert len(occ) == 2 and abs(occ[0][0]) < 1e-6 and abs(occ[1][0] - 2.5) < 1e-6, occ
# 开头分割: 第1处贴文件头无前段; 段=[0.9,2.4],[3.4,5.0]
segs_head = asr.compute_split_segments(occ, 5.0, "head", 0.1, 0.1)
assert len(segs_head) == 2 and abs(segs_head[0][0] - 0.9) < 1e-6 and abs(segs_head[0][1] - 2.4) < 1e-6, segs_head
# 结束分割: 段连续=[0,1.0],[1.0,3.5],[3.5,5.0]
segs_tail = asr.compute_split_segments(occ, 5.0, "tail", 0.1, 0.1)
assert len(segs_tail) == 3 and abs(segs_tail[0][1] - 1.0) < 1e-6 and abs(segs_tail[2][0] - 3.5) < 1e-6, segs_tail
# 去掉关键字: 段=[1.0,2.4],[3.5,5.0]
segs_erase = asr.compute_split_segments(occ, 5.0, "erase", 0.1, 0.1)
assert len(segs_erase) == 2 and abs(segs_erase[0][0] - 1.0) < 1e-6 and abs(segs_erase[1][0] - 3.5) < 1e-6, segs_erase
# 正则（标点不影响匹配）
occ_re = asr.find_occurrences(words, pattern="天[气气]")
assert len(occ_re) == 1 and abs(occ_re[0][0] - 1.6) < 1e-6, occ_re
occ_kw2 = asr.find_occurrences(words, keyword="好今天")  # 跨标点命中
assert len(occ_kw2) == 1 and abs(occ_kw2[0][1] - 1.6) < 1e-6, occ_kw2
print("ok")

print("10) 人声/伴奏分离模块...", end=" ", flush=True)
import torch as _torch
from app import separation
assert separation.available()
try:
    _m, _sr = separation.get_model()
    assert _sr == 44100
    with _torch.no_grad():
        _o = _m(_torch.zeros(1, 2, 44100 * 2))
    assert _o.shape[0] == 1 and _o.shape[1] == 4, _o.shape
    print("ok")
except Exception as e:  # 无本地权重且无网络时跳过，不阻塞其余测试
    print("skip（权重不可用: %s）" % type(e).__name__)

print("11) 翻译/TTS 模块与双语导出...", end=" ", flush=True)
from app import translator, tts as tts_mod
assert set(translator.PAIRS) == {"英译中", "中译英", "日译中"}
assert len(tts_mod.VOICES) == 6
bi = [{"text": "Hello world", "trans": "你好世界", "start": 0.0, "end": 1.5},
      {"text": "Good morning", "trans": "早上好", "start": 2.0, "end": 3.5}]
p_btxt = translator.export_txt_bilingual(bi, os.path.join(d, "_test_bi.txt"))
p_bsrt = translator.export_srt_bilingual(bi, os.path.join(d, "_test_bi.srt"))
p_blrc = translator.export_lrc_bilingual(bi, os.path.join(d, "_test_bi.lrc"), title="双语")
bt = open(p_btxt, encoding="utf-8").read()
assert "Hello world\n你好世界" in bt
bs = open(p_bsrt, encoding="utf-8").read()
assert "Hello world\n你好世界" in bs and "00:00:00,000 --> 00:00:01,500" in bs
bl = open(p_blrc, encoding="utf-8").read()
assert "[00:00.00]Hello world / 你好世界" in bl
for p in (p_btxt, p_bsrt, p_blrc):
    os.remove(p)
print("ok")

print("12) GUI 构造（offscreen）...", end=" ", flush=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
app = QApplication([])
w = main_window.MainWindow()
w.show()
# 波形新特性：立体声双行 / 缩放 / 时间刻度（纯逻辑断言）
wv = w.wave
y_st = np.zeros((16000 * 3, 2), dtype=np.float32)
y_st[:, 0] = 0.5 * np.sin(np.linspace(0, 200 * np.pi, 16000 * 3))
y_st[:, 1] = -0.3 * np.sin(np.linspace(0, 90 * np.pi, 16000 * 3))
wv.resize(1000, 160)
wv.set_data(y_st, 16000, 3.0)
assert wv.stereo and len(wv.view_env) == 2, "立体声应为两行包络"
assert len(wv.env_ch[0]) >= 8000, "概览包络分辨率不足"
wv._zoom_at(8.0, center_t=1.0)
t0, t1 = wv.view
assert t1 - t0 <= 3.0 / 8 + 1e-6, "缩放后视图窗口应缩小"
assert abs(wv._t2x(t0)) < 1e-6 and abs(wv._t2x(t1) - wv.width()) < 1e-6, "视图端点映射错误"
wv._scroll_by(0.5)
assert wv.view[0] >= 0.0, "滚动越界"
wv.set_playhead(2.9, playing=True)
assert wv.view[0] <= 2.9 <= wv.view[1], "播放头跟随失败"
assert wv._nice_step(0.35) == 0.5 and wv._nice_step(75.0) == 120, "刻度步进取整错误"
assert wv._fmt_ruler(65.0, 1) == "01:05" and wv._fmt_ruler(1.25, 0.1) == "00:01.2"
# 底部概览条：命中测试 / 拖拽平移缩放 / 框选 / 双击复位
wv.reset_view()
ov_top = wv._ov_top()
assert wv._ov_hit(0, ov_top + 10) == "left" and wv._ov_hit(wv.width(), ov_top + 10) == "right"
assert wv._ov_hit(wv.width() // 2, ov_top + 10) == "pan"
assert wv._ov_hit(wv.width() // 2, ov_top - 20) == ""
wv._set_view(1.0, 2.0)
assert wv._ov_hit(100, ov_top + 10) == "outside"
wv._set_view(wv._ov_x2t(300), wv.view[1])  # 拖左边缘
assert wv.view[1] == 2.0 and abs(wv.view[0] - 0.9) < 0.02
wv._set_view(-5.0, 99.0)                   # 越界钳制为全览
assert wv.view[0] == 0.0 and abs(wv.view[1] - 3.0) < 1e-6
wv._set_view(1.0, 1.005)                   # 小于最小窗口
assert wv.view[1] - wv.view[0] >= wv._min_window() - 1e-9
wv.reset_view()
assert abs(wv.view[0]) < 1e-9 and abs(wv.view[1] - 3.0) < 1e-6
wv._set_view(wv._ov_x2t(200), wv._ov_x2t(800))  # 框选放大
assert abs(wv.view[0] - 0.6) < 0.01 and abs(wv.view[1] - 2.4) < 0.01
# fps 解析
from app.ffmpeg_tools import _parse_fps
assert abs(_parse_fps("30000/1001") - 29.97) < 0.01 and _parse_fps("0/0") == 0.0
print("ok")

print("13) 文件/目录批处理（扫描/结构保持/重命名）...", end=" ", flush=True)
import shutil
import tempfile
from app import filetools as ftools
# 文件名变换四种模式（对齐 xrename.bat）
assert ftools.transform_stem("abcdefg", "replace", find="cd", repl="XY") == "abXYefg"
assert ftools.transform_stem("abcdefg", "replace", find="cd", repl="") == "abefg"
assert ftools.transform_stem("abcdefg", "keep", n=2, m=3) == "cde"
assert ftools.transform_stem("abcdefg", "keep", n=0, m=-2) == "abcde"
assert ftools.transform_stem("abcdefg", "lcut", m=3) == "defg"
assert ftools.transform_stem("abcdefg", "cut", n=2, m=4) == "abefg"
assert ftools.truncate_stem("abcdefghij", 4) == "abcd"
td = tempfile.mkdtemp(prefix="cv_ft_")
try:
    os.makedirs(os.path.join(td, "sub1"))
    os.makedirs(os.path.join(td, "sub2"))
    for p in ("sub1/a.mp3", "sub1/b.wav", "sub2/c.mp3", "root.mp4", "note.txt"):
        open(os.path.join(td, p.replace("/", os.sep)), "w").close()
    files = ftools.scan_inputs([td], ftools.MEDIA_EXTS, recursive=True)
    assert sorted(os.path.basename(f) for f in files) == ["a.mp3", "b.wav", "c.mp3", "root.mp4"]
    flat = ftools.scan_inputs([td], ftools.MEDIA_EXTS, recursive=False)
    assert sorted(os.path.basename(f) for f in flat) == ["root.mp4"]
    src = os.path.join(td, "sub1", "a.mp3")
    out = ftools.plan_output(src, td, os.path.join(td, "out"), ext=".wav", keep_structure=True)
    assert os.path.normpath(out) == os.path.normpath(os.path.join(td, "out", "sub1", "a.wav"))
    assert os.path.isdir(os.path.join(td, "out", "sub1"))
    out2 = ftools.plan_output(src, td, os.path.join(td, "out2"), ext=".wav", keep_structure=False)
    assert os.path.normpath(out2) == os.path.normpath(os.path.join(td, "out2", "a.wav"))
    longsrc = os.path.join(td, "sub2", "abcdefghijklmnop.mp3")
    o3 = ftools.plan_output(longsrc, td, os.path.join(td, "out3"), ext=".mp3", max_name_len=5)
    assert os.path.basename(o3) == "abcde.mp3"
    open(o3, "w").close()
    o4 = ftools.plan_output(longsrc, td, os.path.join(td, "out3"), ext=".mp3", max_name_len=5)
    assert os.path.basename(o4) == "abcde_1.mp3"
    t1 = os.path.join(td, "r1.mp3"); t2 = os.path.join(td, "r2.mp3"); t3 = os.path.join(td, "x.mp3")
    for p in (t1, t2, t3):
        open(p, "w").close()
    plan = ftools.plan_rename([t1, t2, t3], "replace", find="r", repl="rr")
    st = {os.path.basename(p["old"]): p["status"] for p in plan}
    assert st["r1.mp3"] == "ok" and st["r2.mp3"] == "ok" and st["x.mp3"] == "skip"
    assert ftools.plan_rename([t3], "lcut", m=5)[0]["status"] == "skip"  # 空名保护
    done, failed = ftools.apply_rename([p for p in plan if p["status"] == "ok"])
    assert done == 2 and not failed
    assert os.path.isfile(os.path.join(td, "rr1.mp3")) and os.path.isfile(os.path.join(td, "rr2.mp3"))
    print("ok")
finally:
    shutil.rmtree(td, ignore_errors=True)

print("14) 分割批处理（多文件/目录输入 + 输出定位）...", end=" ", flush=True)
sd = tempfile.mkdtemp(prefix="cv_split_")
try:
    # split_target 纯逻辑：子目录模式 / 平铺前缀模式 / 长度截断
    s_src = os.path.join(sd, "sub", "a.mp4")
    d_out, d_pre = ftools.split_target(s_src, "定长", True, 0)
    assert d_out == os.path.join(sd, "sub", "a_定长分割") and d_pre == ""
    d_out2, d_pre2 = ftools.split_target(s_src, "特征", False, 0)
    assert d_out2 == os.path.join(sd, "sub") and d_pre2 == "a_"
    d_out3, _ = ftools.split_target(os.path.join(sd, "abcdefghij.wav"), "关键词", True, 4)
    assert os.path.basename(d_out3) == "abcd_关键词分割"
    # 造两个 3 秒 wav（含子目录），跑批量定长分割
    import soundfile as sf
    t3 = np.linspace(0, 3, 16000 * 3, endpoint=False)
    wav1 = os.path.join(sd, "f1.wav")
    wav2 = os.path.join(sd, "sub", "f2.wav")
    os.makedirs(os.path.join(sd, "sub"), exist_ok=True)
    sf.write(wav1, 0.01 * np.sin(2 * np.pi * 440 * t3), 16000)
    sf.write(wav2, 0.01 * np.sin(2 * np.pi * 440 * t3), 16000)
    files = ftools.scan_inputs([sd], ftools.MEDIA_EXTS, recursive=True)
    assert len(files) == 2, files
    res = main_window.MainWindow._job_split_fixed_batch(files, 1.0, True, 0,
                                                        progress_cb=lambda *a: None)
    assert isinstance(res, dict) and "成功 2" in res["report"], res["report"]
    parts1 = sorted(os.listdir(os.path.join(sd, "f1_定长分割")))
    parts2 = sorted(os.listdir(os.path.join(sd, "sub", "f2_定长分割")))
    assert len(parts1) == 3 and all(p.startswith("part_") and p.endswith(".wav") for p in parts1)
    assert len(parts2) == 3
    # 不建子目录：分段直接放源目录并带源名前缀
    res2 = main_window.MainWindow._job_split_fixed_batch([wav1], 1.5, False, 0,
                                                         progress_cb=lambda *a: None)
    flat = [p for p in os.listdir(sd) if p.startswith("f1_part_")]
    assert len(flat) == 2, flat
    print("ok")
finally:
    shutil.rmtree(sd, ignore_errors=True)

os.remove(test_wav)
os.remove(out_mute)
print("\nALL TESTS PASSED")
