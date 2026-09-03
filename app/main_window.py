"""ClearVoice 主界面（PySide6）。

功能页签：消除杂音（多类型 DSP + 去背景音乐）/ 分割（批量文件·目录，定长/特征/关键词正则，输出子目录可选）/
合并分离 / 格式转换（批量·目录递归·保持结构）/ 文件名夹处理（批量重命名）/
时间轴（变速 / 音画同步微调 / 裁剪）/
特征剔除 / 语音转文字 / 翻译 / TTS / 设置。
各页签的源文件统一使用 SourceFileGroup：支持单个/多个文件与目录（递归穷举子目录），
列表留空时回退到当前打开的文件；批量任务逐文件执行、单文件失败不影响整体、统一汇总报告。
左栏：播放控制、波形选区（含可拖拽概览条）、说话人与特征参考标记、进度与日志。
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, QUrl, Signal, QPointF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFileDialog, QComboBox,
    QCheckBox, QDoubleSpinBox, QSpinBox, QTabWidget, QPlainTextEdit,
    QProgressBar, QMessageBox, QLineEdit, QGroupBox, QSizePolicy,
    QListWidget,
)

from . import ffmpeg_tools as ft
from . import audio_ops
from . import features
from . import asr as msa
from . import separation
from . import translator
from . import tts as tts_mod
from . import config


# ================================================================ 后台任务

class Worker(QThread):
    progress = Signal(int, str)
    done = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, name: str, fn, *args, **kwargs):
        super().__init__()
        self.name = name
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, progress_cb=self.progress.emit, **self.kwargs)
            self.done.emit(self.name, result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(self.name, f"{e}\n{traceback.format_exc(limit=3)}")


# ================================================================ 波形控件

class WaveformWidget(QWidget):
    """音频波形：时间刻度尺 + 立体声 L/R 双行 + 视图缩放。

    交互：点击定位 / 拖拽选区 / 滚轮水平滚动 / Ctrl+滚轮以光标为中心缩放；
    底部概览条可鼠标拖拽——拖窗口内部平移、拖左右边缘缩放、窗口外框选放大、
    双击复位全览；播放中播放头越出视图自动跟随滚动。短文件（≤10 分钟）保留
    原始采样，可缩放到单采样级；长文件缩放受概览包络分辨率限制。
    """
    clicked = Signal(float)          # 秒
    selection_drawn = Signal(float, float)  # 起止秒（拖拽选择）

    _RULER_H = 18
    _OV_H = 42          # 底部概览条高度
    _EDGE = 6           # 视图窗口边缘抓取宽度（像素）
    _STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.duration = 0.0
        self.playhead = 0.0
        self.sel: tuple[float, float] | None = None
        self.speaker_ref: tuple[float, float] | None = None
        self.feature_ref: tuple[float, float] | None = None
        self._press_x: int | None = None
        self._drag_x: int | None = None
        # 缩放视图（秒区间）
        self.view = [0.0, 0.0]
        self.stereo = False
        self.env_ch: list[np.ndarray] = []    # 整段概览包络（每声道 (n,2) min/max）
        self.view_env: list[np.ndarray] = []  # 当前视图包络（用于绘制）
        self._raw: np.ndarray | None = None   # 短文件保留原始采样（供深度缩放）
        self._raw_sr = 16000
        # 概览条拖拽状态：pan=拖窗口 / left,right=拖边缘 / select=窗口外框选
        self._ov_mode: str | None = None
        self._ov_start_t = 0.0
        self._ov_press_x = 0
        self._ov_sel_x: int | None = None
        self._ov_pan_anchor: tuple[float, float, float] | None = None  # (按下时t, view0, view1)
        self.setMouseTracking(True)

    # ---------- 数据 ----------
    def set_data(self, y: np.ndarray, sr: int, duration: float):
        self.duration = float(duration)
        self._raw_sr = sr
        self.stereo = getattr(y, "ndim", 1) > 1 and y.shape[1] > 1
        # 短文件（≤10 分钟）保留原始采样，支持缩放到采样级
        self._raw = y.astype(np.float32) if (0 < duration <= 600 and len(y)) else None
        chans = [y] if y.ndim == 1 else [y[:, 0], y[:, 1]]
        n = int(np.clip(max(8000, self.width() * 4), 8000, 120000))
        envs: list[np.ndarray] = []
        if len(y):
            for c in chans:
                m = max(1, len(c) // n)
                trim = (len(c) // m) * m
                yy = c[:trim].reshape(-1, m)
                envs.append(np.stack([yy.min(axis=1), yy.max(axis=1)], axis=1))
        self.env_ch = envs
        self.view = [0.0, self.duration] if self.duration > 0 else [0.0, 0.0]
        self._rebuild_view_env()
        self.update()

    def clear_all(self):
        self.duration = 0.0
        self.playhead = 0.0
        self.sel = None
        self.speaker_ref = None
        self.feature_ref = None
        self.view = [0.0, 0.0]
        self.stereo = False
        self.env_ch = []
        self.view_env = []
        self._raw = None
        self.update()

    def _rebuild_view_env(self):
        """按当前视图窗口重算绘制包络（有原始采样时精确到采样级）。"""
        if self.duration <= 0 or not self.env_ch:
            self.view_env = []
            return
        t0, t1 = self.view
        if self._raw is not None:
            sr = self._raw_sr
            i0 = int(t0 * sr)
            i1 = max(i0 + 1, int(t1 * sr))
            raw = self._raw
            chans = [raw] if raw.ndim == 1 else [raw[:, 0], raw[:, 1]]
            bins = int(np.clip(self.width() * 2, 500, 20000))
            envs = []
            for c in chans:
                seg = c[i0:i1]
                b = max(1, len(seg) // bins)
                trim = (len(seg) // b) * b
                if trim <= 0:
                    envs.append(np.array([[0.0, 0.0]], dtype=np.float32))
                    continue
                yy = seg[:trim].reshape(-1, b)
                envs.append(np.stack([yy.min(axis=1), yy.max(axis=1)], axis=1))
            self.view_env = envs
        else:
            envs = []
            for e in self.env_ch:
                nn = len(e)
                j0 = int(t0 / self.duration * nn)
                j1 = max(j0 + 1, int(t1 / self.duration * nn))
                envs.append(e[j0:j1])
            self.view_env = envs

    def _min_window(self) -> float:
        """缩放下限（秒）：有原始采样可放到 0.02s，否则受概览包络分辨率限制。"""
        if self.duration <= 0:
            return 0.02
        if self._raw is not None:
            return 0.02
        n = len(self.env_ch[0]) if self.env_ch else 8000
        return max(0.02, self.duration * max(300, self.width()) / n)

    def _zoom_at(self, factor: float, center_t: float | None = None):
        """以 center_t（默认视图中心）为锚缩放视图，factor>1 放大。"""
        if self.duration <= 0:
            return
        t0, t1 = self.view
        if center_t is None:
            center_t = (t0 + t1) / 2
        half = min(max((t1 - t0) / 2 / factor, self._min_window() / 2), self.duration / 2)
        c = min(max(center_t, half), self.duration - half)
        self.view = [c - half, c + half]
        self._rebuild_view_env()
        self.update()

    def _scroll_by(self, dt: float):
        """视图水平滚动 dt 秒（带边界钳制）。"""
        if self.duration <= 0:
            return
        t0, t1 = self.view
        span = t1 - t0
        t0 = min(max(0.0, t0 + dt), max(0.0, self.duration - span))
        self.view = [t0, t0 + span]
        self._rebuild_view_env()
        self.update()

    def _set_view(self, t0: float, t1: float):
        """设置视图窗口（秒），自动钳制到 [0, duration] 与最小窗口。"""
        if self.duration <= 0:
            return
        t0, t1 = min(t0, t1), max(t0, t1)
        span = min(max(t1 - t0, self._min_window()), self.duration)
        t0 = min(max(0.0, t0), max(0.0, self.duration - span))
        self.view = [t0, t0 + span]
        self._rebuild_view_env()
        self.update()

    def reset_view(self):
        """视图复位为全览。"""
        if self.duration > 0:
            self._set_view(0.0, self.duration)

    # ---------- 概览条 ----------
    def _ov_top(self) -> int:
        h = self.height()
        ruler_h = self._RULER_H if self.duration > 0 else 0
        return h - ruler_h - (self._OV_H if self.duration > 0 else 0)

    def _ov_x2t(self, x: float) -> float:
        if self.width() <= 0 or self.duration <= 0:
            return 0.0
        return min(max(0.0, x / self.width()), 1.0) * self.duration

    def _ov_t2x(self, t: float) -> float:
        if self.duration <= 0:
            return 0.0
        return t / self.duration * self.width()

    def _ov_hit(self, x: float, y: float) -> str:
        """概览条命中测试：pan=窗口内 / left,right=边缘手柄 / outside=窗口外 / ''=不在概览条。"""
        if self.duration <= 0 or y < self._ov_top():
            return ""
        wx0, wx1 = self._ov_t2x(self.view[0]), self._ov_t2x(self.view[1])
        if abs(x - wx0) <= self._EDGE:
            return "left"
        if abs(x - wx1) <= self._EDGE:
            return "right"
        if wx0 < x < wx1:
            return "pan"
        return "outside"

    def set_playhead(self, t: float, playing: bool = False):
        """更新播放头；播放中若越出视图自动跟随滚动。"""
        self.playhead = t
        if playing and self.duration > 0:
            t0, t1 = self.view
            if t < t0 or t > t1:
                span = t1 - t0
                nt0 = min(max(0.0, t - span * 0.08), max(0.0, self.duration - span))
                self.view = [nt0, nt0 + span]
                self._rebuild_view_env()
        self.update()

    # ---------- 坐标换算 ----------
    def _x2t(self, x: float) -> float:
        t0, t1 = self.view
        if t1 <= t0 or self.width() <= 0:
            return 0.0
        f = max(0.0, min(1.0, x / self.width()))
        return t0 + f * (t1 - t0)

    def _t2x(self, t: float) -> float:
        t0, t1 = self.view
        if t1 <= t0:
            return 0
        return (t - t0) / (t1 - t0) * self.width()

    # ---------- 绘制 ----------
    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        w, h = self.width(), self.height()
        ruler_h = self._RULER_H if self.duration > 0 else 0
        ov_h = self._OV_H if self.duration > 0 else 0
        wh = h - ruler_h - ov_h          # 波形区高度
        ov_top = wh + ruler_h            # 概览条顶部 y

        def shade(a, b, color):
            if a is None or b is None or self.duration <= 0:
                return
            x1, x2 = self._t2x(min(a, b)), self._t2x(max(a, b))
            p.fillRect(int(x1), 0, max(2, int(x2 - x1)), wh, color)

        # 选区着色（画在波形之下）
        shade(*(self.speaker_ref or (None, None)), QColor(80, 200, 120, 70))
        shade(*(self.feature_ref or (None, None)), QColor(80, 140, 255, 70))
        shade(*(self.sel or (None, None)), QColor(255, 160, 40, 80))

        # 波形（单声道 1 行 / 立体声 L+R 两行）
        if self.view_env:
            lane_h = wh / len(self.view_env)
            colors = (QColor(60, 220, 130), QColor(90, 180, 255))
            for li, env in enumerate(self.view_env):
                mid = li * lane_h + lane_h / 2
                p.setPen(QPen(QColor(255, 255, 255, 26), 1))
                p.drawLine(0, int(mid), w, int(mid))
                if len(env):
                    p.setPen(QPen(colors[min(li, 1)], 1))
                    n = len(env)
                    step = w / n
                    amp = lane_h * 0.47
                    for i in range(n):
                        lo, hi = env[i]
                        x = int(i * step)
                        p.drawLine(x, int(mid + lo * amp), x, int(mid + hi * amp))
                if self.stereo:
                    p.setPen(QColor(170, 170, 170))
                    p.drawText(4, int(li * lane_h + 13), "L" if li == 0 else "R")
        else:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "打开音频/视频后显示波形（拖拽选区 · 滚轮滚动 · Ctrl+滚轮缩放）")

        # 播放头（越出视图时在边缘画三角标记）
        p.setPen(QPen(QColor(255, 70, 70), 2))
        px = self._t2x(self.playhead)
        if 0 <= px <= w:
            p.drawLine(int(px), 0, int(px), wh)
        elif self.duration > 0:
            p.setBrush(QColor(255, 70, 70))
            ex = 0 if px < 0 else w
            d = 6 if px < 0 else -6
            p.drawPolygon([QPointF(ex, wh / 2 - 7), QPointF(ex, wh / 2 + 7), QPointF(ex + d, wh / 2)])

        # 拖拽预览线
        if self._drag_x is not None:
            p.setPen(QPen(QColor(255, 200, 60), 1))
            p.drawLine(self._drag_x, 0, self._drag_x, wh)

        # 时间刻度尺
        if ruler_h:
            p.setPen(QPen(QColor(70, 70, 82), 1))
            p.drawLine(0, wh, w, wh)
            span = self.view[1] - self.view[0]
            step = self._nice_step(span / max(1.0, w / 70.0))
            t = math.ceil(self.view[0] / step) * step
            p.setPen(QColor(155, 155, 155))
            while t <= self.view[1] + 1e-9:
                x = int(self._t2x(t))
                p.drawLine(x, wh, x, wh + 5)
                p.drawText(x + 3, wh + ruler_h - 5, self._fmt_ruler(t, step))
                t += step

        # 概览条（全长缩略 + 可拖拽视图窗口）
        if ov_h:
            p.fillRect(0, ov_top, w, ov_h, QColor(10, 10, 14))
            # 全长包络（每像素 min/max）
            if self.env_ch:
                env = self.env_ch[0]
                n = len(env)
                mid = ov_top + ov_h / 2
                amp = ov_h * 0.40
                p.setPen(QPen(QColor(70, 130, 170), 1))
                for x in range(w):
                    j = int(x / w * n)
                    if 0 <= j < n:
                        lo, hi = env[j]
                        p.drawLine(x, int(mid + lo * amp), x, int(mid + hi * amp))
            wx0, wx1 = self._ov_t2x(self.view[0]), self._ov_t2x(self.view[1])
            # 窗口外压暗
            p.fillRect(0, ov_top, int(max(0, wx0)), ov_h, QColor(0, 0, 0, 120))
            p.fillRect(int(min(w, wx1)), ov_top, int(max(0, w - wx1)), ov_h, QColor(0, 0, 0, 120))
            # 窗口边框与边缘手柄
            p.setPen(QPen(QColor(255, 200, 60), 1))
            p.drawRect(int(wx0), ov_top + 1, int(max(2, wx1 - wx0)), ov_h - 2)
            p.setPen(QPen(QColor(255, 220, 120), 3))
            for hx in (wx0, wx1):
                p.drawLine(int(hx), ov_top + 3, int(hx), ov_top + ov_h - 3)
            # 框选预览（窗口外拖拽中）
            if self._ov_mode == "select" and getattr(self, "_ov_sel_x", None) is not None:
                sx0, sx1 = sorted((self._ov_sel_x, self._ov_press_x))
                p.fillRect(int(sx0), ov_top, int(max(2, sx1 - sx0)), ov_h, QColor(255, 200, 60, 60))
                p.setPen(QPen(QColor(255, 200, 60), 1))
                p.drawRect(int(sx0), ov_top + 1, int(max(2, sx1 - sx0)), ov_h - 2)
            # 概览条上的播放头
            p.setPen(QPen(QColor(255, 70, 70), 1))
            ox = self._ov_t2x(self.playhead)
            p.drawLine(int(ox), ov_top, int(ox), ov_top + ov_h)
            # 分隔线与提示
            p.setPen(QPen(QColor(70, 70, 82), 1))
            p.drawLine(0, ov_top, w, ov_top)
            if self.view[1] - self.view[0] >= self.duration - 1e-6:
                p.setPen(QColor(110, 110, 120))
                p.drawText(6, ov_top + 13, "概览（全览）：拖窗口移动 · 拖边缘缩放 · 窗口外框选放大 · 双击复位")

        # 左上角信息（播放头时间 + 缩放状态）
        p.setPen(QColor(150, 150, 150))
        txt = self._fmt(self.playhead)
        if self.duration > 0 and self.view[1] - self.view[0] < self.duration - 1e-6:
            txt += f"   视图 {self.view[0]:.2f}~{self.view[1]:.2f}s"
        p.drawText(6, 14, txt)
        p.end()

    @staticmethod
    def _fmt(t: float) -> str:
        m = int(t // 60)
        s = t - m * 60
        return f"{m:02d}:{s:06.3f}"

    @classmethod
    def _nice_step(cls, target: float) -> float:
        for s in cls._STEPS:
            if s >= target:
                return s
        return cls._STEPS[-1]

    @staticmethod
    def _fmt_ruler(t: float, step: float) -> str:
        m = int(t // 60)
        s = t - m * 60
        if step >= 1:
            return f"{m:02d}:{int(s):02d}"
        if step >= 0.1:
            return f"{m:02d}:{s:04.1f}"
        return f"{m:02d}:{s:05.2f}"

    # ---------- 鼠标 / 滚轮 ----------
    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() != Qt.LeftButton:
            return
        x, y = ev.position().x(), ev.position().y()
        hit = self._ov_hit(x, y)
        if hit:
            # 概览条操作（与波形区点击/选区互斥）
            self._ov_mode = hit
            self._ov_press_x = int(x)
            self._ov_sel_x = int(x) if hit == "outside" else None
            self._ov_start_t = self._ov_x2t(x)
            if hit == "pan":
                self._ov_pan_anchor = (self._ov_x2t(x), self.view[0], self.view[1])
            self.setCursor(Qt.ClosedHandCursor)
            return
        self._press_x = x
        self._drag_x = int(x)

    def mouseMoveEvent(self, ev):  # noqa: N802
        x, y = ev.position().x(), ev.position().y()
        # 概览条拖拽中
        if self._ov_mode:
            t = self._ov_x2t(x)
            if self._ov_mode == "pan" and self._ov_pan_anchor:
                t0_anchor, v0, v1 = self._ov_pan_anchor
                self._set_view(v0 + (t - t0_anchor), v1 + (t - t0_anchor))
            elif self._ov_mode == "left":
                self._set_view(t, self.view[1])
            elif self._ov_mode == "right":
                self._set_view(self.view[0], t)
            elif self._ov_mode == "outside":
                self._ov_sel_x = int(x)
                self.update()
            return
        # 波形区拖拽（选区）
        if self._press_x is not None:
            self._drag_x = int(x)
            self.update()
            return
        # 无按键：按 hover 位置切换光标
        hit = self._ov_hit(x, y)
        if hit in ("left", "right"):
            self.setCursor(Qt.SplitHCursor)
        elif hit == "pan":
            self.setCursor(Qt.OpenHandCursor)
        elif hit == "outside":
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

    def mouseReleaseEvent(self, ev):  # noqa: N802
        # 概览条释放
        if self._ov_mode:
            if self._ov_mode == "outside":
                a = self._ov_x2t(min(self._ov_press_x, ev.position().x()))
                b = self._ov_x2t(max(self._ov_press_x, ev.position().x()))
                min_sel = max(self._min_window(), self.duration / max(1, self.width()) * 4)
                if b - a >= min_sel:
                    self._set_view(a, b)
            self._ov_mode = None
            self._ov_sel_x = None
            self._ov_pan_anchor = None
            self.unsetCursor()
            self.update()
            return
        if self._press_x is None:
            return
        x = ev.position().x()
        moved = abs(x - self._press_x) > 6
        if moved:
            a, b = sorted((self._x2t(self._press_x), self._x2t(x)))
            self.sel = (a, b)
            self.selection_drawn.emit(a, b)
        else:
            self.clicked.emit(self._x2t(x))
        self._press_x = None
        self._drag_x = None
        self.update()

    def mouseDoubleClickEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.LeftButton and \
                self._ov_hit(ev.position().x(), ev.position().y()):
            self.reset_view()  # 双击概览条复位全览

    def wheelEvent(self, ev):  # noqa: N802
        if self.duration <= 0:
            return
        dy, dx = ev.angleDelta().y(), ev.angleDelta().x()
        if ev.modifiers() & Qt.ControlModifier:
            self._zoom_at(1.25 ** (dy / 120.0), self._x2t(ev.position().x()))
        elif dy:
            self._scroll_by(-(dy / 120.0) * (self.view[1] - self.view[0]) * 0.15)
        elif dx:
            self._scroll_by(-(dx / 120.0) * (self.view[1] - self.view[0]) * 0.15)

    def resizeEvent(self, ev):  # noqa: N802
        super().resizeEvent(ev)
        if self.duration > 0:
            self._rebuild_view_env()
            self.update()


# ================================================================ 主窗口

class SourceFileGroup(QGroupBox):
    """可复用的源文件选择组：文件+目录混合，目录递归穷举子目录。

    各功能页签统一使用本组件获取源文件；列表留空时由调用方回退到当前打开的文件。
    """

    def __init__(self, title: str = "源文件（支持单个/多个文件与目录，目录递归穷举子目录；"
                                    "列表留空 = 当前打开的文件）", parent=None):
        super().__init__(title, parent)
        grid = QGridLayout(self)
        self.list = QListWidget()
        self.list.setMinimumHeight(84)
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        grid.addWidget(self.list, 0, 0, 1, 3)
        self.btn_add = QPushButton("添加文件…")
        self.btn_adddir = QPushButton("添加目录…")
        self.btn_del = QPushButton("移除所选")
        self.btn_clr = QPushButton("清空")
        grid.addWidget(self.btn_add, 1, 0)
        grid.addWidget(self.btn_adddir, 1, 1)
        grid.addWidget(self.btn_del, 1, 2)
        grid.addWidget(self.btn_clr, 2, 0)
        self.chk_recurse = QCheckBox("递归子目录")
        self.chk_recurse.setChecked(True)
        grid.addWidget(self.chk_recurse, 2, 1)
        self.btn_add.clicked.connect(self.add_files)
        self.btn_adddir.clicked.connect(self.add_dir)
        self.btn_del.clicked.connect(self.remove_selected)
        self.btn_clr.clicked.connect(self.list.clear)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择媒体文件", "", "媒体文件 (*)")
        existing = set(self.paths())
        for f in files:
            f = os.path.abspath(f)
            if f not in existing:
                self.list.addItem(f)
                existing.add(f)

    def add_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择媒体目录（递归扫描子目录）", "")
        if d:
            d = os.path.abspath(d)
            if d not in set(self.paths()):
                self.list.addItem(d)

    def remove_selected(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def paths(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def scan(self, exts=None) -> list[str]:
        from . import filetools
        return filetools.scan_inputs(self.paths(), exts or filetools.MEDIA_EXTS,
                                     recursive=self.chk_recurse.isChecked())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClearVoice — 音视频降噪 / 分割 / 合并工具")
        self.resize(1280, 820)

        self.src: str | None = None          # 当前媒体
        self.info: ft.MediaInfo | None = None
        self.fps: float = 0.0                # 当前视频帧率（帧步进用）
        self.wave_y: np.ndarray | None = None
        self.wave_sr = 16000
        self.speaker_ref: tuple[float, float] | None = None
        self.feature_ref: tuple[float, float] | None = None
        self.feature_ref_file: str | None = None
        self.worker: Worker | None = None

        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.audio_out.setVolume(1.0)
        self.player.setAudioOutput(self.audio_out)

        self._build_ui()
        self.player.setVideoOutput(self.video)  # 绑定视频画面输出（缺失会导致有声音无图像）
        self._load_settings()
        self._connect()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- 左侧：视频 + 波形 + 播放控制
        left = QVBoxLayout()
        self.video = QVideoWidget()
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.video, 6)

        self.wave = WaveformWidget()
        self.wave.setMinimumHeight(210)  # 立体声双行 + 时间刻度尺 + 概览条
        left.addWidget(self.wave, 2)

        ctl = QHBoxLayout()
        self.btn_open = QPushButton("打开文件")
        self.btn_play = QPushButton("播放")
        self.btn_stop = QPushButton("停止")
        self.rate = QComboBox()
        self.rate.addItems(["0.25x", "0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x", "3x", "4x"])
        self.rate.setCurrentText("1x")
        self.btn_mark_in = QPushButton("标记开始")
        self.btn_mark_out = QPushButton("标记结束")
        self.btn_sel_clear = QPushButton("清除选择")
        self.lbl_sel = QLabel("未选择区间")
        for w in (self.btn_open, self.btn_play, self.btn_stop, self.rate,
                  self.btn_mark_in, self.btn_mark_out, self.btn_sel_clear, self.lbl_sel):
            ctl.addWidget(w)
        ctl.addStretch(1)
        left.addLayout(ctl)

        act = QHBoxLayout()
        self.btn_mute_sel = QPushButton("消音所选段")
        self.btn_cut_sel = QPushButton("导出所选段")
        self.btn_set_spk = QPushButton("设为说话人参考")
        self.btn_set_feat = QPushButton("设为相似特征参考")
        for w in (self.btn_mute_sel, self.btn_cut_sel, self.btn_set_spk, self.btn_set_feat):
            act.addWidget(w)
        act.addStretch(1)
        # 帧步进（音画对齐辅助）
        self.btn_frame_back = QPushButton("« 帧")
        self.btn_frame_back.setFixedWidth(52)
        self.lbl_frame = QLabel("帧 --")
        self.lbl_frame.setStyleSheet("color:#8a8aa0;")
        self.btn_frame_fwd = QPushButton("帧 »")
        self.btn_frame_fwd.setFixedWidth(52)
        for w in (self.btn_frame_back, self.lbl_frame, self.btn_frame_fwd):
            act.addWidget(w)
        left.addLayout(act)

        self.time_label = QLabel("00:00.000 / 00:00.000")
        left.addWidget(self.time_label)
        root.addLayout(left, 7)

        # ---- 右侧：功能面板
        tabs = QTabWidget()
        root.addWidget(tabs, 5)

        # Tab1 消除杂音
        t1 = QWidget()
        f1 = QVBoxLayout(t1)
        self.grp_removal = SourceFileGroup(
            "处理源文件（支持单个/多个文件与目录，目录递归穷举子目录；列表留空 = 当前打开的文件）")
        f1.addWidget(self.grp_removal)
        g1 = QGroupBox("选择要消除的声音类型（可多选）")
        gv = QVBoxLayout(g1)
        self.chk: dict[str, QCheckBox] = {}
        for key, label in audio_ops.REMOVAL_TYPES:
            self.chk[key] = QCheckBox(label)
            gv.addWidget(self.chk[key])
        self.chk["noise"].setChecked(True)
        f1.addWidget(g1)
        pf = QGridLayout()
        pf.addWidget(QLabel("降噪强度"), 0, 0)
        self.spin_strength = QDoubleSpinBox(); self.spin_strength.setRange(0.1, 1.0); self.spin_strength.setValue(0.85)
        pf.addWidget(self.spin_strength, 0, 1)
        pf.addWidget(QLabel("说话人匹配阈值"), 1, 0)
        self.spin_spk_thr = QDoubleSpinBox(); self.spin_spk_thr.setRange(0.4, 0.99); self.spin_spk_thr.setSingleStep(0.01); self.spin_spk_thr.setValue(0.70)
        pf.addWidget(self.spin_spk_thr, 1, 1)
        f1.addLayout(pf)
        self.btn_apply_removal = QPushButton("应用消除（生成新文件）")
        f1.addWidget(self.btn_apply_removal)
        self.btn_modelscope = QPushButton("魔塔模型增强（可选，需安装 extra）")
        f1.addWidget(self.btn_modelscope)
        g1b = QGroupBox("去背景音乐（人声/伴奏分离，HDemucs 离线模型）")
        h1b = QHBoxLayout(g1b)
        h1b.addWidget(QLabel("保留伴奏比例"))
        self.spin_acc_keep = QDoubleSpinBox(); self.spin_acc_keep.setRange(0.0, 1.0); self.spin_acc_keep.setSingleStep(0.05); self.spin_acc_keep.setValue(0.0)
        h1b.addWidget(self.spin_acc_keep)
        self.chk_acc_out = QCheckBox("同时输出伴奏文件")
        h1b.addWidget(self.chk_acc_out)
        self.btn_music_remove = QPushButton("开始分离")
        h1b.addWidget(self.btn_music_remove)
        f1.addWidget(g1b)
        f1.addStretch(1)
        tabs.addTab(t1, "消除杂音")

        # Tab2 分割
        t2 = QWidget()
        f2 = QVBoxLayout(t2)
        self.grp_split = SourceFileGroup(
            "分割源文件（支持单个/多个文件与目录，目录递归穷举子目录；列表留空 = 分割当前打开的文件）")
        f2.addWidget(self.grp_split)
        g2 = QGroupBox("定长分割")
        v2 = QVBoxLayout(g2)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("每段长度(秒)"))
        self.spin_seg = QDoubleSpinBox(); self.spin_seg.setRange(0.5, 86400); self.spin_seg.setValue(60)
        h2.addWidget(self.spin_seg)
        self.btn_split_fixed = QPushButton("开始分割")
        h2.addWidget(self.btn_split_fixed)
        h2.addStretch(1)
        v2.addLayout(h2)
        f2.addWidget(g2)
        g3 = QGroupBox("按声音特征分割")
        v3 = QVBoxLayout(g3)
        h3 = QHBoxLayout()
        self.cmb_feat_split = QComboBox()
        self.cmb_feat_split.addItems(["静音边界", "说话人参考区间", "相似特征参考"])
        h3.addWidget(QLabel("特征")); h3.addWidget(self.cmb_feat_split)
        self.btn_split_feat = QPushButton("开始分割")
        h3.addWidget(self.btn_split_feat)
        v3.addLayout(h3)
        f2.addWidget(g3)
        g3b = QGroupBox("按关键词 / 正则分割（语音识别，需魔塔模型）")
        v3b = QVBoxLayout(g3b)
        self.ed_kw = QLineEdit()
        self.ed_kw.setPlaceholderText("关键词如: 大家好 ｜ 正则如: 第.{1,3}章（正则模式下生效）")
        v3b.addWidget(self.ed_kw)
        h3b = QHBoxLayout()
        self.cmb_kw_mode = QComboBox()
        self.cmb_kw_mode.addItems(["以关键字开头分割", "以关键字结束分割", "去掉关键字分割",
                                   "正则前分割", "正则后分割", "抹正则分割"])
        h3b.addWidget(self.cmb_kw_mode)
        h3b.addWidget(QLabel("前留白(s)"))
        self.spin_kw_pb = QDoubleSpinBox(); self.spin_kw_pb.setRange(0.0, 5.0); self.spin_kw_pb.setSingleStep(0.05); self.spin_kw_pb.setValue(0.10)
        h3b.addWidget(self.spin_kw_pb)
        h3b.addWidget(QLabel("后留白(s)"))
        self.spin_kw_pa = QDoubleSpinBox(); self.spin_kw_pa.setRange(0.0, 5.0); self.spin_kw_pa.setSingleStep(0.05); self.spin_kw_pa.setValue(0.10)
        h3b.addWidget(self.spin_kw_pa)
        self.btn_split_kw = QPushButton("开始分割")
        h3b.addWidget(self.btn_split_kw)
        v3b.addLayout(h3b)
        f2.addWidget(g3b)
        g3c = QGroupBox("分割输出选项")
        v3c = QVBoxLayout(g3c)
        self.chk_split_subdir = QCheckBox("为每个源文件创建子目录（按源文件名，分段文件放入其中）")
        self.chk_split_subdir.setChecked(True)
        v3c.addWidget(self.chk_split_subdir)
        h3c = QHBoxLayout()
        self.chk_split_namelen = QCheckBox("限制分段文件名长度（主干字符数，0=不限）")
        self.spin_split_namelen = QSpinBox(); self.spin_split_namelen.setRange(0, 200)
        self.spin_split_namelen.setValue(0); self.spin_split_namelen.setSingleStep(10)
        h3c.addWidget(self.chk_split_namelen)
        h3c.addWidget(self.spin_split_namelen)
        h3c.addStretch(1)
        v3c.addLayout(h3c)
        f2.addWidget(g3c)
        f2.addStretch(1)
        tabs.addTab(t2, "分割")

        # Tab3 合并 / 分离
        t3 = QWidget()
        f3 = QVBoxLayout(t3)
        g4 = QGroupBox("音视频合并")
        v4 = QGridLayout(g4)
        self.ed_video = QLineEdit(); self.ed_video.setPlaceholderText("视频文件")
        self.ed_audio = QLineEdit(); self.ed_audio.setPlaceholderText("音频文件")
        btn_v = QPushButton("..."); btn_a = QPushButton("...")
        btn_v.setFixedWidth(32); btn_a.setFixedWidth(32)
        v4.addWidget(QLabel("视频"), 0, 0); v4.addWidget(self.ed_video, 0, 1); v4.addWidget(btn_v, 0, 2)
        v4.addWidget(QLabel("音频"), 1, 0); v4.addWidget(self.ed_audio, 1, 1); v4.addWidget(btn_a, 1, 2)
        v4.addWidget(QLabel("音频偏移(秒, 正=延后)"), 2, 0)
        self.spin_offset = QDoubleSpinBox(); self.spin_offset.setRange(-3600, 3600); self.spin_offset.setValue(0.0)
        v4.addWidget(self.spin_offset, 2, 1)
        self.chk_shortest = QCheckBox("对齐到较短流(-shortest)"); self.chk_shortest.setChecked(True)
        v4.addWidget(self.chk_shortest, 3, 0, 1, 2)
        self.btn_merge_av = QPushButton("合并")
        v4.addWidget(self.btn_merge_av, 4, 0, 1, 3)
        btn_v.clicked.connect(lambda: self._pick_into(self.ed_video, "视频", ft.VIDEO_EXTS))
        btn_a.clicked.connect(lambda: self._pick_into(self.ed_audio, "音频", ft.AUDIO_EXTS))
        f3.addWidget(g4)
        self.grp_extract = SourceFileGroup(
            "分离源文件（支持单个/多个文件与目录；列表留空 = 当前打开的文件）")
        f3.addWidget(self.grp_extract)
        g5 = QGroupBox("分离音频（保留视频原始采样率 / 声道）")
        v5 = QGridLayout(g5)
        v5.addWidget(QLabel("输出格式"), 0, 0)
        self.cmb_extract_fmt = QComboBox()
        for _ext, _desc in ft.AUDIO_OUT_FORMATS.items():
            self.cmb_extract_fmt.addItem(f"{_desc}（*{_ext}）", _ext)
        v5.addWidget(self.cmb_extract_fmt, 0, 1)
        self.btn_extract = QPushButton("分离音频（批量处理上方列表）")
        v5.addWidget(self.btn_extract, 1, 0, 1, 2)
        f3.addWidget(g5)
        f3.addStretch(1)
        tabs.addTab(t3, "合并 / 分离")

        # Tab 格式转换（批量：视频转音频 / 音频转音频）
        tconv = QWidget()
        fconv = QVBoxLayout(tconv)
        gconv = QGroupBox("批量格式转换（视频转音频 / 音频转音频，支持文件与目录）")
        vconv = QGridLayout(gconv)
        self.grp_convert = SourceFileGroup()
        vconv.addWidget(self.grp_convert, 0, 0, 1, 3)
        self.chk_conv_keepstruct = QCheckBox("输出保持相对目录结构"); self.chk_conv_keepstruct.setChecked(True)
        vconv.addWidget(self.chk_conv_keepstruct, 1, 0, 1, 3)
        vconv.addWidget(QLabel("输出根目录"), 2, 0)
        self.ed_conv_outroot = QLineEdit(); self.ed_conv_outroot.setPlaceholderText("留空 = 输出到源文件同目录")
        btn_conv_outroot = QPushButton("浏览…"); btn_conv_outroot.setFixedWidth(64)
        vconv.addWidget(self.ed_conv_outroot, 2, 1)
        vconv.addWidget(btn_conv_outroot, 2, 2)
        vconv.addWidget(QLabel("输出格式"), 3, 0)
        self.cmb_conv_fmt = QComboBox()
        for _ext, _desc in ft.AUDIO_OUT_FORMATS.items():
            self.cmb_conv_fmt.addItem(f"{_desc}（*{_ext}）", _ext)
        vconv.addWidget(self.cmb_conv_fmt, 3, 1, 1, 2)
        vconv.addWidget(QLabel("码率"), 4, 0)
        self.cmb_conv_br = QComboBox()
        self.cmb_conv_br.setEditable(True)
        self.cmb_conv_br.addItems(["源 / 默认", "128k", "192k", "256k", "320k"])
        vconv.addWidget(self.cmb_conv_br, 4, 1, 1, 2)
        vconv.addWidget(QLabel("采样率"), 5, 0)
        self.cmb_conv_sr = QComboBox()
        self.cmb_conv_sr.addItems(["源", "8000", "16000", "22050", "32000", "44100", "48000"])
        vconv.addWidget(self.cmb_conv_sr, 5, 1, 1, 2)
        h_den = QHBoxLayout()
        self.chk_conv_denoise = QCheckBox("启用降噪（强度）")
        self.spin_conv_denoise = QDoubleSpinBox()
        self.spin_conv_denoise.setRange(0.1, 1.0); self.spin_conv_denoise.setSingleStep(0.05)
        self.spin_conv_denoise.setValue(0.85)
        h_den.addWidget(self.chk_conv_denoise)
        h_den.addWidget(self.spin_conv_denoise)
        h_den.addStretch(1)
        vconv.addLayout(h_den, 6, 0, 1, 3)
        h_name = QHBoxLayout()
        self.chk_conv_namelen = QCheckBox("限制输出文件名长度（主干字符数，0=不限）")
        self.spin_conv_namelen = QSpinBox(); self.spin_conv_namelen.setRange(0, 200)
        self.spin_conv_namelen.setValue(0); self.spin_conv_namelen.setSingleStep(10)
        h_name.addWidget(self.chk_conv_namelen)
        h_name.addWidget(self.spin_conv_namelen)
        h_name.addStretch(1)
        vconv.addLayout(h_name, 7, 0, 1, 3)
        self.btn_convert = QPushButton("开始转换")
        vconv.addWidget(self.btn_convert, 8, 0, 1, 3)
        lbl_conv_tip = QLabel("添加目录时递归穷举子目录中的音视频文件；输出在根目录下按源相对路径"
                              "重建目录结构；同名文件自动加序号；码率/采样率默认跟随源。")
        lbl_conv_tip.setWordWrap(True)
        vconv.addWidget(lbl_conv_tip, 9, 0, 1, 3)
        fconv.addWidget(gconv)
        fconv.addStretch(1)
        tabs.addTab(tconv, "格式转换")
        btn_conv_outroot.clicked.connect(
            lambda: self._pick_dir_into(self.ed_conv_outroot, "选择输出根目录"))

        # Tab 文件名/夹处理（批量重命名，对齐 xrename.bat）
        trn = QWidget()
        frn = QVBoxLayout(trn)
        grn = QGroupBox("批量文件名处理（字符串替换 / 保留子串 / 左切除 / 中间切除）")
        vrn = QGridLayout(grn)
        vrn.addWidget(QLabel("源目录"), 0, 0)
        self.ed_rn_dir = QLineEdit(); self.ed_rn_dir.setPlaceholderText("选择要批量重命名文件所在的目录")
        btn_rn_dir = QPushButton("浏览…"); btn_rn_dir.setFixedWidth(64)
        vrn.addWidget(self.ed_rn_dir, 0, 1)
        vrn.addWidget(btn_rn_dir, 0, 2)
        self.chk_rn_recurse = QCheckBox("递归子目录"); self.chk_rn_recurse.setChecked(True)
        vrn.addWidget(self.chk_rn_recurse, 1, 0)
        vrn.addWidget(QLabel("扩展名过滤（如 mp3,wav；空=全部）"), 1, 1)
        self.ed_rn_exts = QLineEdit(); self.ed_rn_exts.setPlaceholderText("mp3,wav,m4a")
        vrn.addWidget(self.ed_rn_exts, 1, 2)
        vrn.addWidget(QLabel("处理模式"), 2, 0)
        self.cmb_rn_mode = QComboBox()
        self.cmb_rn_mode.addItems(["字符串替换（查找→替换）",
                                   "保留子串 keep（从第 n 位起保留 m 位，m 负=到倒数）",
                                   "左切除 lcut（去掉前 m 位）",
                                   "中间切除 cut（删除第 n 到 m 位）"])
        vrn.addWidget(self.cmb_rn_mode, 2, 1, 1, 2)
        vrn.addWidget(QLabel("查找串"), 3, 0)
        self.ed_rn_find = QLineEdit()
        vrn.addWidget(self.ed_rn_find, 3, 1)
        self.ed_rn_repl = QLineEdit(); self.ed_rn_repl.setPlaceholderText("替换串（可空=删除）")
        vrn.addWidget(self.ed_rn_repl, 3, 2)
        vrn.addWidget(QLabel("起始位 n（0 基）"), 4, 0)
        self.spin_rn_n = QSpinBox(); self.spin_rn_n.setRange(-999, 999); self.spin_rn_n.setValue(0)
        vrn.addWidget(self.spin_rn_n, 4, 1)
        vrn.addWidget(QLabel("长度/位置 m（keep 负=到倒数|m|位）"), 4, 2)
        self.spin_rn_m = QSpinBox(); self.spin_rn_m.setRange(-999, 999); self.spin_rn_m.setValue(0)
        h_rn = QHBoxLayout()
        self.chk_rn_index = QCheckBox("重名自动加序号"); self.chk_rn_index.setChecked(True)
        h_rn.addWidget(self.chk_rn_index)
        self.chk_rn_namelen = QCheckBox("结果名超长截断（主干字符数，0=不限）")
        self.spin_rn_namelen = QSpinBox(); self.spin_rn_namelen.setRange(0, 200); self.spin_rn_namelen.setValue(0)
        h_rn.addWidget(self.chk_rn_namelen); h_rn.addWidget(self.spin_rn_namelen)
        h_rn.addStretch(1)
        vrn.addLayout(h_rn, 5, 0, 1, 3)
        h_rn_btn = QHBoxLayout()
        self.btn_rn_preview = QPushButton("预览（不执行）")
        self.btn_rn_apply = QPushButton("执行重命名")
        h_rn_btn.addWidget(self.btn_rn_preview); h_rn_btn.addWidget(self.btn_rn_apply)
        h_rn_btn.addStretch(1)
        vrn.addLayout(h_rn_btn, 6, 0, 1, 3)
        self.pte_rn = QPlainTextEdit(); self.pte_rn.setPlaceholderText("预览结果：旧文件名 → 新文件名")
        self.pte_rn.setReadOnly(True)
        vrn.addWidget(self.pte_rn, 7, 0, 1, 3)
        frn.addWidget(grn)
        tabs.addTab(trn, "文件名/夹处理")
        btn_rn_dir.clicked.connect(lambda: self._pick_dir_into(self.ed_rn_dir, "选择源目录"))

        # Tab4 时间轴
        t4 = QWidget()
        f4 = QVBoxLayout(t4)
        self.grp_timeline = SourceFileGroup(
            "处理源文件（变速/音画同步支持批量；试听预览与裁剪针对当前打开的文件；列表留空 = 当前打开的文件）")
        f4.addWidget(self.grp_timeline)
        g6 = QGroupBox("调整时间轴速度（视频+音频同步变速）")
        v6 = QGridLayout(g6)
        v6.addWidget(QLabel("倍速"), 0, 0)
        self.spin_speed = QDoubleSpinBox(); self.spin_speed.setRange(0.1, 8.0); self.spin_speed.setSingleStep(0.1); self.spin_speed.setValue(1.0)
        v6.addWidget(self.spin_speed, 0, 1)
        self.btn_speed = QPushButton("应用变速")
        v6.addWidget(self.btn_speed, 0, 2)
        f4.addWidget(g6)
        g7b = QGroupBox("音画同步微调（视频音频不同步时，把音频提前/推迟）")
        v7b = QGridLayout(g7b)
        v7b.addWidget(QLabel("音频偏移"), 0, 0)
        self.spin_av_sync = QDoubleSpinBox(); self.spin_av_sync.setRange(-10.0, 10.0)
        self.spin_av_sync.setSingleStep(0.05); self.spin_av_sync.setValue(0.0)
        self.spin_av_sync.setSuffix(" 秒"); self.spin_av_sync.setMinimumWidth(100)
        v7b.addWidget(self.spin_av_sync, 0, 1)
        lbl_sync_tip = QLabel("正值 = 音频延后（推迟播放）· 负值 = 音频提前")
        lbl_sync_tip.setWordWrap(True)
        v7b.addWidget(lbl_sync_tip, 0, 2)
        self.btn_av_preview = QPushButton("试听预览（当前位置起 20 秒）")
        v7b.addWidget(self.btn_av_preview, 1, 0)
        self.btn_av_restore = QPushButton("恢复播放原文件")
        v7b.addWidget(self.btn_av_restore, 1, 1)
        self.btn_av_apply = QPushButton("应用到文件（生成新文件）")
        v7b.addWidget(self.btn_av_apply, 1, 2)
        f4.addWidget(g7b)
        g7 = QGroupBox("裁剪")
        v7 = QVBoxLayout(g7)
        self.btn_trim = QPushButton("裁剪为所选区间")
        v7.addWidget(self.btn_trim)
        f4.addWidget(g7)
        f4.addStretch(1)
        tabs.addTab(t4, "时间轴")

        # Tab5 特征剔除
        t8 = QWidget()
        f8 = QVBoxLayout(t8)
        self.grp_feature = SourceFileGroup(
            "处理源文件（支持单个/多个文件与目录，目录递归穷举子目录；列表留空 = 当前打开的文件）")
        f8.addWidget(self.grp_feature)
        g8 = QGroupBox("按语言文字特征剔除（需要魔塔 ASR 模型）")
        v8 = QVBoxLayout(g8)
        self.ed_keywords = QLineEdit(); self.ed_keywords.setPlaceholderText("输入关键词，用逗号分隔，例如: 保密,内部资料")
        v8.addWidget(self.ed_keywords)
        h8 = QHBoxLayout()
        h8.addWidget(QLabel("剔除方式"))
        self.cmb_text_mode = QComboBox(); self.cmb_text_mode.addItems(["静音该段", "剪切该段"])
        h8.addWidget(self.cmb_text_mode)
        self.btn_text_remove = QPushButton("开始剔除")
        h8.addWidget(self.btn_text_remove)
        v8.addLayout(h8)
        f8.addWidget(g8)
        g9 = QGroupBox("按相似音频特征剔除（参考=标记区间或参考文件）")
        v9 = QVBoxLayout(g9)
        h9 = QHBoxLayout()
        h9.addWidget(QLabel("相似阈值"))
        self.spin_sim_thr = QDoubleSpinBox(); self.spin_sim_thr.setRange(0.4, 0.99); self.spin_sim_thr.setSingleStep(0.01); self.spin_sim_thr.setValue(0.78)
        h9.addWidget(self.spin_sim_thr)
        h9.addWidget(QLabel("剔除方式"))
        self.cmb_sim_mode = QComboBox(); self.cmb_sim_mode.addItems(["静音该段", "剪切该段"])
        h9.addWidget(self.cmb_sim_mode)
        self.btn_sim_remove = QPushButton("开始剔除")
        h9.addWidget(self.btn_sim_remove)
        v9.addLayout(h9)
        h9b = QHBoxLayout()
        self.btn_feat_file = QPushButton("导入参考音频文件")
        self.lbl_feat_file = QLabel("未导入")
        h9b.addWidget(self.btn_feat_file); h9b.addWidget(self.lbl_feat_file); h9b.addStretch(1)
        v9.addLayout(h9b)
        f8.addWidget(g9)
        f8.addStretch(1)
        tabs.addTab(t8, "特征剔除")

        # Tab6 语音转文字
        t9 = QWidget()
        f9 = QVBoxLayout(t9)
        self.grp_asr = SourceFileGroup(
            "识别源文件（支持单个/多个文件与目录，目录递归穷举子目录；列表留空 = 当前打开的文件）")
        f9.addWidget(self.grp_asr)
        g11 = QGroupBox("输出格式（可多选）")
        v11 = QVBoxLayout(g11)
        self.chk_out_txt = QCheckBox("文本文档 (*.txt)"); self.chk_out_txt.setChecked(True)
        self.chk_out_srt = QCheckBox("字幕文件 (*.srt)"); self.chk_out_srt.setChecked(True)
        self.chk_out_lrc = QCheckBox("歌词文件 (*.lrc)"); self.chk_out_lrc.setChecked(True)
        v11.addWidget(self.chk_out_txt); v11.addWidget(self.chk_out_srt); v11.addWidget(self.chk_out_lrc)
        f9.addWidget(g11)
        self.btn_asr_run = QPushButton("开始识别并导出")
        f9.addWidget(self.btn_asr_run)
        f9.addStretch(1)
        tabs.addTab(t9, "语音转文字")

        self.btn_asr_run.clicked.connect(self.run_asr_export)

        # Tab 翻译 / TTS
        t11 = QWidget()
        f11 = QVBoxLayout(t11)
        g13 = QGroupBox("音频翻译（语音识别 → 翻译 → 双语导出）")
        v13 = QGridLayout(g13)
        self.ed_tr_src = QLineEdit(); self.ed_tr_src.setPlaceholderText("要翻译的音频/视频文件")
        btn_tr_pick = QPushButton("浏览…"); btn_tr_pick.setFixedWidth(64)
        btn_tr_cur = QPushButton("使用当前文件"); btn_tr_cur.setFixedWidth(110)
        v13.addWidget(self.ed_tr_src, 0, 0)
        v13.addWidget(btn_tr_pick, 0, 1)
        v13.addWidget(btn_tr_cur, 0, 2)
        v13.addWidget(QLabel("翻译方向"), 1, 0)
        self.cmb_tr_pair = QComboBox(); self.cmb_tr_pair.addItems(list(translator.PAIRS.keys()))
        v13.addWidget(self.cmb_tr_pair, 1, 1)
        self.btn_translate = QPushButton("开始识别并翻译")
        v13.addWidget(self.btn_translate, 2, 0, 1, 3)
        lbl_tr = QLabel("输出: <源>.翻译.txt（双语对照）/ <源>.双语.srt / <源>.双语.lrc")
        lbl_tr.setWordWrap(True)
        v13.addWidget(lbl_tr, 3, 0, 1, 3)
        f11.addWidget(g13)
        g14 = QGroupBox("文字转语音（中 / 英 / 粤）")
        v14 = QGridLayout(g14)
        self.pte_tts = QPlainTextEdit(); self.pte_tts.setPlaceholderText("输入要合成的文本…")
        self.pte_tts.setMaximumHeight(110)
        v14.addWidget(self.pte_tts, 0, 0, 1, 3)
        v14.addWidget(QLabel("音色"), 1, 0)
        self.cmb_tts_voice = QComboBox()
        for _name, _v in tts_mod.VOICES:
            self.cmb_tts_voice.addItem(_name)
        v14.addWidget(self.cmb_tts_voice, 1, 1, 1, 2)
        self.ed_tts_out = QLineEdit(); self.ed_tts_out.setPlaceholderText("输出 MP3 路径（留空自动命名到 tts_out\\）")
        btn_tts_pick = QPushButton("浏览…"); btn_tts_pick.setFixedWidth(64)
        v14.addWidget(self.ed_tts_out, 2, 0)
        v14.addWidget(btn_tts_pick, 2, 1)
        self.btn_tts = QPushButton("开始合成")
        v14.addWidget(self.btn_tts, 3, 0, 1, 3)
        f11.addWidget(g14)
        f11.addStretch(1)
        tabs.addTab(t11, "翻译 / TTS")

        btn_tr_pick.clicked.connect(self._pick_tr_src)
        btn_tr_cur.clicked.connect(self._tr_use_current)
        self.btn_translate.clicked.connect(self.run_translate)
        btn_tts_pick.clicked.connect(self._pick_tts_out)
        self.btn_tts.clicked.connect(self.run_tts)

        # Tab7 设置
        t10 = QWidget()
        f10 = QVBoxLayout(t10)
        g12 = QGroupBox("语音识别（funasr / 魔塔）")
        v12 = QGridLayout(g12)
        v12.addWidget(QLabel("模型根目录"), 0, 0)
        self.ed_model_dir = QLineEdit()
        self.ed_model_dir.setPlaceholderText(r"如 D:\funasrModel（自动查找 hub\iic 下的模型）")
        v12.addWidget(self.ed_model_dir, 0, 1)
        btn_md = QPushButton("浏览…"); btn_md.setFixedWidth(64)
        v12.addWidget(btn_md, 0, 2)
        v12.addWidget(QLabel("运行设备"), 1, 0)
        self.cmb_device = QComboBox(); self.cmb_device.addItems(["cpu", "cuda"])
        v12.addWidget(self.cmb_device, 1, 1)
        v12.addWidget(QLabel("分离模型目录"), 2, 0)
        self.ed_demucs_dir = QLineEdit()
        self.ed_demucs_dir.setPlaceholderText("可选：HDemucs 权重目录（留空用内置缓存）")
        v12.addWidget(self.ed_demucs_dir, 2, 1)
        btn_dm = QPushButton("浏览…"); btn_dm.setFixedWidth(64)
        v12.addWidget(btn_dm, 2, 2)
        f10.addWidget(g12)
        g12b = QGroupBox("翻译 / TTS 服务（OpenAI 兼容：本地大模型或线上 API）")
        v15 = QGridLayout(g12b)
        v15.addWidget(QLabel("翻译后端"), 0, 0)
        self.cmb_tr_backend = QComboBox()
        self.cmb_tr_backend.addItems(["本地小模型（M2M100，离线）", "OpenAI 兼容大模型 API"])
        v15.addWidget(self.cmb_tr_backend, 0, 1)
        v15.addWidget(QLabel("TTS 后端"), 1, 0)
        self.cmb_tts_backend = QComboBox()
        self.cmb_tts_backend.addItems(["edge-tts（线上免费）", "OpenAI 兼容 TTS API"])
        v15.addWidget(self.cmb_tts_backend, 1, 1)
        v15.addWidget(QLabel("API Base URL"), 2, 0)
        self.ed_api_base = QLineEdit()
        self.ed_api_base.setPlaceholderText("本地: http://localhost:11434/v1 ｜ 线上: https://api.xxx.com/v1")
        v15.addWidget(self.ed_api_base, 2, 1)
        v15.addWidget(QLabel("API Key"), 3, 0)
        self.ed_api_key = QLineEdit(); self.ed_api_key.setEchoMode(QLineEdit.Password)
        v15.addWidget(self.ed_api_key, 3, 1)
        v15.addWidget(QLabel("对话模型名"), 4, 0)
        self.ed_api_model = QLineEdit(); self.ed_api_model.setPlaceholderText("如 qwen2.5:7b / gpt-4o-mini")
        v15.addWidget(self.ed_api_model, 4, 1)
        v15.addWidget(QLabel("翻译模型目录"), 5, 0)
        self.ed_tr_model_dir = QLineEdit()
        self.ed_tr_model_dir.setPlaceholderText("可选：M2M100 权重目录（留空自动下载）")
        v15.addWidget(self.ed_tr_model_dir, 5, 1)
        btn_trmdir = QPushButton("浏览…"); btn_trmdir.setFixedWidth(64)
        v15.addWidget(btn_trmdir, 5, 2)
        f10.addWidget(g12b)
        self.btn_save_settings = QPushButton("保存设置")
        f10.addWidget(self.btn_save_settings)
        self.lbl_cfg_path = QLabel(f"配置文件: {config.CONFIG_PATH}")
        f10.addWidget(self.lbl_cfg_path)
        f10.addStretch(1)
        tabs.addTab(t10, "设置")

        btn_md.clicked.connect(self._pick_model_dir)
        btn_dm.clicked.connect(self._pick_demucs_dir)
        btn_trmdir.clicked.connect(self._pick_tr_model_dir)
        self.btn_save_settings.clicked.connect(self.save_settings)

        # ---- 底部：进度 + 日志（放在左栏底部）
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        left.addWidget(self.bar)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(140)
        left.addWidget(self.log)

    def _connect(self):
        self.btn_open.clicked.connect(self.open_file)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.player.stop)
        self.rate.currentTextChanged.connect(self._set_rate)
        self.btn_mark_in.clicked.connect(lambda: self._mark("in"))
        self.btn_mark_out.clicked.connect(lambda: self._mark("out"))
        self.btn_sel_clear.clicked.connect(self._clear_sel)
        self.wave.clicked.connect(self._seek)
        self.wave.selection_drawn.connect(self._drag_sel)
        self.player.positionChanged.connect(self._pos_changed)
        self.player.mediaStatusChanged.connect(lambda s: self.log_append(f"媒体状态: {s}") if s in (7, 8) else None)
        self.player.errorOccurred.connect(lambda e, s: self.log_append(f"播放错误: {s}"))

        self.btn_mute_sel.clicked.connect(self.mute_selection)
        self.btn_cut_sel.clicked.connect(self.cut_selection)
        self.btn_set_spk.clicked.connect(self.set_speaker_ref)
        self.btn_set_feat.clicked.connect(self.set_feature_ref)
        self.btn_frame_back.clicked.connect(lambda: self._frame_step(-1))
        self.btn_frame_fwd.clicked.connect(lambda: self._frame_step(1))

        self.btn_apply_removal.clicked.connect(self.apply_removals)
        self.btn_modelscope.clicked.connect(self.apply_modelscope_enhance)
        self.btn_music_remove.clicked.connect(self.remove_background_music)
        self.btn_split_fixed.clicked.connect(self.split_fixed)
        self.btn_split_feat.clicked.connect(self.split_by_feature)
        self.btn_split_kw.clicked.connect(self.split_keyword)
        self.btn_merge_av.clicked.connect(self.merge_av)
        self.btn_extract.clicked.connect(self.extract_audio)
        self.btn_convert.clicked.connect(self.convert_media_batch)
        self.btn_rn_preview.clicked.connect(self.rename_preview)
        self.btn_rn_apply.clicked.connect(self.rename_apply)
        self.btn_speed.clicked.connect(self.change_speed)
        self.btn_av_preview.clicked.connect(self.preview_av_sync)
        self.btn_av_restore.clicked.connect(self.restore_av_preview)
        self.btn_av_apply.clicked.connect(self.apply_av_sync)
        self.btn_trim.clicked.connect(self.trim_selection)
        self.btn_text_remove.clicked.connect(self.remove_text_feature)
        self.btn_sim_remove.clicked.connect(self.remove_similar_feature)
        self.btn_feat_file.clicked.connect(self.import_feature_ref_file)

    # ---------------------------------------------------------------- 工具
    def log_append(self, msg: str):
        self.log.appendPlainText(msg)

    def out_path(self, suffix: str, ext: str | None = None) -> str:
        base, e = os.path.splitext(self.src or "untitled")
        return base + suffix + (ext or e)

    def _pick_into(self, edit: QLineEdit, title: str, exts: set[str]):
        f, _ = QFileDialog.getOpenFileName(self, f"选择{title}", "", "媒体文件 (*)")
        if f:
            edit.setText(f)

    def _pick_dir_into(self, edit: QLineEdit, title: str):
        d = QFileDialog.getExistingDirectory(self, title, "")
        if d:
            edit.setText(os.path.abspath(d))

    def _sel_range(self) -> tuple[float, float] | None:
        s = self.wave.sel
        if not s or s[1] - s[0] < 0.05:
            QMessageBox.information(self, "提示", "请先标记/框选一个区间（标记开始/结束 或 在波形上拖拽）")
            return None
        return s

    def _require_src(self) -> bool:
        if not self.src:
            QMessageBox.information(self, "提示", "请先打开音频或视频文件")
            return False
        return True

    def _collect_source(self, group: "SourceFileGroup") -> list[str] | None:
        """统一源收集：组内文件/目录 → 递归扫描媒体文件；列表为空回退当前打开的文件。"""
        from . import filetools
        inputs = group.paths()
        if inputs:
            missing = [p for p in inputs if not os.path.exists(p)]
            if missing:
                QMessageBox.warning(self, "提示", "以下路径不存在：\n" + "\n".join(missing[:5]))
                return None
            files = group.scan(filetools.MEDIA_EXTS)
            if not files:
                QMessageBox.information(self, "提示", "所选路径中没有可处理的音视频文件")
                return None
            return files
        if not self.src:
            QMessageBox.information(self, "提示", "请先添加源文件/目录，或打开一个媒体文件")
            return None
        return [os.path.abspath(self.src)]

    @staticmethod
    def _batch_loop(title: str, files: list[str], fn, progress_cb=None) -> dict:
        """批量执行：fn(src, cb) 返回 ("ok"|"skip", 说明)；抛异常记为失败。

        cb(percent, msg) 为单文件内部进度回调，自动换算为总进度 [i/N]。
        """
        done, failed, skipped = [], [], []
        total = max(1, len(files))
        for i, src in enumerate(files):
            name = os.path.basename(src)

            def cb(p, m="", _i=i):
                progress_cb(min(99, int(100 * _i / total) + int(p / total)),
                            f"[{_i + 1}/{len(files)}] {name}: {m}")

            try:
                status, info = fn(src, cb)
                (done if status == "ok" else skipped).append((name, str(info)))
            except Exception as e:
                failed.append((name, str(e)))
            progress_cb(int(100 * (i + 1) / total), f"{title} {i + 1}/{len(files)} 完成")
        progress_cb(100, f"{title}完成")
        return MainWindow._batch_report(title, done, failed, skipped)

    @staticmethod
    def _batch_report(title: str, done: list, failed: list, skipped: list | None = None) -> dict:
        skipped = skipped or []
        lines = [f"{title}：共 {len(done) + len(failed) + len(skipped)} 个文件，"
                 f"成功 {len(done)}，跳过 {len(skipped)}，失败 {len(failed)}", ""]
        for item in done:
            lines.append(f"✓ {item[0]}  {item[1]}")
        for name, info in skipped:
            lines.append(f"○ {name}  {info}")
        for name, err in failed:
            lines.append(f"✗ {name}  {err}")
        return {"output": "", "report": "\n".join(lines)}

    def _run(self, name: str, fn, *args, on_done=None, **kwargs):
        if getattr(self, "_busy", False):
            QMessageBox.information(self, "提示", "有任务正在进行中，请稍候")
            return
        self._busy = True
        self._on_done_cb = on_done
        self.bar.setValue(0)
        self.statusBar().showMessage(f"执行中: {name}")
        self.worker = Worker(name, fn, *args, **kwargs)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._handle_done)
        self.worker.failed.connect(self._handle_failed)
        self.worker.start()

    def _handle_done(self, tag: str, res):
        self._busy = False
        cb = getattr(self, "_on_done_cb", None)
        if cb:
            cb(res)
        else:
            self._default_done(tag, res)

    def _handle_failed(self, tag: str, err: str):
        self._busy = False
        self._on_failed(tag, err)

    def _on_progress(self, pct: int, msg: str):
        self.bar.setValue(pct)
        if msg:
            self.statusBar().showMessage(msg)

    def _on_failed(self, tag: str, err: str):
        self.statusBar().showMessage(f"失败: {tag}")
        self.log_append(f"[{tag}] 出错:\n{err}")
        QMessageBox.critical(self, "任务失败", err.splitlines()[0][:300])

    def _default_done(self, tag: str, res):
        self.statusBar().showMessage(f"完成: {tag}")
        out = None
        if isinstance(res, dict):
            out = res.get("output")
            if res.get("report"):
                self.log_append(f"[{tag}] 报告: {res['report']}")
            if res.get("matched"):
                self.log_append(f"[{tag}] 命中内容:\n" + "\n".join(res["matched"][:20]))
        elif isinstance(res, str) and os.path.isfile(res):
            out = res
        elif isinstance(res, list):
            self.log_append(f"[{tag}] 共输出 {len(res)} 个文件:\n" + "\n".join(res[:50]))
        if out and os.path.isfile(out):
            self._offer_load(out)

    def _offer_load(self, path: str):
        r = QMessageBox.question(self, "完成", f"已生成:\n{path}\n\n是否加载到播放器？",
                                 QMessageBox.Yes | QMessageBox.No)
        if r == QMessageBox.Yes:
            self.load_file(path)

    # ---------------------------------------------------------------- 打开/播放
    def open_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "打开媒体", "", "媒体文件 (*)")
        if f:
            self.load_file(f)

    def load_file(self, path: str):
        self.player.stop()
        self.src = os.path.abspath(path)
        try:
            self.info = ft.media_info(self.src)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"读取媒体信息失败: {e}")
            return
        self.log_append(f"打开: {self.src}  ({self.info.duration:.2f}s, "
                        f"视频={self.info.has_video}, 音频={self.info.has_audio})")
        self.fps = self.info.fps if self.info.has_video else 0.0
        if self.fps > 0:
            self.lbl_frame.setText(f"帧 0 @ {self.fps:g}fps")
        else:
            self.lbl_frame.setText("帧 --")
        self.player.setSource(QUrl.fromLocalFile(self.src))
        self.wave.clear_all()
        self.wave.duration = self.info.duration
        self.wave.view = [0.0, self.info.duration]
        self.speaker_ref = None
        self.feature_ref = None
        self._mark_in = None
        self._mark_out = None
        self._load_waveform()
        self._update_sel_label()

    def _load_waveform(self):
        if not self.src:
            return
        self.statusBar().showMessage("正在生成波形…")
        self._run("波形", self._job_waveform, self.src, on_done=self._after_waveform)

    @staticmethod
    def _job_waveform(src: str, progress_cb=None):
        return audio_ops.load_wav(src, 16000, mono=False)  # 立体声双行显示

    def _after_waveform(self, res):
        y, sr = res
        self.wave_sr = sr
        self.wave_y = y
        self.wave.set_data(y, sr, self.info.duration if self.info else len(y) / sr)
        self.statusBar().showMessage("就绪")

    def _pos_changed(self, ms: int):
        playing = self.player.playbackState() == QMediaPlayer.PlayingState
        self.wave.set_playhead(ms / 1000.0, playing)
        d = self.player.duration()
        if d > 0:
            self.time_label.setText(
                f"{self._fmt_ms(ms)} / {self._fmt_ms(d)}  [{self.rate.currentText()}]")
        if self.fps > 0:
            self.lbl_frame.setText(f"帧 {int(round(ms * self.fps / 1000))} @ {self.fps:g}fps")

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        s = ms / 1000.0
        return f"{int(s // 60):02d}:{s % 60:06.3f}"

    def _frame_step(self, direction: int):
        """按帧步进播放位置（音画对齐辅助）：direction=-1 上一帧 / +1 下一帧。"""
        if self.fps <= 0:
            QMessageBox.information(self, "提示", "当前文件无视频流，帧步进不可用")
            return
        step_ms = 1000.0 / self.fps
        pos = self.player.position() + direction * step_ms
        self.player.setPosition(int(max(0.0, min(pos, self.player.duration()))))

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_play.setText("播放")
        else:
            self.player.play()
            self.btn_play.setText("暂停")

    def _set_rate(self, text: str):
        r = float(text.rstrip("x"))
        self.player.setPlaybackRate(r)

    def _seek(self, t: float):
        self.player.setPosition(int(t * 1000))

    def _mark(self, which: str):
        t = self.player.position() / 1000.0
        if which == "in":
            self._mark_in = t
        else:
            self._mark_out = t
        if getattr(self, "_mark_in", None) is not None and getattr(self, "_mark_out", None) is not None:
            a, b = sorted((self._mark_in, self._mark_out))
            if b - a > 0.05:
                self.wave.sel = (a, b)
                self._mark_in = self._mark_out = None
        else:
            cur = self.wave.sel or (t, t)
            self.wave.sel = (t, cur[1]) if which == "in" else (cur[0], t)
        self._update_sel_label()
        self.wave.update()
        self.log_append(f"标记{'开始' if which == 'in' else '结束'}: {t:.3f}s")

    def _drag_sel(self, a: float, b: float):
        self._update_sel_label()

    def _clear_sel(self):
        self.wave.sel = None
        self._update_sel_label()
        self.wave.update()

    def _update_sel_label(self):
        s = self.wave.sel
        self.lbl_sel.setText(f"已选: {s[0]:.2f}s ~ {s[1]:.2f}s" if s else "未选择区间")

    # ---------------------------------------------------------------- 区间操作
    def mute_selection(self):
        if not self._require_src():
            return
        r = self._sel_range()
        if not r:
            return
        ext = ".mp4" if ft.is_video(self.src) else ".m4a"
        out = self.out_path("_静音段", ext)
        self._run("消音所选段", lambda src, o, s, e, progress_cb=None: ft.mute_ranges(src, o, [(s, e)]),
                  self.src, out, r[0], r[1])

    def cut_selection(self):
        if not self._require_src():
            return
        r = self._sel_range()
        if not r:
            return
        out = self.out_path(f"_片段{r[0]:.1f}-{r[1]:.1f}")
        self._run("导出所选段", lambda src, o, s, e, progress_cb=None: ft.cut_range(src, o, s, e),
                  self.src, out, r[0], r[1])

    def set_speaker_ref(self):
        r = self._sel_range()
        if not r:
            return
        self.speaker_ref = r
        self.wave.speaker_ref = r
        self.wave.update()
        self.log_append(f"说话人参考区间: {r[0]:.2f}~{r[1]:.2f}s")

    def set_feature_ref(self):
        r = self._sel_range()
        if not r:
            return
        self.feature_ref = r
        self.wave.feature_ref = r
        self.wave.update()
        self.log_append(f"相似特征参考区间: {r[0]:.2f}~{r[1]:.2f}s")

    def import_feature_ref_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择参考音频", "", "音频文件 (*)")
        if f:
            self.feature_ref_file = f
            self.lbl_feat_file.setText(os.path.basename(f))

    # ---------------------------------------------------------------- 消除
    def _selected_removals(self) -> list[str]:
        return [k for k, cb in self.chk.items() if cb.isChecked()]

    def apply_removals(self):
        files = self._collect_source(self.grp_removal)
        if not files:
            return
        kinds = self._selected_removals()
        if not kinds:
            QMessageBox.information(self, "提示", "请至少勾选一种要消除的声音")
            return
        if "speaker" in kinds and self.speaker_ref is None:
            QMessageBox.information(self, "提示", "消除指定说话人前，请先播放并标记其声音区间，点击『设为说话人参考』")
            return
        strength = float(self.spin_strength.value())
        thr = float(self.spin_spk_thr.value())
        names = "+".join(dict(audio_ops.REMOVAL_TYPES)[k] for k in kinds)
        # 说话人参考音频取自标记所在的当前文件（批量时只提取一次，应用到全部源文件）
        ref_src = os.path.abspath(self.src) if ("speaker" in kinds and self.src) else None
        self._run(f"消除({names})", self._job_removals_batch, files, kinds, strength, thr,
                  self.speaker_ref, ref_src, names, on_done=self._reload_wave)

    def _reload_wave(self, res=None):
        if res:
            self._default_done("消除", res)
        self._load_waveform()

    @staticmethod
    def _job_removals_batch(files, kinds, strength, thr, spk_ref, ref_src, names,
                            progress_cb=None):
        from . import filetools
        import soundfile as sf
        ref_audio = None
        if "speaker" in kinds and spk_ref is not None and ref_src:
            tmp = ref_src + ".ref.wav"
            ft.extract_audio(ref_src, tmp, sr=16000, start=spk_ref[0], end=spk_ref[1])
            ref_audio, _ = sf.read(tmp, dtype="float32")
            if ref_audio.ndim > 1:
                ref_audio = ref_audio.mean(axis=1)
            try:
                os.remove(tmp)
            except OSError:
                pass
        suffix = f"_消除_{names}"

        def per(src, cb):
            out_wav = filetools.plan_output(src, "", "", suffix=suffix, ext=".wav")
            out = filetools.plan_output(src, "", "", suffix=suffix)
            res = MainWindow._job_removals(src, out, out_wav, kinds, strength, thr,
                                           spk_ref, progress_cb=cb, ref_audio=ref_audio)
            return "ok", f"→ {os.path.basename(res['output'])}"

        return MainWindow._batch_loop(f"消除({names})", files, per, progress_cb)

    @staticmethod
    def _job_removals(src, out, out_wav, kinds, strength, thr, spk_ref,
                      progress_cb=None, ref_audio=None):
        def prog(p, m):
            progress_cb(p, m)
        y, sr = audio_ops.load_wav(src, 16000)
        ref = None
        if kinds and "speaker" in kinds and spk_ref is not None:
            if ref_audio is not None:
                ref = ref_audio
            else:
                import soundfile as sf
                tmp = out_wav + ".ref.wav"
                ft.extract_audio(src, tmp, sr=16000, start=spk_ref[0], end=spk_ref[1])
                ref, _ = sf.read(tmp, dtype="float32")
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        y2, report = audio_ops.process_removals(y, sr, kinds, strength,
                                                speaker_ref=ref, speaker_threshold=thr,
                                                progress=prog)
        audio_ops.save_wav(out_wav, y2, sr)
        if ft.is_video(src):
            out = os.path.splitext(out)[0] + ".mp4"
            ft.mux_replace_audio(src, out_wav, out)
        else:
            out = ft.encode_wav_to_media(out_wav, out)
        return {"output": out, "report": report}

    def apply_modelscope_enhance(self):
        files = self._collect_source(self.grp_removal)
        if not files:
            return
        self._run("魔塔增强", self._job_modelscope_batch, files)

    @staticmethod
    def _job_modelscope_batch(files, progress_cb=None):
        from . import filetools

        def per(src, cb):
            out_wav = filetools.plan_output(src, "", "", suffix="_魔塔增强", ext=".wav")
            out = filetools.plan_output(src, "", "", suffix="_魔塔增强")
            r = MainWindow._job_modelscope(src, out_wav, out, progress_cb=cb)
            return "ok", f"→ {os.path.basename(r)}"

        return MainWindow._batch_loop("魔塔增强", files, per, progress_cb)

    @staticmethod
    def _job_modelscope(src, out_wav, out, progress_cb=None):
        progress_cb(5, "加载魔塔模型(首次需下载)…")
        tmp = out_wav + ".in.wav"
        ft.extract_audio(src, tmp, sr=16000, mono=True)
        msa.modelscope_enhance(tmp, out_wav)
        if ft.is_video(src):
            out = os.path.splitext(out)[0] + ".mp4"
            ft.mux_replace_audio(src, out_wav, out)
        else:
            out = ft.encode_wav_to_media(out_wav, out)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return out

    # ---------------------------------------------------------------- 分割
    def _split_collect(self) -> list[str] | None:
        """收集分割源：统一源文件组（文件+目录递归扫描）；列表为空回退当前打开的文件。"""
        return self._collect_source(self.grp_split)

    def _split_opts(self) -> tuple[bool, int]:
        use_subdir = self.chk_split_subdir.isChecked()
        name_len = int(self.spin_split_namelen.value()) if self.chk_split_namelen.isChecked() else 0
        return use_subdir, name_len

    def _split_targets(self, kind: str, src: str | None = None) -> tuple[str, str]:
        """按分割输出选项返回 (输出目录, 文件名前缀)。

        勾选子目录：<src目录>/<源文件名>_<kind>分割/，前缀为空；
        不勾选：输出到源目录，前缀为 <源文件名>_（避免多文件混淆）。
        """
        from . import filetools
        use_subdir, name_len = self._split_opts()
        return filetools.split_target(src or self.src or "", kind, use_subdir, name_len)

    @staticmethod
    def _split_report(title: str, done: list, failed: list, skipped: list | None = None) -> dict:
        return MainWindow._batch_report(title, done, failed, skipped)

    def split_fixed(self):
        files = self._split_collect()
        if not files:
            return
        seg = float(self.spin_seg.value())
        use_subdir, name_len = self._split_opts()
        self._run("定长分割", self._job_split_fixed_batch, files, seg, use_subdir, name_len)

    @staticmethod
    def _job_split_fixed_batch(files, seg, use_subdir, name_len, progress_cb=None):
        from . import filetools
        n = len(files)
        done, failed = [], []
        for i, src in enumerate(files):
            name = os.path.basename(src)
            try:
                out_dir, prefix = filetools.split_target(src, "定长", use_subdir, name_len)
                base = f"{prefix}part" if prefix else "part"
                outs = ft.split_fixed(src, out_dir, seg, base=base)
                done.append((name, f"→ {len(outs)} 段 → {out_dir}", outs[0] if outs else ""))
            except Exception as e:
                failed.append((name, str(e)))
            progress_cb(int(100 * (i + 1) / n), f"定长分割 {i + 1}/{n} 完成")
        progress_cb(100, "定长分割完成")
        return MainWindow._split_report("定长分割", done, failed)

    def split_by_feature(self):
        files = self._split_collect()
        if not files:
            return
        mode = self.cmb_feat_split.currentText()
        if mode == "说话人参考区间" and self.speaker_ref is None:
            QMessageBox.information(self, "提示", "请先设置说话人参考区间")
            return
        if mode == "相似特征参考" and self.feature_ref is None and self.feature_ref_file is None:
            QMessageBox.information(self, "提示", "请先设置相似特征参考区间或导入参考文件")
            return
        use_subdir, name_len = self._split_opts()
        # 参考音频统一取自设置标记的文件（当前打开的文件）；未打开则取列表第一个
        ref_src = os.path.abspath(self.src) if self.src else files[0]
        self._run("特征分割", self._job_split_feat_batch, files, mode,
                  self.speaker_ref, self.feature_ref, self.feature_ref_file,
                  float(self.spin_sim_thr.value()), use_subdir, name_len, ref_src)

    @staticmethod
    def _job_split_feat_batch(files, mode, spk_ref, feat_ref, feat_file, sim_thr,
                              use_subdir, name_len, ref_src, progress_cb=None):
        from . import filetools
        import soundfile as sf
        # 批量分割时参考音频只提取一次（说话人区间/相似区间均取自标记所在的源文件）
        ref_audio = None
        if mode == "说话人参考区间" and spk_ref:
            tmp = ref_src + ".ref.wav"
            ft.extract_audio(ref_src, tmp, sr=16000, start=spk_ref[0], end=spk_ref[1])
            ref, _ = sf.read(tmp, dtype="float32")
            if ref.ndim > 1:
                ref = ref.mean(axis=1)
            ref_audio = (ref, 16000)
            try:
                os.remove(tmp)
            except OSError:
                pass
        elif mode == "相似特征参考" and not feat_file and feat_ref:
            tmp = ref_src + ".feat.wav"
            ft.extract_audio(ref_src, tmp, sr=16000, start=feat_ref[0], end=feat_ref[1])
            ref, _ = sf.read(tmp, dtype="float32")
            if ref.ndim > 1:
                ref = ref.mean(axis=1)
            ref_audio = (ref, 16000)
            try:
                os.remove(tmp)
            except OSError:
                pass
        n = len(files)
        done, failed = [], []
        for i, src in enumerate(files):
            name = os.path.basename(src)

            def cb(p, m="", _i=i):
                progress_cb(min(99, int(100 * _i / n) + int(p / n)),
                            f"[{_i + 1}/{n}] {name}: {m}")

            try:
                out_dir, prefix = filetools.split_target(src, "特征", use_subdir, name_len)
                outs = MainWindow._job_split_feat(src, out_dir, mode, spk_ref, feat_ref,
                                                  feat_file, sim_thr, prefix,
                                                  progress_cb=cb, ref_audio=ref_audio)
                done.append((name, f"→ {len(outs)} 段 → {out_dir}", outs[0] if outs else ""))
            except Exception as e:
                failed.append((name, str(e)))
            progress_cb(int(100 * (i + 1) / n), f"特征分割 {i + 1}/{n} 完成")
        progress_cb(100, "特征分割完成")
        return MainWindow._split_report("特征分割", done, failed)

    @staticmethod
    def _job_split_feat(src, out_dir, mode, spk_ref, feat_ref, feat_file, sim_thr,
                        prefix="", progress_cb=None, ref_audio=None):
        import soundfile as sf
        y, sr = audio_ops.load_wav(src, 16000)
        total = len(y) / sr
        progress_cb(10, "分析特征…")
        if mode == "静音边界":
            ranges = features.detect_silence(y, sr)
        elif mode == "说话人参考区间":
            if ref_audio is not None:
                ref, _ = ref_audio
            else:
                tmp = src + ".ref.wav"
                ft.extract_audio(src, tmp, sr=16000, start=spk_ref[0], end=spk_ref[1])
                ref, _ = sf.read(tmp, dtype="float32")
                if ref.ndim > 1:
                    ref = ref.mean(axis=1)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            ranges = features.find_similar_segments(y, sr, ref, sr, threshold=0.80)
        else:
            if feat_file:
                ref, rsr = sf.read(feat_file, dtype="float32")
                if ref.ndim > 1:
                    ref = ref.mean(axis=1)
                if rsr != sr:
                    n_new = int(len(ref) * sr / rsr)
                    ref = np.interp(np.arange(n_new) / sr,
                                    np.arange(len(ref)) / rsr, ref)
            elif ref_audio is not None:
                ref, _ = ref_audio
            else:
                tmp = src + ".feat.wav"
                ft.extract_audio(src, tmp, sr=16000, start=feat_ref[0], end=feat_ref[1])
                ref, _ = sf.read(tmp, dtype="float32")
                if ref.ndim > 1:
                    ref = ref.mean(axis=1)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            ranges = features.find_similar_segments(y, sr, ref, sr, threshold=sim_thr)
        bounds = features.segment_boundaries_from_ranges(ranges, total)
        pieces = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 0.1]
        progress_cb(40, f"共 {len(pieces)} 段，导出中…")
        os.makedirs(out_dir, exist_ok=True)
        outs = []
        ext = os.path.splitext(src)[1] or ".mp4"
        for i, (a, b) in enumerate(pieces):
            o = os.path.join(out_dir, f"{prefix}piece_{i + 1:04d}_{a:.2f}-{b:.2f}{ext}")
            ft.cut_range(src, o, a, b)
            outs.append(o)
            progress_cb(40 + int(60 * (i + 1) / max(1, len(pieces))), f"导出 {i + 1}/{len(pieces)}")
        return outs

    # ---------------------------------------------------------------- 关键词/正则分割
    _KW_KIND = {"以关键字开头分割": "head", "以关键字结束分割": "tail", "去掉关键字分割": "erase",
                "正则前分割": "head", "正则后分割": "tail", "抹正则分割": "erase"}

    def split_keyword(self):
        files = self._split_collect()
        if not files:
            return
        pat = self.ed_kw.text().strip()
        if not pat:
            QMessageBox.information(self, "提示", "请输入关键词或正则表达式")
            return
        mode = self.cmb_kw_mode.currentText()
        if "正则" in mode:
            try:
                re.compile(pat)
            except re.error as e:
                QMessageBox.warning(self, "正则表达式错误", str(e))
                return
        if not msa.asr_available():
            QMessageBox.warning(self, "缺少依赖",
                                "关键词分割需要魔塔语音识别模型。\n\n请先执行:\n  uv sync --extra modelscope\n\n"
                                f"模型: {msa.ASR_MODEL_ID}")
            return
        use_subdir, name_len = self._split_opts()
        self._run(f"关键词分割({mode})", self._job_split_keyword_batch, files, mode, pat,
                  float(self.spin_kw_pb.value()), float(self.spin_kw_pa.value()),
                  use_subdir, name_len)

    @staticmethod
    def _job_split_keyword_batch(files, mode, pattern, pad_b, pad_a,
                                 use_subdir, name_len, progress_cb=None):
        from . import filetools
        n = len(files)
        done, failed, skipped = [], [], []
        for i, src in enumerate(files):
            name = os.path.basename(src)

            def cb(p, m="", _i=i):
                progress_cb(min(99, int(100 * _i / n) + int(p / n)),
                            f"[{_i + 1}/{n}] {name}: {m}")

            try:
                out_dir, prefix = filetools.split_target(src, "关键词", use_subdir, name_len)
                res = MainWindow._job_split_keyword(src, out_dir, mode, pattern,
                                                    pad_b, pad_a, prefix, progress_cb=cb)
                if isinstance(res, dict):
                    mcnt = re.search(r"输出\s*(\d+)\s*段", res.get("report", ""))
                    seg_n = mcnt.group(1) if mcnt else "?"
                    done.append((name, f"→ {seg_n} 段 → {out_dir}", res.get("output", "")))
                else:
                    skipped.append((name, str(res)))
            except Exception as e:
                failed.append((name, str(e)))
            progress_cb(int(100 * (i + 1) / n), f"关键词分割 {i + 1}/{n} 完成")
        progress_cb(100, "关键词分割完成")
        return MainWindow._split_report(f"关键词分割（{mode}）", done, failed, skipped)

    @staticmethod
    def _job_split_keyword(src, out_dir, mode, pattern, pad_b, pad_a,
                           prefix="", progress_cb=None):
        progress_cb(5, "提取音频…")
        wav = src + ".asr.wav"
        ft.extract_audio(src, wav, sr=16000, mono=True)
        try:
            progress_cb(15, "语音识别中…")
            words = msa.transcribe_words(wav)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
        if not words:
            return "未识别到语音内容，无法按关键词分割"
        if "正则" in mode:
            occ = msa.find_occurrences(words, pattern=pattern)
        else:
            occ = msa.find_occurrences(words, keyword=pattern)
        if not occ:
            return f"未找到匹配内容: {pattern}"
        progress_cb(55, f"找到 {len(occ)} 处匹配，计算分割段…")
        total = ft.media_info(src).duration
        segs = msa.compute_split_segments(occ, total,
                                          MainWindow._KW_KIND[mode], pad_b, pad_a)
        if not segs:
            return "分割结果为空（留白设置可能过大）"
        os.makedirs(out_dir, exist_ok=True)
        ext = os.path.splitext(src)[1] or ".mp4"
        outs = []
        for i, (a, b) in enumerate(segs):
            o = os.path.join(out_dir, f"{prefix}seg_{i + 1:03d}_{a:.2f}-{b:.2f}{ext}")
            ft.cut_range(src, o, a, b)
            outs.append(o)
            progress_cb(60 + int(35 * (i + 1) / len(segs)), f"导出 {i + 1}/{len(segs)}")
        occ_desc = "\n".join(f"  {s:.2f}s ~ {e:.2f}s  {t}" for s, e, t in occ[:30])
        return {"output": outs[0],
                "report": f"模式: {mode}；匹配 {len(occ)} 处；输出 {len(outs)} 段\n"
                          f"输出目录: {out_dir}\n匹配位置:\n{occ_desc}"}

    # ---------------------------------------------------------------- 合并 / 分离
    def merge_av(self):
        v, a = self.ed_video.text().strip(), self.ed_audio.text().strip()
        if not (v and a and os.path.isfile(v) and os.path.isfile(a)):
            QMessageBox.information(self, "提示", "请选择有效的视频与音频文件")
            return
        out = os.path.splitext(v)[0] + "_合并.mp4"
        self._run("音视频合并", self._job_merge, v, a, out,
                  float(self.spin_offset.value()), self.chk_shortest.isChecked())

    @staticmethod
    def _job_merge(v, a, out, offset, shortest, progress_cb=None):
        return ft.merge_av(v, a, out, audio_offset=offset, use_shortest=shortest)

    def extract_audio(self):
        files = self._collect_source(self.grp_extract)
        if not files:
            return
        ext = self.cmb_extract_fmt.currentData() or ".wav"
        self._run("分离音频", self._job_extract_batch, files, ext)

    @staticmethod
    def _job_extract_batch(files, ext, progress_cb=None):
        from . import filetools

        def per(src, cb):
            out = filetools.plan_output(src, "", "", suffix="_音频", ext=ext)
            ft.extract_audio_keep(src, out)
            cb(100, "完成")
            return "ok", f"→ {os.path.basename(out)}"

        return MainWindow._batch_loop("分离音频", files, per, progress_cb)

    # ---------------------------------------------------------------- 格式转换
    def convert_media_batch(self):
        inputs = self.grp_convert.paths()
        if not inputs:
            QMessageBox.information(self, "提示", "请先添加要转换的媒体文件或目录（列表留空 = 转换当前打开的文件）")
            return
        missing = [p for p in inputs if not os.path.exists(p)]
        if missing:
            QMessageBox.warning(self, "提示", "以下路径不存在：\n" + "\n".join(missing[:5]))
            return
        fmt = self.cmb_conv_fmt.currentData() or ".mp3"
        br_txt = self.cmb_conv_br.currentText().strip()
        bitrate = None if br_txt in ("", "源", "源 / 默认", "源/默认") else br_txt
        sr_txt = self.cmb_conv_sr.currentText().strip()
        try:
            sr = None if sr_txt in ("源", "") else int(sr_txt)
        except ValueError:
            sr = None
        denoise = float(self.spin_conv_denoise.value()) if self.chk_conv_denoise.isChecked() else 0.0
        out_root = self.ed_conv_outroot.text().strip()
        recurse = self.grp_convert.chk_recurse.isChecked()
        keep_struct = self.chk_conv_keepstruct.isChecked()
        name_len = int(self.spin_conv_namelen.value()) if self.chk_conv_namelen.isChecked() else 0
        self._run("格式转换", self._job_convert, inputs, fmt, bitrate, sr, denoise,
                  out_root, recurse, keep_struct, name_len)

    @staticmethod
    def _job_convert(inputs, fmt, bitrate, sr, denoise,
                     out_root, recurse, keep_struct, name_len, progress_cb=None):
        from . import filetools
        files = filetools.scan_inputs(inputs, filetools.MEDIA_EXTS, recursive=recurse)
        if not files:
            return {"output": "", "report": "未在所选路径中找到可转换的音视频文件"}
        root = filetools.source_root(inputs)
        outs = []
        n = len(files)
        for i, src in enumerate(files):
            dst = filetools.plan_output(src, root, out_root, ext=fmt,
                                        keep_structure=keep_struct, max_name_len=name_len)
            progress_cb(int(95 * i / n), f"转换 {i + 1}/{n}: {os.path.basename(src)}")
            ft.convert_media(src, dst, bitrate=bitrate, sr=sr, denoise=denoise)
            outs.append(dst)
        progress_cb(100, "转换完成")
        return {"output": outs[0],
                "report": f"共转换 {n} 个文件（格式 {fmt}，码率 {bitrate or '默认'}，"
                          f"采样率 {sr or '源'}，降噪 {denoise:g}）:\n" + "\n".join(outs)}

    # ---------------------------------------------------------------- 文件名/夹处理
    _RN_MODES = ["replace", "keep", "lcut", "cut"]

    def _rn_collect(self) -> tuple[list[str], dict]:
        from . import filetools
        d = self.ed_rn_dir.text().strip()
        if not d or not os.path.isdir(d):
            QMessageBox.information(self, "提示", "请先选择有效的源目录")
            return [], {}
        ext_txt = self.ed_rn_exts.text().strip().lower()
        exts = None
        if ext_txt:
            exts = {("." + e.strip().lstrip(".")) for e in ext_txt.replace("，", ",").split(",") if e.strip()}
        files = filetools.scan_inputs([d], exts, recursive=self.chk_rn_recurse.isChecked())
        mode = self._RN_MODES[self.cmb_rn_mode.currentIndex()]
        params = dict(mode=mode, n=int(self.spin_rn_n.value()), m=int(self.spin_rn_m.value()),
                      find=self.ed_rn_find.text(), repl=self.ed_rn_repl.text(),
                      max_name_len=int(self.spin_rn_namelen.value()) if self.chk_rn_namelen.isChecked() else 0,
                      auto_index=self.chk_rn_index.isChecked())
        return files, params

    def rename_preview(self):
        files, params = self._rn_collect()
        if not files:
            self.pte_rn.setPlainText("未找到匹配的文件")
            return
        if params["mode"] == "replace" and not params["find"]:
            QMessageBox.information(self, "提示", "字符串替换模式需要填写「查找串」")
            return
        from . import filetools
        plan = filetools.plan_rename(files, **params)
        ok = [p for p in plan if p["status"] == "ok"]
        lines = [f"共 {len(plan)} 个文件，将重命名 {len(ok)} 个：", ""]
        for p in plan:
            if p["status"] == "ok":
                lines.append(f"{os.path.basename(p['old'])}  →  {os.path.basename(p['new'])}")
            else:
                lines.append(f"[跳过] {os.path.basename(p['old'])}  （{p['note']}）")
        self.pte_rn.setPlainText("\n".join(lines))

    def rename_apply(self):
        files, params = self._rn_collect()
        if not files:
            return
        if params["mode"] == "replace" and not params["find"]:
            QMessageBox.information(self, "提示", "字符串替换模式需要填写「查找串」")
            return
        from . import filetools
        plan = filetools.plan_rename(files, **params)
        ok_n = sum(1 for p in plan if p["status"] == "ok")
        if ok_n == 0:
            QMessageBox.information(self, "提示", "没有需要重命名的文件（请先预览查看）")
            return
        if QMessageBox.question(self, "确认", f"将重命名 {ok_n} 个文件，是否继续？") \
                != QMessageBox.StandardButton.Yes:
            return
        self._run("批量重命名", self._job_rename, plan)

    @staticmethod
    def _job_rename(plan, progress_cb=None):
        from . import filetools
        done, failed = filetools.apply_rename(plan, progress_cb)
        progress_cb(100, "重命名完成")
        lines = [f"成功 {done} 个" + (f"，跳过/失败 {len(failed)} 个：" if failed else "："), ""]
        for p in plan:
            if p["status"] == "ok":
                lines.append(f"{os.path.basename(p['old'])}  →  {os.path.basename(p['new'])}")
            else:
                lines.append(f"[跳过] {os.path.basename(p['old'])}  （{p['note']}）")
        return {"output": plan[0]["new"] if plan else "", "report": "\n".join(lines)}

    # ---------------------------------------------------------------- 时间轴
    def change_speed(self):
        files = self._collect_source(self.grp_timeline)
        if not files:
            return
        sp = float(self.spin_speed.value())
        self._run("变速", self._job_speed_batch, files, sp)

    @staticmethod
    def _job_speed_batch(files, sp, progress_cb=None):
        from . import filetools
        suffix = f"_x{sp:g}"

        def per(src, cb):
            out = filetools.plan_output(src, "", "", suffix=suffix)
            ft.change_speed(src, out, sp)
            cb(100, "完成")
            return "ok", f"→ {os.path.basename(out)}"

        return MainWindow._batch_loop(f"变速 x{sp:g}", files, per, progress_cb)

    def _require_video(self) -> bool:
        if not self._require_src():
            return False
        if not (self.info and self.info.has_video):
            QMessageBox.information(self, "提示", "音画同步微调仅对视频文件有效（当前打开的是纯音频）")
            return False
        return True

    def preview_av_sync(self):
        """生成带音频偏移的 20 秒校准片段并自动播放，供试听同步效果。"""
        if not self._require_video():
            return
        off = float(self.spin_av_sync.value())
        if abs(off) < 0.001:
            QMessageBox.information(self, "提示", "请先设置音频偏移量（非 0）")
            return
        start = self.player.position() / 1000.0
        out = os.path.join(tempfile.gettempdir(), "clearvoice_sync_preview.mp4")
        self._run("同步预览",
                  lambda src, o, f, s, progress_cb=None: ft.av_sync_preview(src, o, f, s),
                  self.src, out, off, start,
                  on_done=lambda _res: self._play_preview(out, off))

    def _play_preview(self, path: str, off: float):
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()
        self.btn_play.setText("暂停")
        self.log_append(f"[同步预览] 正在播放校准片段（音频偏移 {off:+g} 秒）：{path}")

    def restore_av_preview(self):
        if not self._require_src():
            return
        self.player.setSource(QUrl.fromLocalFile(self.src))
        self.log_append("[同步预览] 已恢复播放原文件")

    def apply_av_sync(self):
        """按当前偏移量批量生成新文件：视频流 copy，音频提前/推迟（纯音频自动跳过）。"""
        files = self._collect_source(self.grp_timeline)
        if not files:
            return
        off = float(self.spin_av_sync.value())
        if abs(off) < 0.001:
            QMessageBox.information(self, "提示", "偏移量为 0，无需处理")
            return
        self._run("音画同步", self._job_sync_batch, files, off)

    @staticmethod
    def _job_sync_batch(files, off, progress_cb=None):
        from . import filetools
        suffix = f"_同步{off:+g}s"

        def per(src, cb):
            if not ft.is_video(src):
                return "skip", "纯音频文件，音画同步仅对视频有效"
            out = filetools.plan_output(src, "", "", suffix=suffix, ext=".mp4")
            ft.av_sync_offset(src, out, off)
            cb(100, "完成")
            return "ok", f"→ {os.path.basename(out)}"

        return MainWindow._batch_loop(f"音画同步 {off:+g}s", files, per, progress_cb)

    def trim_selection(self):
        if not self._require_src():
            return
        r = self._sel_range()
        if not r:
            return
        out = self.out_path(f"_裁剪{r[0]:.1f}-{r[1]:.1f}")
        self._run("裁剪", lambda src, o, s, e, progress_cb=None: ft.cut_range(src, o, s, e),
                  self.src, out, r[0], r[1])

    # ---------------------------------------------------------------- 特征剔除
    def remove_text_feature(self):
        files = self._collect_source(self.grp_feature)
        if not files:
            return
        kws = [k.strip() for k in self.ed_keywords.text().replace("，", ",").split(",") if k.strip()]
        if not kws:
            QMessageBox.information(self, "提示", "请输入关键词")
            return
        if not msa.asr_available():
            QMessageBox.warning(self, "缺少依赖",
                                "文字特征剔除需要魔塔语音识别模型。\n\n"
                                "请先执行:\n  uv sync --extra modelscope\n\n"
                                f"模型: {msa.ASR_MODEL_ID}（首次运行自动下载）")
            return
        mute = self.cmb_text_mode.currentText() == "静音该段"
        self._run("文字特征剔除", self._job_text_remove_batch, files, kws, mute)

    @staticmethod
    def _job_text_remove_batch(files, kws, mute, progress_cb=None):
        from . import filetools

        def per(src, cb):
            out = filetools.plan_output(src, "", "", suffix="_文字剔除",
                                        ext=".mp4" if ft.is_video(src) else ".m4a")
            r = MainWindow._job_text_remove(src, out, kws, mute, progress_cb=cb)
            if isinstance(r, dict):
                return "ok", f"→ {os.path.basename(r['output'])}"
            return "skip", str(r)

        return MainWindow._batch_loop("文字特征剔除", files, per, progress_cb)

    @staticmethod
    def _job_text_remove(src, out, kws, mute, progress_cb=None):
        progress_cb(5, "提取音频…")
        wav = src + ".asr.wav"
        ft.extract_audio(src, wav, sr=16000, mono=True)
        progress_cb(15, "语音识别中（首次需下载魔塔模型）…")
        hits, matched = msa.find_keyword_segments(wav, kws)
        try:
            os.remove(wav)
        except OSError:
            pass
        if not hits:
            return "未发现包含关键词的内容"
        progress_cb(60, f"命中 {len(hits)} 段，处理中…")
        out = os.path.splitext(out)[0] + (".mp4" if ft.is_video(src) else ".m4a")
        if mute:
            ft.mute_ranges(src, out, hits)
        else:
            ft.remove_ranges(src, out, hits)
        return {"output": out, "matched": matched}

    def remove_similar_feature(self):
        files = self._collect_source(self.grp_feature)
        if not files:
            return
        if self.feature_ref is None and self.feature_ref_file is None:
            QMessageBox.information(self, "提示", "请先『设为相似特征参考』或导入参考音频文件")
            return
        mute = self.cmb_sim_mode.currentText() == "静音该段"
        thr = float(self.spin_sim_thr.value())
        # 参考音频取自标记所在的当前文件（批量时只提取一次）
        ref_src = os.path.abspath(self.src) if (self.feature_ref is not None and self.src) else None
        self._run("相似特征剔除", self._job_sim_remove_batch, files,
                  self.feature_ref, self.feature_ref_file, thr, mute, ref_src)

    @staticmethod
    def _job_sim_remove_batch(files, feat_ref, feat_file, thr, mute, ref_src,
                              progress_cb=None):
        from . import filetools
        import soundfile as sf
        ref_audio = None
        if not feat_file and feat_ref is not None and ref_src:
            tmp = ref_src + ".feat.wav"
            ft.extract_audio(ref_src, tmp, sr=16000, start=feat_ref[0], end=feat_ref[1])
            ref_audio, _ = sf.read(tmp, dtype="float32")
            if ref_audio.ndim > 1:
                ref_audio = ref_audio.mean(axis=1)
            try:
                os.remove(tmp)
            except OSError:
                pass

        def per(src, cb):
            out = filetools.plan_output(src, "", "", suffix="_相似剔除")
            r = MainWindow._job_sim_remove(src, out, feat_ref, feat_file, thr, mute,
                                           progress_cb=cb, ref_audio=ref_audio)
            if isinstance(r, str) and os.path.isfile(r):
                return "ok", f"→ {os.path.basename(r)}"
            return "skip", str(r)

        return MainWindow._batch_loop("相似特征剔除", files, per, progress_cb)

    @staticmethod
    def _job_sim_remove(src, out, feat_ref, feat_file, thr, mute,
                        progress_cb=None, ref_audio=None):
        import soundfile as sf
        progress_cb(10, "提取音频…")
        y, sr = audio_ops.load_wav(src, 16000)
        if feat_file:
            ref, rsr = sf.read(feat_file, dtype="float32")
            if ref.ndim > 1:
                ref = ref.mean(axis=1)
        elif ref_audio is not None:
            ref = ref_audio
        else:
            tmp = src + ".feat.wav"
            ft.extract_audio(src, tmp, sr=16000, start=feat_ref[0], end=feat_ref[1])
            ref, _ = sf.read(tmp, dtype="float32")
            if ref.ndim > 1:
                ref = ref.mean(axis=1)
            try:
                os.remove(tmp)
            except OSError:
                pass
        progress_cb(30, "相似特征检测…")
        ranges = features.find_similar_segments(y, sr, ref, sr, threshold=thr)
        if not ranges:
            return "未发现相似特征内容"
        progress_cb(60, f"命中 {len(ranges)} 段，处理中…")
        out = os.path.splitext(out)[0] + (".mp4" if ft.is_video(src) else ".m4a")
        if mute:
            ft.mute_ranges(src, out, ranges)
        else:
            ft.remove_ranges(src, out, ranges)
        return out

    # ---------------------------------------------------------------- 语音转文字
    def run_asr_export(self):
        files = self._collect_source(self.grp_asr)
        if not files:
            return
        fmts = [name for name, chk in
                (("txt", self.chk_out_txt), ("srt", self.chk_out_srt), ("lrc", self.chk_out_lrc))
                if chk.isChecked()]
        if not fmts:
            QMessageBox.information(self, "提示", "请至少选择一种输出格式")
            return
        if not msa.asr_available():
            QMessageBox.warning(self, "缺少依赖",
                                "语音识别需要魔塔 ASR 模型。\n\n请先执行:\n  uv sync --extra modelscope\n\n"
                                f"模型: {msa.ASR_MODEL_ID}（首次运行自动下载）")
            return
        self._run("语音识别导出", self._job_asr_batch, files, fmts,
                  on_done=self._after_asr_export)

    @staticmethod
    def _job_asr_batch(files, fmts, progress_cb=None):
        def per(src, cb):
            base = os.path.splitext(src)[0]
            r = MainWindow._job_asr_export(src, base, fmts, progress_cb=cb)
            if isinstance(r, dict):
                return "ok", f"→ {os.path.basename(base)}.{'/'.join(fmts)}"
            return "skip", str(r)

        return MainWindow._batch_loop("语音识别导出", files, per, progress_cb)

    @staticmethod
    def _job_asr_export(src, base, fmts, progress_cb=None):
        progress_cb(5, "提取音频…")
        wav = src + ".asr.wav"
        ft.extract_audio(src, wav, sr=16000, mono=True)
        try:
            progress_cb(15, "语音识别中（首次需下载魔塔模型，耗时较长）…")
            sents = msa.transcribe_with_timestamps(wav)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
        if not sents:
            return "未识别到语音内容"
        progress_cb(85, f"识别到 {len(sents)} 句，导出中…")
        outs = []
        if "txt" in fmts:
            outs.append(msa.export_txt(sents, base + ".txt"))
        if "srt" in fmts:
            outs.append(msa.export_srt(sents, base + ".srt"))
        if "lrc" in fmts:
            outs.append(msa.export_lrc(sents, base + ".lrc", title=os.path.basename(base)))
        preview = " ".join(s["text"].strip() for s in sents)[:300]
        return {"report": f"共识别 {len(sents)} 句；生成 {len(outs)} 个文件:\n" + "\n".join(outs)
                          + f"\n\n文本预览: {preview}"}

    def _after_asr_export(self, res=None):
        self._default_done("语音识别导出", res)

    # ---------------------------------------------------------------- 去背景音乐
    def remove_background_music(self):
        files = self._collect_source(self.grp_removal)
        if not files:
            return
        if not separation.available():
            QMessageBox.warning(self, "缺少依赖",
                                "人声/伴奏分离需要 torchaudio。\n\n请先执行:\n  uv sync --extra modelscope")
            return
        keep = float(self.spin_acc_keep.value())
        acc_out = self.chk_acc_out.isChecked()
        self._run("去背景音乐", self._job_music_batch, files, keep, acc_out)

    @staticmethod
    def _job_music_batch(files, keep, acc_out, progress_cb=None):
        def per(src, cb):
            acc_path = os.path.splitext(src)[0] + "_伴奏.wav" if acc_out else None
            r = MainWindow._job_music_remove(src, keep, acc_path, progress_cb=cb)
            return "ok", f"→ {os.path.basename(r)}"

        return MainWindow._batch_loop("去背景音乐", files, per, progress_cb)

    @staticmethod
    def _job_music_remove(src, keep_ratio, acc_path, progress_cb=None):
        base = os.path.splitext(src)[0]
        tmp_wav = base + ".sep.wav"
        voc_wav = base + "_人声.wav"
        progress_cb(2, "提取音频（44.1kHz 立体声）…")
        ft.extract_audio(src, tmp_wav, sr=44100, mono=False)
        try:
            separation.separate(tmp_wav, voc_wav, acc_path, keep_ratio, progress_cb)
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
        ext = os.path.splitext(src)[1].lower()
        if ft.is_video(src):
            out = base + "_去伴奏" + (ext or ".mp4")
            ft.mux_replace_audio(src, voc_wav, out)
        else:
            out = base + "_去伴奏" + (ext or ".wav")
            ft.encode_wav_to_media(voc_wav, out)
        try:
            os.remove(voc_wav)          # 中间产物不再需要
        except OSError:
            pass
        return out

    # ---------------------------------------------------------------- 音频翻译
    def _pick_tr_src(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择要翻译的文件", "", "媒体文件 (*)")
        if f:
            self.ed_tr_src.setText(f)

    def _tr_use_current(self):
        if not self._require_src():
            return
        self.ed_tr_src.setText(self.src)

    def run_translate(self):
        src = self.ed_tr_src.text().strip()
        if not (src and os.path.isfile(src)):
            QMessageBox.information(self, "提示", "请选择有效的音频或视频文件")
            return
        pair = self.cmb_tr_pair.currentText()
        if not msa.asr_available():
            QMessageBox.warning(self, "缺少依赖",
                                "音频翻译需要语音识别。\n\n请先执行:\n  uv sync --extra modelscope")
            return
        from . import config as _cfg
        if _cfg.load_config().get("translate_backend", "local") == "local" \
                and not translator.mt_available():
            QMessageBox.warning(self, "缺少依赖",
                                "本地翻译需要 transformers（M2M100 多语言模型）。\n\n"
                                "请先执行:\n  uv sync --extra modelscope\n\n"
                                "或到「设置」页改用 OpenAI 兼容大模型 API。")
            return
        self._run(f"音频翻译({pair})", self._job_translate, src, pair)

    @staticmethod
    def _job_translate(src, pair, progress_cb=None):
        src_lang, tgt_lang = translator.PAIRS[pair]
        progress_cb(3, "提取音频…")
        wav = src + ".tr.wav"
        ft.extract_audio(src, wav, sr=16000, mono=True)
        try:
            progress_cb(8, "语音识别中…")
            sents = msa.transcribe_with_timestamps(wav)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
        if not sents:
            return "未识别到语音内容，无法翻译"
        texts = [s.get("text", "").strip() for s in sents]
        progress_cb(20, f"识别 {len(texts)} 句，开始翻译…")
        trans = translator.translate(texts, src_lang, tgt_lang,
                                     lambda p: progress_cb(20 + int(70 * p / 100)))
        for s, t in zip(sents, trans):
            s["trans"] = t
        base = os.path.splitext(src)[0]
        outs = [
            translator.export_txt_bilingual(sents, base + ".翻译.txt"),
            translator.export_srt_bilingual(sents, base + ".双语.srt"),
            translator.export_lrc_bilingual(sents, base + ".双语.lrc",
                                            title=os.path.basename(base)),
        ]
        preview = "\n".join(f"  {t} → {tr}" for t, tr in zip(texts[:5], trans[:5]))
        return {"report": f"翻译方向: {pair}；共 {len(sents)} 句；生成:\n" + "\n".join(outs)
                          + f"\n\n预览:\n{preview}"}

    # ---------------------------------------------------------------- 文字转语音
    def _pick_tts_out(self):
        f, _ = QFileDialog.getSaveFileName(self, "选择输出位置", "tts_output.mp3",
                                           "MP3 音频 (*.mp3)")
        if f:
            self.ed_tts_out.setText(f)

    def run_tts(self):
        text = self.pte_tts.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "请输入要合成的文本")
            return
        out = self.ed_tts_out.text().strip()
        if not out:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tts_out")
            os.makedirs(out_dir, exist_ok=True)
            import time as _time
            out = os.path.join(out_dir, "tts_" + _time.strftime("%Y%m%d_%H%M%S") + ".mp3")
        voice = tts_mod.VOICES[self.cmb_tts_voice.currentIndex()][1]
        from . import config as _cfg
        if _cfg.load_config().get("tts_backend", "edge") == "edge" \
                and not tts_mod.edge_available():
            QMessageBox.warning(self, "缺少依赖",
                                "TTS 需要 edge-tts。\n\n请先执行:\n  uv sync --extra modelscope\n\n"
                                "或到「设置」页改用 OpenAI 兼容 TTS API。")
            return
        self._run("文字转语音", self._job_tts, text, voice, out)

    @staticmethod
    def _job_tts(text, voice, out, progress_cb=None):
        tts_mod.synth(text, out, voice, progress_cb)
        return out

    # ---------------------------------------------------------------- 设置
    def _load_settings(self):
        cfg = config.load_config()
        self.ed_model_dir.setText(cfg.get("funasr_model_dir", ""))
        self.cmb_device.setCurrentText(cfg.get("asr_device", "cpu"))
        self.ed_demucs_dir.setText(cfg.get("demucs_model_dir", ""))
        self.cmb_tr_backend.setCurrentIndex(1 if cfg.get("translate_backend") == "api" else 0)
        self.cmb_tts_backend.setCurrentIndex(1 if cfg.get("tts_backend") == "api" else 0)
        self.ed_api_base.setText(cfg.get("api_base", ""))
        self.ed_api_key.setText(cfg.get("api_key", ""))
        self.ed_api_model.setText(cfg.get("api_model", ""))
        self.ed_tr_model_dir.setText(cfg.get("translate_model_dir", ""))

    def _pick_model_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择 funasr 模型根目录",
                                             self.ed_model_dir.text() or "")
        if d:
            self.ed_model_dir.setText(d)

    def _pick_demucs_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择分离模型权重目录",
                                             self.ed_demucs_dir.text() or "")
        if d:
            self.ed_demucs_dir.setText(d)

    def _pick_tr_model_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择 M2M100 翻译模型目录",
                                             self.ed_tr_model_dir.text() or "")
        if d:
            self.ed_tr_model_dir.setText(d)

    def save_settings(self):
        cfg = {"funasr_model_dir": self.ed_model_dir.text().strip(),
               "asr_device": self.cmb_device.currentText(),
               "demucs_model_dir": self.ed_demucs_dir.text().strip(),
               "translate_backend": "api" if self.cmb_tr_backend.currentIndex() == 1 else "local",
               "tts_backend": "api" if self.cmb_tts_backend.currentIndex() == 1 else "edge",
               "api_base": self.ed_api_base.text().strip(),
               "api_key": self.ed_api_key.text().strip(),
               "api_model": self.ed_api_model.text().strip(),
               "translate_model_dir": self.ed_tr_model_dir.text().strip()}
        config.save_config(cfg)
        msa.reset_pipeline()
        self.log_append("设置已保存: 模型目录=" + cfg["funasr_model_dir"]
                        + f", 设备={cfg['asr_device']}, 翻译后端={cfg['translate_backend']}"
                        + f", TTS后端={cfg['tts_backend']}, API={cfg['api_base']}")
        QMessageBox.information(self, "设置", "已保存，下次识别/翻译/合成时生效。")
