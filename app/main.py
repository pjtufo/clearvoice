"""ClearVoice 启动入口。用法: uv run python -m app.main"""
from __future__ import annotations

import sys


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    # 必须在创建 QApplication 之前设置
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("ClearVoice")

    from .main_window import MainWindow
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
