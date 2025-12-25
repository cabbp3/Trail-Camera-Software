"""
Entry point: launch the labeler as the main window.
"""
import sys
from PyQt6.QtWidgets import QApplication
from training.label_tool import TrainerWindow, APP_STYLE


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = TrainerWindow()
    win.show()
    # Quit app when the labeler closes
    win.destroyed.connect(app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
