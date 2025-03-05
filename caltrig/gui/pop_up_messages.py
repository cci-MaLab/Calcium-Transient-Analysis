from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QProgressBar, QApplication


def print_error(s, extra_info="", severity=QMessageBox.Critical):
        dlg = QMessageBox()
        dlg.setWindowTitle("Error Message")
        if isinstance(s, tuple):
            text = f"For path {s[1]} the following error occurred:\n {s[0]}"
        else:
            text = s
        if extra_info != "":
            text += f"\n{extra_info}"
        dlg.setText(text)
        dlg.setIcon(severity)
        dlg.exec()


class ProgressWindow(QDialog):
    def __init__(self, total_steps, text="Shuffling", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Progress")
        self.setGeometry(600, 300, 300, 100)
        self.text = text

        layout = QVBoxLayout(self)

        self.label = QLabel("Progress:", self)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(total_steps)
        layout.addWidget(self.progress_bar)

        self.setFixedSize(400, self.sizeHint().height())

    def update_progress(self, step):
        self.progress_bar.setValue(step)
        self.label.setText(f"{self.text} {step}/{self.progress_bar.maximum()}...")
        QApplication.processEvents()