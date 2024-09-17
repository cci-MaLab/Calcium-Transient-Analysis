from PyQt5.QtWidgets import QMessageBox


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