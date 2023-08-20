from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout)
from PyQt5.QtGui import QIntValidator

class ParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Specify Parameters")

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.day_label = QLabel("Day")
        self.session_label = QLabel("Session")
        self.mouseid_label = QLabel("Mouse ID")

        self.day_edit = QLineEdit()
        onlyInt = QIntValidator()
        onlyInt.setRange(0, 1000)
        self.day_edit.setValidator(onlyInt)
        self.day_layout = QHBoxLayout()
        self.day_layout.addWidget(self.day_label)
        self.day_layout.addWidget(self.day_edit)

        self.session_edit = QLineEdit()
        self.session_layout = QHBoxLayout()
        self.session_layout.addWidget(self.session_label)
        self.session_layout.addWidget(self.session_edit)

        self.mouseid_edit = QLineEdit()
        self.mouseid_layout = QHBoxLayout()
        self.mouseid_layout.addWidget(self.mouseid_label)
        self.mouseid_layout.addWidget(self.mouseid_edit)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.day_layout)
        self.layout.addLayout(self.session_layout)
        self.layout.addLayout(self.mouseid_layout)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        