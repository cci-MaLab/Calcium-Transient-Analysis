from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QWidget,
                            QCheckBox)
from PyQt5.QtGui import QIntValidator

class ParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Specify Parameters")

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

        self.group_chkbox = QCheckBox("Cocaine Group? (Otherwise Saline)")

        self.ALP_chkbox = QCheckBox("ALP")
        self.ALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.ALP_chkbox, self.ALP_param))
        self.ALP_chkbox.stateChanged.connect(self.release_button)
        self.IALP_chkbox = QCheckBox("IALP")
        self.IALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.IALP_chkbox, self.IALP_param))
        self.ALP_chkbox.stateChanged.connect(self.release_button)
        self.RNFS_chkbox = QCheckBox("RNFS")
        self.RNFS_chkbox.stateChanged.connect(lambda: hide_unhide(self.RNFS_chkbox, self.RNFS_param))
        self.ALP_chkbox.stateChanged.connect(self.release_button)

        self.ALP_param = ParamWidget()
        self.ALP_param.setEnabled(False)
        self.IALP_param = ParamWidget()
        self.IALP_param.setEnabled(False)
        self.RNFS_param = ParamWidget()
        self.RNFS_param.setEnabled(False)

        layout_param = QHBoxLayout()
        ALP_layout = QVBoxLayout()
        IALP_layout = QVBoxLayout()
        RNFS_layout = QVBoxLayout()

        ALP_layout.addWidget(self.ALP_chkbox)
        ALP_layout.addWidget(self.ALP_param)
        IALP_layout.addWidget(self.IALP_chkbox)
        IALP_layout.addWidget(self.IALP_param)
        RNFS_layout.addWidget(self.RNFS_chkbox)
        RNFS_layout.addWidget(self.RNFS_param)

        layout_param.addLayout(ALP_layout)
        layout_param.addLayout(IALP_layout)
        layout_param.addLayout(RNFS_layout)
        layout_param.addWidget(self.group_chkbox)

        layout = QVBoxLayout()
        layout.addLayout(layout_param)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def release_button(self):
        if self.ALP_chkbox.isChecked() or self.IALP_chkbox.isChecked() or self.RNFS_chkbox.isChecked():
            self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        else:
            self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def get_result(self):
        result = {}
        if self.ALP_chkbox.isChecked():
            result["ALP"] = {}
            result["ALP"]["window"] = int(self.ALP_param.duration_edit.text())
            result["ALP"]["delay"] = int(self.ALP_param.delay_edit.text())
        if self.IALP_chkbox.isChecked():
            result["IALP"] = {}
            result["IALP"]["window"] = int(self.IALP_param.duration_edit.text())
            result["IALP"]["delay"] = int(self.IALP_param.delay_edit.text())
        if self.RNFS_chkbox.isChecked():
            result["RNFS"] = {}
            result["RNFS"]["window"] = int(self.RNFS_param.duration_edit.text())
            result["RNFS"]["delay"] = int(self.RNFS_param.delay_edit.text())
        
        if self.group_chkbox.isChecked():
            result["group"] = "cocaine"
        else:
            result["group"] = "saline"
        
        return result


def hide_unhide(chkbox, param):
    if chkbox.isChecked():
        param.setEnabled(True)
    else:
        param.setEnabled(False)



class ParamWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        duration_label = QLabel("Window: ")
        delay_label = QLabel("Delay: ")

        self.duration_edit = QLineEdit("20")
        onlyInt = QIntValidator()
        onlyInt.setRange(20, 120)
        self.duration_edit.setValidator(onlyInt)

        self.delay_edit = QLineEdit("0")
        onlyInt = QIntValidator()
        onlyInt.setRange(-20, 20)
        self.delay_edit.setValidator(onlyInt)

        delay_layout = QHBoxLayout()
        delay_layout.addWidget(delay_label)
        delay_layout.addWidget(self.delay_edit)

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_edit)

        layout.addLayout(duration_layout)
        layout.addLayout(delay_layout)

        self.setLayout(layout)

        

class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Load Data")

        QBtn = QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel("Detected paths.txt in your directory. Do you want to load the files?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)