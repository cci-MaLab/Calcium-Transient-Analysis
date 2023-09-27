from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QWidget,
                            QCheckBox, QGridLayout, QFrame, QGraphicsView, QGraphicsScene, QPushButton, 
                            QComboBox, QListWidget, QAbstractItemView, QSplitter, QApplication, QStyleFactory,
                            QAction, QFileDialog)
from PyQt5.QtGui import (QIntValidator, QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont)
from PyQt5.QtCore import (Qt, pyqtSlot, QRunnable, QThreadPool, pyqtSignal)
import pyqtgraph as pg
from pyqtgraph import PlotItem
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtGui import QPixmap
import os
import bisect



class ParamDialog(QDialog):
    def __init__(self, event_defaults, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Specify Parameters")

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

        self.ALP_chkbox = QCheckBox("ALP")
        self.ALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.ALP_chkbox, self.ALP_param))
        self.ALP_chkbox.stateChanged.connect(self.release_button)
        self.IALP_chkbox = QCheckBox("IALP")
        self.IALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.IALP_chkbox, self.IALP_param))
        self.IALP_chkbox.stateChanged.connect(self.release_button)
        self.RNFS_chkbox = QCheckBox("RNFS")
        self.RNFS_chkbox.stateChanged.connect(lambda: hide_unhide(self.RNFS_chkbox, self.RNFS_param))
        self.RNFS_chkbox.stateChanged.connect(self.release_button)
        self.ALP_Timeout_chkbox = QCheckBox("ALP_Timeout")
        self.ALP_Timeout_chkbox.stateChanged.connect(lambda: hide_unhide(self.ALP_Timeout_chkbox, self.ALP_Timeout_param))
        self.ALP_Timeout_chkbox.stateChanged.connect(self.release_button)

        self.ALP_param = ParamWidget("ALP", event_defaults)
        self.ALP_param.setEnabled(False)
        self.IALP_param = ParamWidget("IALP", event_defaults)
        self.IALP_param.setEnabled(False)
        self.RNFS_param = ParamWidget("RNFS", event_defaults)
        self.RNFS_param.setEnabled(False)
        self.ALP_Timeout_param = ParamWidget("ALP_Timeout", event_defaults)
        self.ALP_Timeout_param.setEnabled(False)

        layout_param = QHBoxLayout()
        ALP_layout = QVBoxLayout()
        IALP_layout = QVBoxLayout()
        RNFS_layout = QVBoxLayout()
        ALP_Timeout_layout = QVBoxLayout()

        ALP_layout.addWidget(self.ALP_chkbox)
        ALP_layout.addWidget(self.ALP_param)
        IALP_layout.addWidget(self.IALP_chkbox)
        IALP_layout.addWidget(self.IALP_param)
        RNFS_layout.addWidget(self.RNFS_chkbox)
        RNFS_layout.addWidget(self.RNFS_param)
        ALP_Timeout_layout.addWidget(self.ALP_Timeout_chkbox)
        ALP_Timeout_layout.addWidget(self.ALP_Timeout_param)

        layout_param.addLayout(ALP_layout)
        layout_param.addLayout(IALP_layout)
        layout_param.addLayout(RNFS_layout)
        layout_param.addLayout(ALP_Timeout_layout)

        layout = QVBoxLayout()
        layout.addLayout(layout_param)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def release_button(self):
        if self.ALP_chkbox.isChecked() or self.IALP_chkbox.isChecked() or self.RNFS_chkbox.isChecked() or self.ALP_Timeout_chkbox.isChecked():
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
        if self.ALP_Timeout_chkbox.isChecked():
            result["ALP_Timeout"] = {}
            result["ALP_Timeout"]["window"] = int(self.ALP_Timeout_param.duration_edit.text())
            result["ALP_Timeout"]["delay"] = int(self.ALP_Timeout_param.delay_edit.text())
        
        return result


class UpdateDialog(QDialog):
    def __init__(self, event_defaults, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Specify New Defaults")

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.ALP_label = QLabel("ALP")
        self.IALP_label = QLabel("IALP")
        self.RNFS_label = QLabel("RNFS")
        self.ALP_Timeout_label = QLabel("ALP_Timeout")

        self.ALP_param = ParamWidget("ALP", event_defaults)
        self.IALP_param = ParamWidget("IALP", event_defaults)
        self.RNFS_param = ParamWidget("RNFS", event_defaults)
        self.ALP_Timeout_param = ParamWidget("ALP_Timeout", event_defaults)

        layout_param = QHBoxLayout()
        ALP_layout = QVBoxLayout()
        IALP_layout = QVBoxLayout()
        RNFS_layout = QVBoxLayout()
        ALP_Timeout_layout = QVBoxLayout()

        ALP_layout.addWidget(self.ALP_label)
        ALP_layout.addWidget(self.ALP_param)
        IALP_layout.addWidget(self.IALP_label)
        IALP_layout.addWidget(self.IALP_param)
        RNFS_layout.addWidget(self.RNFS_label)
        RNFS_layout.addWidget(self.RNFS_param)
        ALP_Timeout_layout.addWidget(self.ALP_Timeout_label)
        ALP_Timeout_layout.addWidget(self.ALP_Timeout_param)

        layout_param.addLayout(ALP_layout)
        layout_param.addLayout(IALP_layout)
        layout_param.addLayout(RNFS_layout)
        layout_param.addLayout(ALP_Timeout_layout)

        layout = QVBoxLayout()
        layout.addLayout(layout_param)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def get_result(self):
        result = {}
        result["ALP"] = {}
        result["ALP"]["window"] = int(self.ALP_param.duration_edit.text())
        result["ALP"]["delay"] = int(self.ALP_param.delay_edit.text())
        result["IALP"] = {}
        result["IALP"]["window"] = int(self.IALP_param.duration_edit.text())
        result["IALP"]["delay"] = int(self.IALP_param.delay_edit.text())
        result["RNFS"] = {}
        result["RNFS"]["window"] = int(self.RNFS_param.duration_edit.text())
        result["RNFS"]["delay"] = int(self.RNFS_param.delay_edit.text())
        result["ALP_Timeout"] = {}
        result["ALP_Timeout"]["window"] = int(self.ALP_Timeout_param.duration_edit.text())
        result["ALP_Timeout"]["delay"] = int(self.ALP_Timeout_param.delay_edit.text())
        
        return result

class ToolWidget(QWidget):
    def __init__(self, event_defaults, parent=None):
        super().__init__(parent)
        self.all_cells = None
        self.event_defaults = event_defaults

        label_cluster_select = QLabel()
        label_cluster_select.setText("Pick number of clusters:")
        self.cluster_select = QComboBox()
        for i in range (2, 20):
            self.cluster_select.addItem(str(i))
        self.cluster_select.setCurrentIndex(2)

        self.button = QPushButton("Update")
        self.button.setStyleSheet("background-color : green")
        self.button.setEnabled(False)
        self.button.setFixedWidth(120)
        self.button.clicked.connect(self.get_result)

        self.button_inspect = QPushButton("Inspect Cluster")
        self.button_inspect.setStyleSheet("background-color : green")
        self.button_inspect.setFixedWidth(120)
        self.button_inspect.clicked.connect(self.inspect)

        self.button_delete = QPushButton("Delete Cluster")
        self.button_delete.setStyleSheet("background-color : red")
        self.button_delete.setFixedWidth(120)
        self.button_delete.clicked.connect(self.delete)

        self.setFixedWidth(300)

        self.ALP_chkbox = QCheckBox("ALP")
        self.ALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.ALP_chkbox, self.ALP_param))
        self.ALP_chkbox.stateChanged.connect(self.release_button)
        self.IALP_chkbox = QCheckBox("IALP")
        self.IALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.IALP_chkbox, self.IALP_param))
        self.IALP_chkbox.stateChanged.connect(self.release_button)
        self.RNFS_chkbox = QCheckBox("RNFS")
        self.RNFS_chkbox.stateChanged.connect(lambda: hide_unhide(self.RNFS_chkbox, self.RNFS_param))
        self.RNFS_chkbox.stateChanged.connect(self.release_button)
        self.ALP_Timeout_chkbox = QCheckBox("ALP_Timeout")
        self.ALP_Timeout_chkbox.stateChanged.connect(lambda: hide_unhide(self.ALP_Timeout_chkbox, self.ALP_Timeout_param))
        self.ALP_Timeout_chkbox.stateChanged.connect(self.release_button)

        self.ALP_param = ParamWidget("ALP", self.event_defaults)
        self.ALP_param.setEnabled(False)
        self.IALP_param = ParamWidget("IALP", self.event_defaults)
        self.IALP_param.setEnabled(False)
        self.RNFS_param = ParamWidget("RNFS", self.event_defaults)
        self.RNFS_param.setEnabled(False)
        self.ALP_Timeout_param = ParamWidget("ALP_Timeout", self.event_defaults)
        self.ALP_Timeout_param.setEnabled(False)

        self.outlier_input_label = QLabel("Type in outlier to exclude")
        self.outlier_input = QLineEdit("")
        onlyInt = QIntValidator()
        self.outlier_input.setValidator(onlyInt)
        self.outlier_input.returnPressed.connect(self.add_outlier)
        self.outlier_input_button = QPushButton("Remove Outlier")
        self.outlier_input_button.clicked.connect(self.add_outlier)
        self.outlier_combo_label = QLabel("Current Outliers")
        self.outlier_combo_box = QComboBox()
        self.outlier_return_button = QPushButton("Return Outlier")
        self.outlier_return_button.clicked.connect(self.remove_outlier)


        layout_sub = QVBoxLayout()
        layout_sub.addStretch()
        layout_sub.setDirection(3)
        ALP_layout = QVBoxLayout()
        IALP_layout = QVBoxLayout()
        RNFS_layout = QVBoxLayout()
        ALP_Timeout_layout = QVBoxLayout()

        ALP_layout.addWidget(self.ALP_chkbox)
        ALP_layout.addWidget(self.ALP_param)
        IALP_layout.addWidget(self.IALP_chkbox)
        IALP_layout.addWidget(self.IALP_param)
        RNFS_layout.addWidget(self.RNFS_chkbox)
        RNFS_layout.addWidget(self.RNFS_param)
        ALP_Timeout_layout.addWidget(self.ALP_Timeout_chkbox)
        ALP_Timeout_layout.addWidget(self.ALP_Timeout_param)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.button_inspect)

        layout_sub.addWidget(self.button_delete)
        layout_sub.addLayout(button_layout)
        layout_sub.addWidget(self.outlier_return_button)
        layout_sub.addWidget(self.outlier_combo_box)
        layout_sub.addWidget(self.outlier_combo_label)
        layout_sub.addWidget(self.outlier_input_button)
        layout_sub.addWidget(self.outlier_input)
        layout_sub.addWidget(self.outlier_input_label)
        layout_sub.addLayout(ALP_Timeout_layout)
        layout_sub.addLayout(RNFS_layout)
        layout_sub.addLayout(IALP_layout)
        layout_sub.addLayout(ALP_layout)
        layout_sub.addWidget(self.cluster_select)
        layout_sub.addWidget(label_cluster_select)


        layout_tools = QHBoxLayout()
        layout_tools.addStretch()
        layout_tools.addLayout(layout_sub)


        self.setLayout(layout_tools)

    def update_defaults(self, event_defaults):
        self.event_defaults = event_defaults
        if not self.ALP_chkbox.isChecked():
            self.ALP_param.duration_edit.setText(str(event_defaults["ALP"]["window"]))
            self.ALP_param.delay_edit.setText(str(event_defaults["ALP"]["delay"]))
        if not self.IALP_chkbox.isChecked():
            self.IALP_chkbox.setChecked(True)
            self.IALP_param.duration_edit.setText(str(event_defaults["IALP"]["window"]))
            self.IALP_param.delay_edit.setText(str(event_defaults["IALP"]["delay"]))
        if not self.RNFS_chkbox.isChecked():
            self.RNFS_chkbox.setChecked(True)
            self.RNFS_param.duration_edit.setText(str(event_defaults["RNFS"]["window"]))
            self.RNFS_param.delay_edit.setText(str(event_defaults["RNFS"]["delay"]))
        if not self.ALP_Timeout_chkbox.isChecked():
            self.ALP_Timeout_chkbox.setChecked(True)
            self.ALP_Timeout_param.duration_edit.setText(str(event_defaults["ALP_Timeout"]["window"]))
            self.ALP_Timeout_param.delay_edit.setText(str(event_defaults["ALP_Timeout"]["delay"]))


    def add_outlier(self, click=None):
        potential_outlier = int(self.outlier_input.text())
        current_outliers = [int(self.outlier_combo_box.itemText(i)) for i in range(self.outlier_combo_box.count())]
        self.outlier_input.setText("")

        if potential_outlier in self.all_cells and potential_outlier not in current_outliers:
            self.outlier_combo_box.addItem(str(potential_outlier))
            self.outlier_combo_box.setCurrentIndex(self.outlier_combo_box.count()-1)

    def remove_outlier(self, click):
        val = self.outlier_combo_box.currentText
        if val != "":
            self.outlier_combo_box.removeItem(self.outlier_combo_box.currentIndex())



    def release_button(self):
        if self.ALP_chkbox.isChecked() or self.IALP_chkbox.isChecked() or self.RNFS_chkbox.isChecked() or self.ALP_Timeout_chkbox.isChecked():
            self.button.setEnabled(True)
        else:
            self.button.setEnabled(False)

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
        if self.ALP_Timeout_chkbox.isChecked():
            result["ALP_Timeout"] = {}
            result["ALP_Timeout"]["window"] = int(self.ALP_Timeout_param.duration_edit.text())
            result["ALP_Timeout"]["delay"] = int(self.ALP_Timeout_param.delay_edit.text())
        result["no_of_clusters"] = int(self.cluster_select.currentText())
        result["outliers"] = [int(self.outlier_combo_box.itemText(i)) for i in range(self.outlier_combo_box.count())]
        
        
        root_parent = self.parent().parent()
        root_parent.updateCluster(result)

    def inspect(self, event):
        root_parent = self.parent().parent()
        root_parent.startInspection()

    def delete(self, event):
        root_parent = self.parent().parent()
        root_parent.deleteSelection()
    
    def update(self, result, cell_list):
        self.all_cells = cell_list
        self.outlier_combo_box.clear()
        if "ALP" in result:
            self.ALP_chkbox.setChecked(True)
            self.ALP_param.duration_edit.setText(str(result["ALP"]["window"]))
            self.ALP_param.delay_edit.setText(str(result["ALP"]["delay"]))
        else:
            self.ALP_chkbox.setChecked(False)
            self.ALP_param.duration_edit.setText(str(self.event_defaults["ALP"]["window"]))
            self.ALP_param.delay_edit.setText(str(self.event_defaults["ALP"]["delay"]))
        if "IALP" in result:
            self.IALP_chkbox.setChecked(True)
            self.IALP_param.duration_edit.setText(str(result["IALP"]["window"]))
            self.IALP_param.delay_edit.setText(str(result["IALP"]["delay"]))
        else:
            self.IALP_chkbox.setChecked(False)
            self.IALP_param.duration_edit.setText(str(self.event_defaults["IALP"]["window"]))
            self.IALP_param.delay_edit.setText(str(self.event_defaults["IALP"]["delay"]))
        if "RNFS" in result:
            self.RNFS_chkbox.setChecked(True)
            self.RNFS_param.duration_edit.setText(str(result["RNFS"]["window"]))
            self.RNFS_param.delay_edit.setText(str(result["RNFS"]["delay"]))
        else:
            self.RNFS_chkbox.setChecked(False)
            self.RNFS_param.duration_edit.setText(str(self.event_defaults["RNFS"]["window"]))
            self.RNFS_param.delay_edit.setText(str(self.event_defaults["RNFS"]["delay"]))
        if "ALP_Timeout" in result:
            self.ALP_Timeout_chkbox.setChecked(True)
            self.ALP_Timeout_param.duration_edit.setText(str(result["ALP_Timeout"]["window"]))
            self.ALP_Timeout_param.delay_edit.setText(str(result["ALP_Timeout"]["delay"]))
        else:
            self.ALP_Timeout_chkbox.setChecked(False)
            self.ALP_Timeout_param.duration_edit.setText(str(self.event_defaults["ALP_Timeout"]["window"]))
            self.ALP_Timeout_param.delay_edit.setText(str(self.event_defaults["ALP_Timeout"]["delay"]))
        if "outliers" in result:
            for outlier in result["outliers"]:
                self.outlier_combo_box.addItem(str(outlier))
        
        self.cluster_select.setCurrentIndex(result["no_of_clusters"] - 2)



def hide_unhide(chkbox, param):
    if chkbox.isChecked():
        param.setEnabled(True)
    else:
        param.setEnabled(False)



class ParamWidget(QWidget):
    def __init__(self, name, event_defaults, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        duration_label = QLabel("Window: ")
        delay_label = QLabel("Delay: ")


        self.duration_edit = QLineEdit(str(event_defaults[name]["window"]))
        onlyInt = QIntValidator()
        onlyInt.setRange(20, 120)
        self.duration_edit.setValidator(onlyInt)

        self.delay_edit = QLineEdit(str(event_defaults[name]["delay"]))
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
        message = QLabel("Detected paths.json in your directory. Do you want to load the files?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class VisualizeClusterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout()
        self.groups = {}

        self.setLayout(self.layout)

    def removeVisualization(self, group, mouse, session, day):
        self.groups[group].removeVisualization(mouse, session, day)
        if self.groups[group].isEmpty():
            self.layout.removeWidget(self.groups[group])
            self.groups[group].setParent(None)
            del self.groups[group]
    
    def addVisualization(self, group, mouseID, image, session, day):
        if group not in self.groups:
            self.groups[group] = GroupGridLayout(group)
            self.layout.addWidget(self.groups[group])
        
        self.groups[group].addVisualization(mouseID, image, session, day)


class GroupGridLayout(QWidget):
    def __init__(self, group:str, parent=None):
        super().__init__(parent)
        self.group = group
        self.layout = QHBoxLayout()

        self.mice = {}
        
        self.setLayout(self.layout)

    def addVisualization(self, mouseID, image, session, day):
        if mouseID not in self.mice:
            self.mice[mouseID] = MouseGrid(mouseID, self.group)
            self.layout.addWidget(self.mice[mouseID])
        self.mice[mouseID].addVisualization(image, session, day)
    
    def removeVisualization(self, mouseID, session, day):
        self.mice[mouseID].removeVisualization(session, day)
        if self.mice[mouseID].isEmpty():
            self.layout.removeWidget(self.mice[mouseID])
            self.mice[mouseID].setParent(None)
            del self.mice[mouseID]

    
    def isEmpty(self):
        return self.layout.count() == 0
            
def deleteItemsOfLayout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            else:
                deleteItemsOfLayout(item.layout())

class MouseGrid(QWidget):
    def __init__(self, mouseID:str, group:str, parent=None):
        super().__init__(parent)
        self.mouseID = mouseID
        self.group = group
        self.days = []
        self.sessions = []
        self.images = {}
        self.day_labels = {}
        self.session_labels = {}
        self.needs_redraw = False
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(GridQLabel(f"{self.mouseID}/{self.group}"), 0, 0, Qt.AlignCenter)
        
    
    def addVisualization(self, image, session, day):
        imageViewer = Viewer(image, self.group, self.mouseID, session, day)
        day = int(day[1:])
        session = int(session[1:])
        self.images[f"{session}:{day}"] = imageViewer

        if day not in self.days:
            bisect.insort(self.days, day)
        if session not in self.sessions:
            bisect.insort(self.sessions, session)
        if day not in self.day_labels:
            self.day_labels[day] = GridQLabel("D" + str(day))
        if session not in self.session_labels:
            self.session_labels[session] = GridQLabel("S" + str(session))
        self.redrawGrid()
        
    def removeVisualization(self, session, day):
        day = int(day[1:])
        session = int(session[1:])
        session_index = self.sessions.index(session)
        day_index = self.days.index(day)
        wid = self.layout.itemAtPosition(session_index+1, day_index+1).widget()
        self.layout.removeWidget(wid)
        wid.setParent(None)
        del self.images[f"{session}:{day}"]

        # Now check if we need to remove the labels as well
        found_wid_row = False
        for i in range(1,len(self.days)+1):
            if self.layout.itemAtPosition(session_index+1, i) is not None:
                found_wid_row = True
                break
        found_wid_column = False
        for i in range(1,len(self.sessions)+1):
            if self.layout.itemAtPosition(i, day_index+1) is not None:
                found_wid_column = True
                break
        
        if not found_wid_row:
            wid = self.layout.itemAtPosition(session_index+1, 0).widget()
            self.layout.removeWidget(wid)
            wid.setParent(None)
            wid.deleteLater()
            self.sessions.remove(session)
            del self.session_labels[session]
        
        if not found_wid_column:
            wid = self.layout.itemAtPosition(0, day_index+1).widget()
            self.layout.removeWidget(wid)
            wid.setParent(None)
            wid.deleteLater()
            self.days.remove(day)
            del self.day_labels[day]

        self.redrawGrid()

        if self.layout.count() == 1:
            wid = self.layout.itemAtPosition(0, 0).widget()
            self.layout.removeWidget(wid)
            wid.setParent(None)
            wid.deleteLater()




            
    def redrawGrid(self):
        # Iterate through the lists check if they have corresponding images/labels
        for i, session in enumerate(self.sessions):
            for j, day in enumerate(self.days):
                if i == 0:
                    self.layout.addWidget(self.day_labels[day], i, j+1, Qt.AlignCenter)
                if j == 0:
                    self.layout.addWidget(self.session_labels[session], i+1, j, Qt.AlignCenter)
                if f"{session}:{day}" in self.images:
                    # Reset the Widget
                    self.images[f"{session}:{day}"].reset()
                    self.layout.addWidget(self.images[f"{session}:{day}"], i+1, j+1)
    
    def isEmpty(self):
        return self.layout.count() == 0
        



class Viewer(QGraphicsView):
    def __init__(self, image, group, mouseID, session, day, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.m_pixmapItem = self.scene().addPixmap(QPixmap())
        self.setAlignment(Qt.AlignCenter)


        self.p = self.palette()
        self.p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(self.p)
        
        self.selected = False
        self.mouseReleaseEvent=self.updateParams
        self.mouseDoubleClickEvent=self.inspect
        self.group = group
        self.mouseID = mouseID
        self.session = session
        self.day = day
        self.image = image
        self.createPixmap()

    def createPixmap(self):
        ov = (self.image*255).astype('uint8')
        qimg = QImage(ov, ov.shape[1], ov.shape[0], ov.shape[1] * 3, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)

    def reset(self):       
        self.resetTransform()

    @property
    def pixmap(self):
        return self.m_pixmapItem.pixmap()

    @pixmap.setter
    def pixmap(self, newPixmap):
        self.m_pixmapItem.setPixmap(newPixmap)
        self.fitInView(self.m_pixmapItem, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.m_pixmapItem, Qt.KeepAspectRatio)

    def changeToWhite(self):
        self.p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(self.p)
        self.selected = False

    def changeToRed(self):
        self.p.setColor(self.backgroundRole(), Qt.red)
        self.setPalette(self.p)
        self.selected = True

    def updateParams(self, event):
        root_parent = self.parent().parent().parent().parent().parent()
        root_parent.activateParams(self)
        
    def updateVisualization(self, image):
        ov = (image*255).astype('uint8')
        qimg = QImage(ov, ov.shape[1], ov.shape[0], ov.shape[1] * 3, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)

    def inspect(self, event):
        root_parent = self.parent().parent().parent().parent()
        root_parent.startInspection(self)

    def __eq__(self, other):
        return (self.group, self.session, self.day, self.mouseID) == (other.group, other.session, other.day, other.mouseID)

    def returnInfo(self):
        return self.group, self.session, self.day, self.mouseID


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class GridQLabel(QLabel):
    def __init__(self, parent=None, *args):
        super().__init__(parent, *args)

        self.setFrameStyle(QFrame.Panel | QFrame.Plain)
        self.setBaseSize(300,300)
        self.setStyleSheet("font-size: 14pt;")
        self.setLineWidth(2)


class InspectionWidget(QWidget):
    def __init__(self, session, main_win_ref, parent=None):
        super().__init__(parent)
        self.session = session
        self.total_neurons = len(self.session.clustering_result["all"]["ids"]) - len(self.session.outliers_list)
        self.selected_plot = 0
        self.cell_ids = None
        self.displaying = "None"
        self.current_labels = []
        self.name = f"{session.mouseID} {session.day} {session.session}"
        self.main_window_ref = main_win_ref

        # Brushes
        self.brushes_lines = {"ALP": pg.mkColor(255, 0, 0, 255),
                   "IALP": pg.mkColor(255, 255, 0, 255),
                   "RNFS": pg.mkColor(0, 255, 0, 255),
                   "ALP_Timeout": pg.mkColor(0, 0, 255, 255)
                   }
        self.brushes_boxes = {"ALP": pg.mkColor(255, 0, 0, 60),
                   "IALP": pg.mkColor(255, 255, 0, 60),
                   "RNFS": pg.mkColor(0, 255, 0, 60),
                   "ALP_Timeout": pg.mkColor(0, 0, 255, 60)
                   }
        self.colors = {"ALP": QColor(255, 0, 0, 255),
                   "IALP": QColor(255, 255, 0, 255),
                   "RNFS": QColor(0, 255, 0, 255),
                   "ALP_Timeout": QColor(0, 0, 255, 255)
                   }

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        mid_layout = QHBoxLayout()
        left_mid_layout = QHBoxLayout()

        mid_layout.setDirection(3)
        mid_layout.addStretch()


        # Select Cluster
        label_cluster_select = QLabel()
        label_cluster_select.setText("Pick cluster to visualize:")
        self.cluster_select = QComboBox()
        self.cluster_select.addItem("Show all")
        for i in range (1, session.no_of_clusters + 1):
            self.cluster_select.addItem(f"Cluster {i}")
        self.cluster_select.setCurrentIndex(0)
        self.cluster_select.currentIndexChanged.connect(self.indexChanged)

        # Select Cells
        w_cell_label = QLabel("Pick which cells to visualize (Hold ctrl):")
        self.w_cell_list = QListWidget()
        self.w_cell_list.setMaximumSize(250, 600)
        self.w_cell_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.w_cell_button = QPushButton("Visualize Selection")
        self.w_cell_button.clicked.connect(self.visualizeSignals)

        self.total_neurons_label = QLabel(f"Looking at {self.total_neurons} out of {self.total_neurons} neurons")
        self.total_neurons_label.setWordWrap(True)

        # Legend
        ALP_label = self.generate_legend_element("ALP")
        IALP_label = self.generate_legend_element("IALP")
        RNFS_label = self.generate_legend_element("RNFS")
        ALP_Timeout_label = self.generate_legend_element("ALP_Timeout")

        # Set View
        view_button = QPushButton("Set view to last clicked plot")
        view_button.clicked.connect(self.set_view)

        # Visualize Cluster
        self.imv = pg.ImageView()
        self.imv.setImage(self.session.clustering_result['all']['image'])
        self.imv.setMinimumWidth(800)

        # Add Context Menu Action
        button_pdf = QAction("&Export PDF Enhanced", self.imv.getView().menu)
        button_pdf.setStatusTip("Export Image using PDF with Vector Graphics Instead of Raster")
        button_pdf.triggered.connect(self.pdfExport)
        self.imv.getView().menu.addAction(button_pdf)

        # Dendrogram
        self.w_dendro = MplCanvas()
        self.session.get_dendrogram(self.w_dendro.axes)
        toolbar = NavigationToolbar(self.w_dendro, self)

        layout_mpl = QVBoxLayout()
        layout_mpl.addWidget(toolbar)
        layout_mpl.addWidget(self.w_dendro)

        # Visualize Signals
        self.w_signals = pg.GraphicsLayoutWidget()
        self.w_signals.scene().sigMouseClicked.connect(self.onClick)

        # Show labels
        show_all_button = QPushButton("Show/Hide All")
        show_all_button.clicked.connect(lambda: self.show_labels("all"))
        show_selected_button = QPushButton("Show/Hide Selected")
        show_selected_button.clicked.connect(lambda: self.show_labels("select"))
        button_layout = QHBoxLayout()
        button_layout.addWidget(show_all_button)
        button_layout.addWidget(show_selected_button)

        # Splitters
        splitter_vertical = QSplitter(Qt.Vertical)
        splitter_horizontal = QSplitter(Qt.Horizontal)        

        # Layouts
        left_layout.addWidget(label_cluster_select)
        left_layout.addWidget(self.cluster_select)
        left_layout.addWidget(self.imv)

        mid_layout.addWidget(view_button)
        mid_layout.addWidget(ALP_Timeout_label)
        mid_layout.addWidget(RNFS_label)
        mid_layout.addWidget(IALP_label)
        mid_layout.addWidget(ALP_label)
        mid_layout.addLayout(button_layout)
        mid_layout.addWidget(self.w_cell_button)
        mid_layout.addWidget(self.w_cell_list)
        mid_layout.addWidget(w_cell_label)
        mid_layout.addWidget(self.total_neurons_label)

        mpl_widget = QWidget()
        mpl_widget.setLayout(layout_mpl)
        splitter_vertical.addWidget(mpl_widget)
        splitter_vertical.addWidget(self.w_signals)

        left_mid_layout.addLayout(left_layout)
        left_mid_layout.addLayout(mid_layout)
        left_mid_widget = QWidget()
        left_mid_widget.setLayout(left_mid_layout)

        splitter_horizontal.addWidget(left_mid_widget)
        splitter_horizontal.addWidget(splitter_vertical)
        layout.addWidget(splitter_horizontal)

        self.setLayout(layout)
        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))

        for id in self.session.clustering_result["all"]['ids']:
            if id not in self.session.outliers_list:
                self.w_cell_list.addItem(str(id))

    def pdfExport(self):
        default_dir = os.getcwd()
        default_filename = os.path.join(default_dir, "image.pdf")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", default_filename, "PDF Files (*.pdf)"
        )
        if filename:
            unit_ids = [int(self.w_cell_list.item(x).text()) for x in range(self.w_cell_list.count())]
            cluster = self.cluster_select.currentIndex()
            self.session.get_pdf_format(unit_ids, cluster, filename)

    def show_labels(self, type):
        if type == "select":
            if self.displaying == "select":
                self.displaying = "None"
                self.remove_labels()
            else:                
                self.displaying = "select"
                self.remove_labels()
                selections = [int(item.text()) for item in self.w_cell_list.selectedItems()]
                if selections:
                    self.populate_image(selections)
                else:
                    self.displaying = None
        
        else:
            if self.displaying == "all":
                self.displaying = "None"
                self.remove_labels()
            else:
                self.displaying = "all"
                self.remove_labels()
                selections = [int(self.w_cell_list.item(x).text()) for x in range(self.w_cell_list.count())]
                self.populate_image(selections)

    def populate_image(self, selections):
        self.current_labels = []
        for sel in selections:
            x, y = self.session.centroids[sel]
            text = pg.TextItem(text=str(sel), anchor=(0.4,0.4), color=(255, 255, 255, 255))
            self.imv.addItem(text)
            text.setFont(QFont('Times', 7))
            # Calculate relative position
            text.setPos(round(x), round(y))
            self.current_labels.append(text)

    def remove_labels(self):
        for label in self.current_labels:
            self.imv.getView().removeItem(label)
        
        self.current_labels = []

    def set_view(self):
        if self.cell_ids is not None and self.selected_plot is not None:
            view = None
            focus_plot = self.w_signals.getItem(self.selected_plot, 0)
            view = focus_plot.getViewBox().viewRect()
            for i in range(len(self.cell_ids)):
                if i != self.selected_plot:
                    plot  = self.w_signals.getItem(i, 0)
                    plot.getViewBox().setRange(view)
            


    def generate_legend_element(self, name):
        label = QLabel()
        if name == "ALP_Timeout":
            label.setFixedSize(150, 40)
        else:
            label.setFixedSize(100, 40)
        pixmap = QPixmap(label.size())
        pixmap.fill(Qt.black)
        qp = QPainter(pixmap)
        pen = QPen(self.colors[name], 3)
        qp.setPen(pen)
        qp.fillRect(10, 10, 20, 20, QBrush(self.colors[name], Qt.SolidPattern))
        qp.drawText(pixmap.rect(), Qt.AlignCenter, name)
        qp.end()
        label.setPixmap(pixmap)

        return label

    def indexChanged(self, value):
        if value == -1:
            return # Necessary for the refresh step
        value = "all" if value == 0 else value
        self.imv.setImage(self.session.clustering_result[value]['image'])

        self.w_cell_list.clear()

        total_in_group = 0
        for id in self.session.clustering_result[value]['ids']:
            if id not in self.session.outliers_list:
                self.w_cell_list.addItem(str(id))
                total_in_group += 1
        
        self.total_neurons_label.setText(f"Looking at {total_in_group} out of {self.total_neurons} neurons")

    def visualizeSignals(self, event):
        self.cell_ids = [int(item.text()) for item in self.w_cell_list.selectedItems()]
        self.selected_plot = None

        if self.cell_ids:
            self.w_signals.clear()

            for i, id in enumerate(self.cell_ids):
                p = MyPlotWidget(id=i)
                self.w_signals.addItem(p, row=i, col=0)
                data = self.session.data['C'].sel(unit_id=id)
                p.plot(data)
                p.setTitle(f"Cell {id}")

                # Add event lines and boxes
                for event in self.session.events.values():
                    brush_line = self.brushes_lines[event.event_type]
                    brush_box = self.brushes_boxes[event.event_type]
                    for t in event.timesteps:
                        p.addItem(pg.InfiniteLine(t, pen=brush_line, movable=False, name=f"{event}"))
                    
                    for w in event.windows:
                        start, end = w
                        p.addItem(pg.LinearRegionItem((start, end), brush=brush_box, pen=brush_box, movable=False))

    def onClick(self, event):
        # Funky stuff to get the PlotItem clicked
        current_height = self.w_signals.range.height()
        click_height = event.scenePos().y()
        self.selected_plot = round(click_height / current_height * (len(self.cell_ids) - 1))
        if self.selected_plot > len(self.cell_ids) - 1:
            self.selected_plot = len(self.cell_ids) - 1


    def closeEvent(self, event):
        super(InspectionWidget, self).closeEvent(event)
        self.main_window_ref.removeWindow(self.name)

    def refresh(self):
        self.imv.setImage(self.session.clustering_result['all']['image'])

        self.cluster_select.clear()
        self.cluster_select.addItem("Show all")
        for i in range (1, self.session.no_of_clusters + 1):
            self.cluster_select.addItem(f"Cluster {i}")
        self.cluster_select.setCurrentIndex(0)

        self.w_cell_list.clear()
        for id in self.session.clustering_result["all"]['ids']:
            if id not in self.session.outliers_list:
                self.w_cell_list.addItem(str(id))

        self.session.get_dendrogram(self.w_dendro.axes)

        self.w_signals.clear()


class MyPlotWidget(PlotItem):
    def __init__(self, id=None, **kwargs):
        super(MyPlotWidget, self).__init__(**kwargs)
        self.id = id