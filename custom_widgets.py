from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QWidget,
                            QCheckBox, QGridLayout, QFrame, QGraphicsView, QGraphicsScene, QPushButton, 
                            QComboBox, QListWidget, QAbstractItemView)
from PyQt5.QtGui import QIntValidator, QImage, QPixmap, QPalette
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtGui import QPixmap


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
        self.IALP_chkbox.stateChanged.connect(self.release_button)
        self.RNFS_chkbox = QCheckBox("RNFS")
        self.RNFS_chkbox.stateChanged.connect(lambda: hide_unhide(self.RNFS_chkbox, self.RNFS_param))
        self.RNFS_chkbox.stateChanged.connect(self.release_button)

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

class ToolWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        label_cluster_select = QLabel()
        label_cluster_select.setText("Pick number of clusters:")
        self.cluster_select = QComboBox()
        for i in range (2, 20):
            self.cluster_select.addItem(str(i))
        self.cluster_select.setCurrentIndex(2)

        self.button = QPushButton("Update")
        self.button.setEnabled(False)
        self.button.setFixedWidth(120)
        self.button.clicked.connect(self.get_result)

        self.button_inspect = QPushButton("Inspect Cluster")
        self.button_inspect.setFixedWidth(120)
        self.button_inspect.clicked.connect(self.inspect)

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

        self.ALP_param = ParamWidget()
        self.ALP_param.setEnabled(False)
        self.IALP_param = ParamWidget()
        self.IALP_param.setEnabled(False)
        self.RNFS_param = ParamWidget()
        self.RNFS_param.setEnabled(False)

        layout_sub = QVBoxLayout()
        layout_sub.addStretch()
        layout_sub.setDirection(3)
        ALP_layout = QVBoxLayout()
        IALP_layout = QVBoxLayout()
        RNFS_layout = QVBoxLayout()

        ALP_layout.addWidget(self.ALP_chkbox)
        ALP_layout.addWidget(self.ALP_param)
        IALP_layout.addWidget(self.IALP_chkbox)
        IALP_layout.addWidget(self.IALP_param)
        RNFS_layout.addWidget(self.RNFS_chkbox)
        RNFS_layout.addWidget(self.RNFS_param)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.button_inspect)

        layout_sub.addLayout(button_layout)
        layout_sub.addLayout(RNFS_layout)
        layout_sub.addLayout(IALP_layout)
        layout_sub.addLayout(ALP_layout)
        layout_sub.addWidget(self.cluster_select)
        layout_sub.addWidget(label_cluster_select)
        
        
        

        layout_tools = QHBoxLayout()
        layout_tools.addStretch()
        layout_tools.addLayout(layout_sub)


        self.setLayout(layout_tools)

    def release_button(self):
        if self.ALP_chkbox.isChecked() or self.IALP_chkbox.isChecked() or self.RNFS_chkbox.isChecked():
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
        result["no_of_clusters"] = int(self.cluster_select.currentText())
        
        
        root_parent = self.parent().parent()
        root_parent.updateCluster(result)

    def inspect(self, event):
        root_parent = self.parent().parent()
        root_parent.startInspection()
    
    def update(self, result):
        if "ALP" in result:
            self.ALP_chkbox.setChecked(True)
            self.ALP_param.duration_edit.setText(str(result["ALP"]["window"]))
            self.ALP_param.delay_edit.setText(str(result["ALP"]["delay"]))
        else:
            self.ALP_chkbox.setChecked(False)
            self.ALP_param.duration_edit.setText("20")
            self.ALP_param.delay_edit.setText("0")
        if "IALP" in result:
            self.IALP_chkbox.setChecked(True)
            self.IALP_param.duration_edit.setText(str(result["IALP"]["window"]))
            self.IALP_param.delay_edit.setText(str(result["IALP"]["delay"]))
        else:
            self.IALP_chkbox.setChecked(False)
            self.IALP_param.duration_edit.setText("20")
            self.IALP_param.delay_edit.setText("0")
        if "RNFS" in result:
            self.RNFS_chkbox.setChecked(True)
            self.RNFS_param.duration_edit.setText(str(result["RNFS"]["window"]))
            self.RNFS_param.delay_edit.setText(str(result["RNFS"]["delay"]))
        else:
            self.RNFS_chkbox.setChecked(False)
            self.RNFS_param.duration_edit.setText("20")
            self.RNFS_param.delay_edit.setText("0")
        
        self.cluster_select.setCurrentIndex(result["no_of_clusters"] - 2)



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
        message = QLabel("Detected paths.json in your directory. Do you want to load the files?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class VisualizeClusterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.grids = {}
        self.grids["cocaine"] = GridLayoutWidget("Cocaine")
        self.grids["saline"] = GridLayoutWidget("Saline")

        layout.addWidget(self.grids["cocaine"])
        layout.addWidget(self.grids["saline"])

        self.setLayout(layout)

class GridLayoutWidget(QWidget):
    def __init__(self, type:str, parent=None):
        super().__init__(parent)
        self.type = type
        self.layout = QHBoxLayout()

        self.mouse_group = {}

        
        self.setLayout(self.layout)
    
    def addGrid(self, mouseID: str):
        layout = QGridLayout()
        layout.addWidget(GridQLabel(f"{mouseID}/{self.type}"), 0, 0, Qt.AlignCenter)
        layout.addWidget(GridQLabel("First 15 min"), 1, 0, Qt.AlignCenter)
        layout.addWidget(GridQLabel("Last 15 min"), 2, 0, Qt.AlignCenter)
        layout.addWidget(GridQLabel("First Day"), 0, 1, Qt.AlignCenter)
        layout.addWidget(GridQLabel("Last Day"), 0, 2, Qt.AlignCenter)

        self.mouse_group[mouseID] = layout
        self.layout.addLayout(layout)
        

    def addVisualization(self, group, mouseID, image, x, y):
        ov = (image*255).astype('uint8')
        qimg = QImage(ov, ov.shape[1], ov.shape[0], ov.shape[1] * 3, QImage.Format_RGB888)
        imageViewer = Viewer(group, mouseID, x, y)
        imageViewer.pixmap = QPixmap.fromImage(qimg)
        self.mouse_group[mouseID].addWidget(imageViewer, x, y)



    


class Viewer(QGraphicsView):
    def __init__(self, group, mouseID, x, y, parent=None):
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
        self.x = x
        self.y = y


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
        root_parent = self.parent().parent().parent().parent()
        root_parent.activateParams(self)
        
    def updateVisualization(self, image):
        ov = (image*255).astype('uint8')
        qimg = QImage(ov, ov.shape[1], ov.shape[0], ov.shape[1] * 3, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)

    def inspect(self, event):
        root_parent = self.parent().parent().parent().parent()
        root_parent.startInspection(self)

    def __eq__(self, other):
        return (self.group, self.x, self.y, self.mouseID) == (other.group, other.x, other.y, other.mouseID)

    def returnInfo(self):
        return self.group, self.x, self.y, self.mouseID


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
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        mid_layout = QHBoxLayout()

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

        # Visualize Cluster
        self.imv = pg.ImageView()
        self.imv.setImage(self.session.clustering_result['all']['image'])
        self.imv.setMinimumWidth(800)

        # Dendrogram
        self.w_dendro = MplCanvas()
        self.session.get_dendrogram(self.w_dendro.axes)

        # Visualize Signals
        self.w_signals = pg.GraphicsLayoutWidget()
        

        # Layouts
        left_layout.addWidget(label_cluster_select)
        left_layout.addWidget(self.cluster_select)
        left_layout.addWidget(self.imv)

        mid_layout.addWidget(self.w_cell_button)
        mid_layout.addWidget(self.w_cell_list)
        mid_layout.addWidget(w_cell_label)

        right_layout.addWidget(self.w_dendro)
        right_layout.addWidget(self.w_signals)

        layout.addLayout(left_layout)
        layout.addLayout(mid_layout)
        layout.addLayout(right_layout)

        self.setLayout(layout)

        for id in self.session.clustering_result["all"]['ids']:
            self.w_cell_list.addItem(str(id))

    def indexChanged(self, value):
        value = "all" if value == 0 else value
        self.imv.setImage(self.session.clustering_result[value]['image'])

        self.w_cell_list.clear()

        for id in self.session.clustering_result[value]['ids']:
            self.w_cell_list.addItem(str(id))

    def visualizeSignals(self, event):
        cell_ids = [int(item.text()) for item in self.w_cell_list.selectedItems()]

        if cell_ids:
            self.w_signals.clear()

            for i, id in enumerate(cell_ids):
                p = self.w_signals.addPlot(row=i, col=0)
                p.plot(self.session.values[id])
                p.setTitle(f"Cell {id}")

    
            

