from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QWidget,
                            QCheckBox, QGridLayout, QFrame, QGraphicsView, QGraphicsScene, QPushButton, 
                            QComboBox, QMainWindow, QBoxLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import (QIntValidator, QDoubleValidator, QImage, QPixmap)
from PyQt5.QtCore import (Qt)
import bisect
from ..core.backend import DataInstance

from ..core.genetic_algorithm import Genetic_Algorithm


class ParamDialog(QDialog):
    def __init__(self, event_defaults, parent=None):
        super().__init__(parent)
        self.events = list(event_defaults.keys())[:-1]
        self.setWindowTitle("Specify Parameters")

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

        self.event_chkboxes = []
        self.event_param = []

        for idx,i in enumerate(self.events):
            self.event_chkboxes.append(QCheckBox(i)) 
            self.event_chkboxes[idx].stateChanged.connect(lambda: hide_unhide(self.event_chkboxes[idx], self.event_param[idx]))
            self.event_chkboxes[idx].stateChanged.connect(self.release_button)
            
            



        # self.ALP_chkbox = QCheckBox("ALP")
        # self.ALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.ALP_chkbox, self.ALP_param))
        # self.ALP_chkbox.stateChanged.connect(self.release_button)
        # self.IALP_chkbox = QCheckBox("IALP")
        # self.IALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.IALP_chkbox, self.IALP_param))
        # self.IALP_chkbox.stateChanged.connect(self.release_button)
        # self.RNFS_chkbox = QCheckBox("RNFS")
        # self.RNFS_chkbox.stateChanged.connect(lambda: hide_unhide(self.RNFS_chkbox, self.RNFS_param))
        # self.RNFS_chkbox.stateChanged.connect(self.release_button)
        # self.ALP_Timeout_chkbox = QCheckBox("ALP_Timeout")
        # self.ALP_Timeout_chkbox.stateChanged.connect(lambda: hide_unhide(self.ALP_Timeout_chkbox, self.ALP_Timeout_param))
        # self.ALP_Timeout_chkbox.stateChanged.connect(self.release_button)

        distance_metric_label = QLabel("Distance Metric")
        self.distance_metric_combo = QComboBox()

        for idx,i in enumerate(self.events):
            self.event_param.append(ParamWidget(i, event_defaults))
            self.event_param[idx].setEnabled(False)

        # self.ALP_param = ParamWidget("ALP", event_defaults)
        # self.ALP_param.setEnabled(False)
        # self.IALP_param = ParamWidget("IALP", event_defaults)
        # self.IALP_param.setEnabled(False)
        # self.RNFS_param = ParamWidget("RNFS", event_defaults)
        # self.RNFS_param.setEnabled(False)
        # self.ALP_Timeout_param = ParamWidget("ALP_Timeout", event_defaults)
        # self.ALP_Timeout_param.setEnabled(False)
        self.distance_metric_combo.addItems(DataInstance.distance_metric_list)
        self.distance_metric_combo.setCurrentIndex(self.distance_metric_combo.findText(event_defaults["distance_metric"]))

        layout_param = QHBoxLayout()
        event_layout = []
        for i,j in zip(self.event_chkboxes,self.event_param):
            single_layout = QVBoxLayout()
            single_layout.addWidget(i)
            single_layout.addWidget(j)
            event_layout.append(single_layout)

        # ALP_layout = QVBoxLayout()
        # IALP_layout = QVBoxLayout()
        # RNFS_layout = QVBoxLayout()
        # ALP_Timeout_layout = QVBoxLayout()
        distance_metric_layout = QVBoxLayout()


        # ALP_layout.addWidget(self.event_chkboxes[0])
        # ALP_layout.addWidget(self.event_param[0])
        # IALP_layout.addWidget(self.event_chkboxes[1])
        # IALP_layout.addWidget(self.event_param[1])
        # RNFS_layout.addWidget(self.event_chkboxes[2])
        # RNFS_layout.addWidget(self.event_param[2])
        # ALP_Timeout_layout.addWidget(self.event_chkboxes[3])
        # ALP_Timeout_layout.addWidget(self.event_param[3])
        distance_metric_layout.addWidget(distance_metric_label)
        distance_metric_layout.addWidget(self.distance_metric_combo)

        for i in event_layout:
            layout_param.addLayout(i)

        # layout_param.addLayout(ALP_layout)
        # layout_param.addLayout(IALP_layout)
        # layout_param.addLayout(RNFS_layout)
        # layout_param.addLayout(ALP_Timeout_layout)
        layout_param.addLayout(distance_metric_layout)

        layout = QVBoxLayout()
        layout.addLayout(layout_param)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def release_button(self):
        mark = 0
        for i in self.event_chkboxes:
            if i.isChecked():
                mark += 1
        if mark > 0:
            self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        else:
            self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

        # if self.ALP_chkbox.isChecked() or self.IALP_chkbox.isChecked() or self.RNFS_chkbox.isChecked() or self.ALP_Timeout_chkbox.isChecked():
        #     self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        # else:
        #     self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def get_result(self):
        result = {}
        for i, j, l in zip(self.event_chkboxes,self.events,self.event_param):
            if i.isChecked():
                result[j] = {}
                result[j]["window"] = int(l.duration_edit.text())
                result[j]["delay"] = int(l.delay_edit.text())
        # if self.ALP_chkbox.isChecked():
        #     result["ALP"] = {}
        #     result["ALP"]["window"] = int(self.ALP_param.duration_edit.text())
        #     result["ALP"]["delay"] = int(self.ALP_param.delay_edit.text())
        # if self.IALP_chkbox.isChecked():
        #     result["IALP"] = {}
        #     result["IALP"]["window"] = int(self.IALP_param.duration_edit.text())
        #     result["IALP"]["delay"] = int(self.IALP_param.delay_edit.text())
        # if self.RNFS_chkbox.isChecked():
        #     result["RNFS"] = {}
        #     result["RNFS"]["window"] = int(self.RNFS_param.duration_edit.text())
        #     result["RNFS"]["delay"] = int(self.RNFS_param.delay_edit.text())
        # if self.ALP_Timeout_chkbox.isChecked():
        #     result["ALP_Timeout"] = {}
        #     result["ALP_Timeout"]["window"] = int(self.ALP_Timeout_param.duration_edit.text())
        #     result["ALP_Timeout"]["delay"] = int(self.ALP_Timeout_param.delay_edit.text())
        result["distance_metric"] = self.distance_metric_combo.currentText()
        return result


class UpdateDialog(QDialog):
    def __init__(self, event_defaults, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Specify New Defaults")

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        self.events = list(event_defaults.keys())[:-1]
        self.event_label = []
        for i in self.events:
            self.event_label.append(QLabel(i))

        # self.ALP_label = QLabel("ALP")
        # self.IALP_label = QLabel("IALP")
        # self.RNFS_label = QLabel("RNFS")
        # self.ALP_Timeout_label = QLabel("ALP_Timeout")

        self.event_param = []
        for i in self.events:
            self.event_param.append(ParamWidget(i,event_defaults))

        # self.ALP_param = ParamWidget("ALP", event_defaults)
        # self.IALP_param = ParamWidget("IALP", event_defaults)
        # self.RNFS_param = ParamWidget("RNFS", event_defaults)
        # self.ALP_Timeout_param = ParamWidget("ALP_Timeout", event_defaults)

        distance_metric_label = QLabel("Distance Metric")
        self.distance_metric_combo = QComboBox()
        self.distance_metric_combo.addItems(DataInstance.distance_metric_list)
        self.distance_metric_combo.setCurrentIndex(self.distance_metric_combo.findText(event_defaults["distance_metric"]))    

        layout_param = QHBoxLayout()
        event_layout = []
        for i,j in zip(self.event_label,self.event_param):
            single_layout = QBoxLayout()
            single_layout.addWidget(i)
            single_layout.addWidget(j)
            event_layout.append(single_layout)

        # ALP_layout = QVBoxLayout()
        # IALP_layout = QVBoxLayout()
        # RNFS_layout = QVBoxLayout()
        # ALP_Timeout_layout = QVBoxLayout()
        distance_metric_layout = QVBoxLayout()

        # ALP_layout.addWidget(self.ALP_label)
        # ALP_layout.addWidget(self.ALP_param)
        # IALP_layout.addWidget(self.IALP_label)
        # IALP_layout.addWidget(self.IALP_param)
        # RNFS_layout.addWidget(self.RNFS_label)
        # RNFS_layout.addWidget(self.RNFS_param)
        # ALP_Timeout_layout.addWidget(self.ALP_Timeout_label)
        # ALP_Timeout_layout.addWidget(self.ALP_Timeout_param)
        distance_metric_layout.addWidget(distance_metric_label)
        distance_metric_layout.addWidget(self.distance_metric_combo)

        for i in event_layout:
            layout_param.addLayout(i)
        # layout_param.addLayout(ALP_layout)
        # layout_param.addLayout(IALP_layout)
        # layout_param.addLayout(RNFS_layout)
        # layout_param.addLayout(ALP_Timeout_layout)
        layout_param.addWidget(distance_metric_layout)

        layout = QVBoxLayout()
        layout.addLayout(layout_param)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def get_result(self):
        result = {}
        for i,j in zip(self.events, self.event_param):
            result[i] = {}
            result[i]["window"] = int(j.duration_edit.text())
            result[i]["delay"] = int(j.delay_edit.text())

        # result["ALP"] = {}
        # result["ALP"]["window"] = int(self.ALP_param.duration_edit.text())
        # result["ALP"]["delay"] = int(self.ALP_param.delay_edit.text())
        # result["IALP"] = {}
        # result["IALP"]["window"] = int(self.IALP_param.duration_edit.text())
        # result["IALP"]["delay"] = int(self.IALP_param.delay_edit.text())
        # result["RNFS"] = {}
        # result["RNFS"]["window"] = int(self.RNFS_param.duration_edit.text())
        # result["RNFS"]["delay"] = int(self.RNFS_param.delay_edit.text())
        # result["ALP_Timeout"] = {}
        # result["ALP_Timeout"]["window"] = int(self.ALP_Timeout_param.duration_edit.text())
        # result["ALP_Timeout"]["delay"] = int(self.ALP_Timeout_param.delay_edit.text())
        result["distance_metric"] = self.distance_metric_combo.currentText()
        
        return result

class EventComponent(QWidget):
    def __init__(self,event_name):
        super().__init__()
        self.create_event_component(event_name)

    def create_event_component(self,event_name):
        layout = QVBoxLayout()
        checkbox_layout = QHBoxLayout()
        self.checkbox = QCheckBox(event_name)
        self.checkbox.setChecked(False)
        checkbox_layout.addWidget(self.checkbox)

        self.window_input_label = QLabel("Window:")
        self.window_input = QLineEdit()
        self.window_input.setEnabled(False)


        self.delay_input_label = QLabel("Delay:")
        self.delay_input = QLineEdit()
        self.delay_input.setEnabled(False)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.window_input_label)
        input_layout.addWidget(self.window_input)
        input_layout.addWidget(self.delay_input_label)
        input_layout.addWidget(self.delay_input)


        layout.addLayout(checkbox_layout)
        layout.addLayout(input_layout)

        self.checkbox.stateChanged.connect(self.update_input_state)
        self.setLayout(layout)

    def update_input_state(self,state):
        self.window_input.setEnabled(state == 2)
        self.delay_input.setEnabled(state == 2)


# update the clustering widget
class ClusteringToolWidget(QWidget):
    def __init__(self, main_ref, event_defaults, parent=None):
        super().__init__(parent)
        self.all_cells = None
        self.event_defaults = event_defaults
        self.events = list(event_defaults.keys())[:-1]
        self.main_ref = main_ref

        label_cluster_select = QLabel()
        label_cluster_select.setText("Pick number of clusters:")
        self.cluster_select = QComboBox()
        for i in range (2, 20):
            self.cluster_select.addItem(str(i))
        self.cluster_select.setCurrentIndex(2)

        self.btn_update = QPushButton("Update")
        self.btn_update.setStyleSheet("background-color : green")
        self.btn_update.setEnabled(False)
        self.btn_update.clicked.connect(self.get_result)

        self.button_inspect = QPushButton("Inspect Cluster")
        self.button_inspect.setStyleSheet("background-color : green")
        self.button_inspect.clicked.connect(self.inspect)

        self.button_delete = QPushButton("Delete Cluster")
        self.button_delete.setStyleSheet("background-color : red")
        self.button_delete.clicked.connect(self.delete)

        self.event_chkboxes = []
        self.event_component1 = EventComponent('ALP')
        self.event_component2 = EventComponent('IALP')
        self.event_component3 = EventComponent('RNFS')
        self.event_component4 = EventComponent('ALP_Timeout')
        for idx,i in enumerate(self.events):
            single_chkbox = QCheckBox(i)
            single_chkbox.stateChanged.connect(lambda: hide_unhide(self.event_chkboxes[idx],self.event_param[idx]))
            single_chkbox.stateChanged.connect(self.release_button)
            self.event_chkboxes.append(single_chkbox)

        # self.ALP_chkbox = QCheckBox("ALP")
        # self.ALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.ALP_chkbox, self.ALP_param))
        # self.ALP_chkbox.stateChanged.connect(self.release_button)
        # self.IALP_chkbox = QCheckBox("IALP")
        # self.IALP_chkbox.stateChanged.connect(lambda: hide_unhide(self.IALP_chkbox, self.IALP_param))
        # self.IALP_chkbox.stateChanged.connect(self.release_button)
        # self.RNFS_chkbox = QCheckBox("RNFS")
        # self.RNFS_chkbox.stateChanged.connect(lambda: hide_unhide(self.RNFS_chkbox, self.RNFS_param))
        # self.RNFS_chkbox.stateChanged.connect(self.release_button)
        # self.ALP_Timeout_chkbox = QCheckBox("ALP_Timeout")
        # self.ALP_Timeout_chkbox.stateChanged.connect(lambda: hide_unhide(self.ALP_Timeout_chkbox, self.ALP_Timeout_param))
        # self.ALP_Timeout_chkbox.stateChanged.connect(self.release_button)

        self.event_param = []
        for i in self.events:
            single_param = ParamWidget(i, self.event_defaults)
            single_param.setEnabled(False)
            self.event_param.append(single_param)
        # self.ALP_param = ParamWidget("ALP", self.event_defaults)
        # self.ALP_param.setEnabled(False)
        # self.IALP_param = ParamWidget("IALP", self.event_defaults)
        # self.IALP_param.setEnabled(False)
        # self.RNFS_param = ParamWidget("RNFS", self.event_defaults)
        # self.RNFS_param.setEnabled(False)
        # self.ALP_Timeout_param = ParamWidget("ALP_Timeout", self.event_defaults)
        # self.ALP_Timeout_param.setEnabled(False)

        distance_metric_label = QLabel("Distance Metric")
        self.distance_metric_combo = QComboBox()
        self.distance_metric_combo.addItems(DataInstance.distance_metric_list)
        self.distance_metric_combo.setCurrentIndex(self.distance_metric_combo.findText(event_defaults["distance_metric"]))

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

        # to do
        layout_sub = QVBoxLayout()
        layout_sub.addStretch()
        layout_sub.setDirection(3)
        event_layout = []

        for idx in range(len(self.events)):
            single_layout = QVBoxLayout()
            single_layout.addWidget(self.event_chkboxes[idx])
            single_layout.addWidget(self.event_param[idx])
            event_layout.append(single_layout)
        # ALP_layout = QVBoxLayout()
        # IALP_layout = QVBoxLayout()
        # RNFS_layout = QVBoxLayout()
        # ALP_Timeout_layout = QVBoxLayout()
        distance_metric_layout = QVBoxLayout()

        # ALP_layout.addWidget(self.ALP_chkbox)
        # ALP_layout.addWidget(self.ALP_param)
        # IALP_layout.addWidget(self.IALP_chkbox)
        # IALP_layout.addWidget(self.IALP_param)
        # RNFS_layout.addWidget(self.RNFS_chkbox)
        # RNFS_layout.addWidget(self.RNFS_param)
        # ALP_Timeout_layout.addWidget(self.ALP_Timeout_chkbox)
        # ALP_Timeout_layout.addWidget(self.ALP_Timeout_param)
        distance_metric_layout.addWidget(distance_metric_label)
        distance_metric_layout.addWidget(self.distance_metric_combo)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_update)
        button_layout.addWidget(self.button_inspect)

        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(self.button_delete)

        layout_sub.addLayout(button_layout2)
        layout_sub.addLayout(button_layout)
        layout_sub.addWidget(self.outlier_return_button)
        layout_sub.addWidget(self.outlier_combo_box)
        layout_sub.addWidget(self.outlier_combo_label)
        layout_sub.addWidget(self.outlier_input_button)
        layout_sub.addWidget(self.outlier_input)
        layout_sub.addWidget(self.outlier_input_label)
        for i in event_layout:
            layout_sub.addLayout(i)
        # layout_sub.addLayout(ALP_Timeout_layout)
        # layout_sub.addLayout(RNFS_layout)
        # layout_sub.addLayout(IALP_layout)
        # layout_sub.addLayout(ALP_layout)
        layout_sub.addWidget(self.cluster_select)
        layout_sub.addWidget(label_cluster_select)
        layout_sub.addLayout(distance_metric_layout)

        # Convert into a Widget so we can hide it on command
        self.wid_sub = QWidget()
        self.wid_sub.setLayout(layout_sub)
        self.wid_sub.setHidden(True)

        # Set up default view

        # Input section
        # Pre Bin Number
        layout_event_ALP = QVBoxLayout()

        self.checkBox1 = QCheckBox("ALP")
        self.checkBox1.setChecked(False)
        layout_event_ALP.addWidget(self.checkBox1)
        layout_event_input = QHBoxLayout()
        
        label_pre_bin = QLabel("Window:")
        self.input_pre_bin = QLineEdit()
        onlyInt = QIntValidator()
        self.input_pre_bin.setValidator(onlyInt)
        layout_pre_bin = QHBoxLayout()
        layout_pre_bin.addWidget(label_pre_bin)
        layout_pre_bin.addWidget(self.input_pre_bin)
        layout_event_input.addLayout(layout_pre_bin)

        # Post Bin Number 
        label_post_bin = QLabel("Delay")
        self.input_post_bin = QLineEdit()
        self.input_post_bin.setValidator(onlyInt)
        layout_post_bin = QHBoxLayout()
        layout_post_bin.addWidget(label_post_bin)
        layout_post_bin.addWidget(self.input_post_bin)

        layout_event_input.addLayout(layout_post_bin)
        layout_event_ALP.addLayout(layout_event_input)

        # # Crossover Rate
        # label_cross_rate = QLabel("Crossover Rate:")
        # self.input_cross_rate = QLineEdit("0.5")
        # onlyFloat = QDoubleValidator()
        # self.input_cross_rate.setValidator(onlyFloat)
        # layout_cross_rate = QHBoxLayout()
        # layout_cross_rate.addWidget(label_cross_rate)
        # layout_cross_rate.addWidget(self.input_cross_rate)

        # Mutation Rate
        # label_mut_rate = QLabel("Mutation Rate:")
        # self.input_mut_rate = QLineEdit("0.15")
        # self.input_mut_rate.setValidator(onlyFloat)
        # layout_mut_rate = QHBoxLayout()
        # layout_mut_rate.addWidget(label_mut_rate)
        # layout_mut_rate.addWidget(self.input_mut_rate)

        # Start Clustering button
        self.btn_clustering = QPushButton("Start Clustering")
        self.btn_clustering.clicked.connect(self.init_clustering)
        self.btn_clustering.setStyleSheet("background-color : green")
        


        self.layout_tools = QVBoxLayout()
        self.layout_tools.addWidget(self.wid_sub)
        # self.layout_tools.addLayout(layout_event_ALP)
        self.layout_tools.addWidget(self.event_component1)
        self.layout_tools.addWidget(self.event_component2)
        self.layout_tools.addWidget(self.event_component3)
        self.layout_tools.addWidget(self.event_component4)
        self.layout_tools.addWidget(self.btn_clustering)


        self.setLayout(self.layout_tools)

    def display_params(self):
        self.wid_sub.setHidden(False)
        self.btn_clustering.setHidden(True)

    def display_default(self):
        self.wid_sub.setHidden(True)
        self.btn_clustering.setHidden(False)

    def init_clustering(self):
        self.main_ref.load_clustering_params()

    def update_defaults(self, event_defaults):
        self.event_defaults = event_defaults
        for idx,i in enumerate(self.events):
            if not self.event_chkboxes[idx].isChecked():
                self.event_param[idx].duration_edit.setText(str(event_defaults[i]["window"]))
                self.event_param[idx].delay_edit.setText(str(event_defaults[i]["delay"]))
        # if not self.ALP_chkbox.isChecked():
        #     self.ALP_param.duration_edit.setText(str(event_defaults["ALP"]["window"]))
        #     self.ALP_param.delay_edit.setText(str(event_defaults["ALP"]["delay"]))
        # if not self.IALP_chkbox.isChecked():
        #     self.IALP_chkbox.setChecked(True)
        #     self.IALP_param.duration_edit.setText(str(event_defaults["IALP"]["window"]))
        #     self.IALP_param.delay_edit.setText(str(event_defaults["IALP"]["delay"]))
        # if not self.RNFS_chkbox.isChecked():
        #     self.RNFS_chkbox.setChecked(True)
        #     self.RNFS_param.duration_edit.setText(str(event_defaults["RNFS"]["window"]))
        #     self.RNFS_param.delay_edit.setText(str(event_defaults["RNFS"]["delay"]))
        # if not self.ALP_Timeout_chkbox.isChecked():
        #     self.ALP_Timeout_chkbox.setChecked(True)
        #     self.ALP_Timeout_param.duration_edit.setText(str(event_defaults["ALP_Timeout"]["window"]))
        #     self.ALP_Timeout_param.delay_edit.setText(str(event_defaults["ALP_Timeout"]["delay"]))
        self.distance_metric_combo.setCurrentIndex(self.distance_metric_combo.findText(event_defaults["distance_metric"]))


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


    # double check
    def release_button(self):
        self.btn_update.setEnabled(False)
        for i in self.event_chkboxes:
            if i.isChecked():
                self.btn_update.setEnabled(True)
        # if self.ALP_chkbox.isChecked() or self.IALP_chkbox.isChecked() or self.RNFS_chkbox.isChecked() or self.ALP_Timeout_chkbox.isChecked():
        #     self.btn_update.setEnabled(True)
        # else:
        #     self.btn_update.setEnabled(False)

    def get_result(self):
        result = {}
        for idx, i in self.events:
            if self.event_chkboxes[idx].isChecked():
                result[i] = {}
                result[i]["window"] = int(self.event_param[i].duration_edit.text())
                result[i]["delay"] = int(self.event_param[i].delay_edit.text())
        # if self.ALP_chkbox.isChecked():
        #     result["ALP"] = {}
        #     result["ALP"]["window"] = int(self.ALP_param.duration_edit.text())
        #     result["ALP"]["delay"] = int(self.ALP_param.delay_edit.text())
        # if self.IALP_chkbox.isChecked():
        #     result["IALP"] = {}
        #     result["IALP"]["window"] = int(self.IALP_param.duration_edit.text())
        #     result["IALP"]["delay"] = int(self.IALP_param.delay_edit.text())
        # if self.RNFS_chkbox.isChecked():
        #     result["RNFS"] = {}
        #     result["RNFS"]["window"] = int(self.RNFS_param.duration_edit.text())
        #     result["RNFS"]["delay"] = int(self.RNFS_param.delay_edit.text())
        # if self.ALP_Timeout_chkbox.isChecked():
        #     result["ALP_Timeout"] = {}
        #     result["ALP_Timeout"]["window"] = int(self.ALP_Timeout_param.duration_edit.text())
        #     result["ALP_Timeout"]["delay"] = int(self.ALP_Timeout_param.delay_edit.text())
        result["no_of_clusters"] = int(self.cluster_select.currentText())
        result["outliers"] = [int(self.outlier_combo_box.itemText(i)) for i in range(self.outlier_combo_box.count())]
        result["distance_metric"] = self.distance_metric_combo.currentText()
        
        
        self.main_ref.updateCluster(result)

    def inspect(self, event):
        self.main_ref.start_inspection()

    def delete(self, event):
        self.main_ref.delete_selection()
    
    def update(self, result, cell_list):
        self.all_cells = cell_list
        self.outlier_combo_box.clear()
        for idx,i in enumerate(self.events):
            if i in result:
                self.event_chkboxes[idx].setChecked(True)
                self.event_param[idx].duration_edit.setText(str(result[i]["window"]))
                self.event_param[idx].delay_edit.setText(str(result[i]["delay"]))
            else:
                self.event_chkboxes[idx].setChecked(False)
                self.event_param[idx].duration_edit.setText(str(result[i]["window"]))
                self.event_param[idx].delay_edit.setText(str(result[i]["delay"]))
        # if "ALP" in result:
        #     self.ALP_chkbox.setChecked(True)
        #     self.ALP_param.duration_edit.setText(str(result["ALP"]["window"]))
        #     self.ALP_param.delay_edit.setText(str(result["ALP"]["delay"]))
        # else:
        #     self.ALP_chkbox.setChecked(False)
        #     self.ALP_param.duration_edit.setText(str(self.event_defaults["ALP"]["window"]))
        #     self.ALP_param.delay_edit.setText(str(self.event_defaults["ALP"]["delay"]))
        # if "IALP" in result:
        #     self.IALP_chkbox.setChecked(True)
        #     self.IALP_param.duration_edit.setText(str(result["IALP"]["window"]))
        #     self.IALP_param.delay_edit.setText(str(result["IALP"]["delay"]))
        # else:
        #     self.IALP_chkbox.setChecked(False)
        #     self.IALP_param.duration_edit.setText(str(self.event_defaults["IALP"]["window"]))
        #     self.IALP_param.delay_edit.setText(str(self.event_defaults["IALP"]["delay"]))
        # if "RNFS" in result:
        #     self.RNFS_chkbox.setChecked(True)
        #     self.RNFS_param.duration_edit.setText(str(result["RNFS"]["window"]))
        #     self.RNFS_param.delay_edit.setText(str(result["RNFS"]["delay"]))
        # else:
        #     self.RNFS_chkbox.setChecked(False)
        #     self.RNFS_param.duration_edit.setText(str(self.event_defaults["RNFS"]["window"]))
        #     self.RNFS_param.delay_edit.setText(str(self.event_defaults["RNFS"]["delay"]))
        # if "ALP_Timeout" in result:
        #     self.ALP_Timeout_chkbox.setChecked(True)
        #     self.ALP_Timeout_param.duration_edit.setText(str(result["ALP_Timeout"]["window"]))
        #     self.ALP_Timeout_param.delay_edit.setText(str(result["ALP_Timeout"]["delay"]))
        # else:
        #     self.ALP_Timeout_chkbox.setChecked(False)
        #     self.ALP_Timeout_param.duration_edit.setText(str(self.event_defaults["ALP_Timeout"]["window"]))
        #     self.ALP_Timeout_param.delay_edit.setText(str(self.event_defaults["ALP_Timeout"]["delay"]))
        if "outliers" in result:
            for outlier in result["outliers"]:
                self.outlier_combo_box.addItem(str(outlier))
        
        self.cluster_select.setCurrentIndex(result["no_of_clusters"] - 2)


class GAToolWidget(QWidget):
    def __init__(self, main_ref, parent=None):
        super().__init__(parent)
        self.value_feature_dict = {"signal":["C","S","C_filtered","DFF","E"],"AUC":["C","S","C_filtered","DFF","E"],"Frequency":["S","DFF","E"]}
        # Max Generations
        self.main_ref = main_ref
        label_max_gen = QLabel("Max Generations:")
        self.input_max_gen = QLineEdit("50")
        onlyInt = QIntValidator()
        self.input_max_gen.setValidator(onlyInt)
        layout_max_gen = QHBoxLayout()
        layout_max_gen.addWidget(label_max_gen)
        layout_max_gen.addWidget(self.input_max_gen)

        #Population 
        label_population = QLabel("Population:")
        self.input_population = QLineEdit("20")
        self.input_population.setValidator(onlyInt)
        layout_population = QHBoxLayout()
        layout_population.addWidget(label_population)
        layout_population.addWidget(self.input_population)

        # Crossover Rate
        label_cross_rate = QLabel("Crossover Rate:")
        self.input_cross_rate = QLineEdit("0.5")
        onlyFloat = QDoubleValidator()
        self.input_cross_rate.setValidator(onlyFloat)
        layout_cross_rate = QHBoxLayout()
        layout_cross_rate.addWidget(label_cross_rate)
        layout_cross_rate.addWidget(self.input_cross_rate)

        # Mutation Rate
        label_mut_rate = QLabel("Mutation Rate:")
        self.input_mut_rate = QLineEdit("0.15")
        self.input_mut_rate.setValidator(onlyFloat)
        layout_mut_rate = QHBoxLayout()
        layout_mut_rate.addWidget(label_mut_rate)
        layout_mut_rate.addWidget(self.input_mut_rate)

        # Event Type
        event_list = ["ALP", "ILP", "RNF", "ALP_Timeout"]
        label_event_type = QLabel("Event Type:")
        self.dropdown_event_type = QComboBox()
        self.dropdown_event_type.addItems(event_list)
        layout_event_type = QHBoxLayout()
        layout_event_type.addWidget(label_event_type)
        layout_event_type.addWidget(self.dropdown_event_type)

        # Feature Vector
        label_feature_type = QLabel("Feature Type:")
        self.dropdown_feature_type = QComboBox()
        self.dropdown_feature_type.addItems(self.value_feature_dict.keys())
        layout_feature_type = QHBoxLayout()
        layout_feature_type.addWidget(label_feature_type)
        layout_feature_type.addWidget(self.dropdown_feature_type)

        # Value Type
        label_value_type = QLabel("Value Type:")
        self.dropdown_value_type = QComboBox()
        self.dropdown_value_type.addItems(self.value_feature_dict[self.dropdown_feature_type.currentText()])
        layout_value_type = QHBoxLayout()
        layout_value_type.addWidget(label_value_type)
        layout_value_type.addWidget(self.dropdown_value_type)

        self.dropdown_feature_type.currentIndexChanged.connect(lambda: self.update_feature(self.dropdown_feature_type.currentText()))

        # Log File
        label_file_name = QLabel("Log File:")
        self.input_log_file_name = QLineEdit("log.txt")
        layout_log_file = QHBoxLayout()
        layout_log_file.addWidget(label_file_name)
        layout_log_file.addWidget(self.input_log_file_name)
    

        btn_start = QPushButton("Start Genetic Algorithm")
        btn_start.clicked.connect(self.run_ga)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.setDirection(3)
        layout.addWidget(btn_start)
        layout.addLayout(layout_log_file)
        layout.addLayout(layout_value_type)
        layout.addLayout(layout_feature_type)
        layout.addLayout(layout_event_type)
        layout.addLayout(layout_mut_rate)
        layout.addLayout(layout_cross_rate)
        layout.addLayout(layout_population)
        layout.addLayout(layout_max_gen)
        self.setLayout(layout)
    
    def update_feature(self,key):
        self.dropdown_value_type.clear()
        self.dropdown_value_type.addItems(self.value_feature_dict[key])

    def run_ga(self):
        print("Started GA")
        max_gen = int(self.input_max_gen.text())
        population = int(self.input_population.text())
        cross_rate = float(self.input_cross_rate.text())
        mut_rate = float(self.input_mut_rate.text())
        event_type = self.dropdown_event_type.currentText()
        value_type = self.dropdown_value_type.currentText()
        feature_type = self.dropdown_feature_type.currentText()    
        print("-------GA mice amount: {}--------".format(len(self.main_ref.instances_list)))
        print("mice id: {}".format([x.mouseID for x in self.main_ref.instances_list]))
        ga = Genetic_Algorithm(self.main_ref.instances_list, max_gen, population, cross_rate, mut_rate, event_type,value_type, feature_type)
        ga.addLog(self.input_log_file_name.text())
        ga.execute()
        self.main_ref.start_ga(ga)


def hide_unhide(chkbox, param):
    if chkbox.isChecked():
        param.setEnabled(True)
    else:
        param.setEnabled(False)

class ExplorationToolWidget(QWidget):
    def __init__(self, main_ref, parent=None):
        super().__init__(parent)
        self.main_ref = main_ref

        button_explore = QPushButton("Data Exploration")
        button_explore.setStyleSheet("background-color : blue")
        button_explore.clicked.connect(self.explore)

        button_delete = QPushButton("Delete Cluster")
        button_delete.setStyleSheet("background-color : red")
        button_delete.clicked.connect(self.delete)

        layout = QVBoxLayout()
        layout.addWidget(button_explore)
        layout.addWidget(button_delete)
        layout.addStretch()
        self.setLayout(layout)

    def explore(self, event):
        self.main_ref.start_exploration()

    def delete(self, event):
        self.main_ref.delete_selection()

class SDAToolWidget(QWidget):
    def __init__(self, main_ref, parent=None):
        super().__init__(parent)
        self.main_ref = main_ref

        button_sda = QPushButton("Start Spatial Distribution Analysis")
        button_sda.setStyleSheet("background-color : green")
        button_sda.clicked.connect(self.sda)

        button_delete = QPushButton("Delete Cluster")
        button_delete.setStyleSheet("background-color : red")
        button_delete.clicked.connect(self.delete)

        layout = QVBoxLayout()
        layout.addWidget(button_sda)
        layout.addWidget(button_delete)
        layout.addStretch()
        self.setLayout(layout)

    def sda(self, event):
        self.main_ref.start_sda()

    def delete(self, event):
        self.main_ref.delete_selection()

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


class VisualizeInstanceWidget(QWidget):
    def __init__(self, main_ref: QMainWindow, parent=None):
        super().__init__(parent)
        self.main_ref = main_ref
        self.layout = QVBoxLayout()
        self.groups = {}

        self.setLayout(self.layout)

    def remove_visualization(self, group, mouse, session, day):
        self.groups[group].remove_visualization(mouse, session, day)
        if self.groups[group].is_empty():
            self.layout.removeWidget(self.groups[group])
            self.groups[group].setParent(None)
            del self.groups[group]
    
    def add_visualization(self, group, mouseID, image, session, day):
        if group not in self.groups:
            self.groups[group] = GroupGridLayout(self.main_ref, group)
            self.layout.addWidget(self.groups[group])
        
        self.groups[group].add_visualization(mouseID, image, session, day)


class GroupGridLayout(QWidget):
    def __init__(self, main_ref, group:str, parent=None):
        super().__init__(parent)
        self.main_ref = main_ref
        self.group = group
        self.layout = QHBoxLayout()

        self.mice = {}
        
        self.setLayout(self.layout)

    def add_visualization(self, mouseID, image, session, day):
        if mouseID not in self.mice:
            self.mice[mouseID] = MouseGrid(self.main_ref, mouseID, self.group)
            self.layout.addWidget(self.mice[mouseID])
        self.mice[mouseID].add_visualization(image, session, day)
    
    def remove_visualization(self, mouseID, session, day):
        self.mice[mouseID].remove_visualization(session, day)
        if self.mice[mouseID].is_empty():
            self.layout.removeWidget(self.mice[mouseID])
            self.mice[mouseID].setParent(None)
            del self.mice[mouseID]

    
    def is_empty(self):
        return self.layout.count() == 0

class MouseGrid(QWidget):
    def __init__(self, main_ref, mouseID:str, group:str, parent=None):
        super().__init__(parent)
        self.main_ref = main_ref
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
        
        # Grid Frame
        self.grid_frame = QFrame()
        self.grid_frame.setFrameShape(QFrame.Box)
        self.grid_frame.setFrameShadow(QFrame.Raised)
        self.grid_frame.setLineWidth(3)

        self.add_frame()


    def remove_frame(self):
        """
        Due to a quirk of PyQT5 in  qgridlayout the frame overlaps the widgets, which to the best  of my knowledge
        doesn't allow us to access the widgets that are "beneath" the frame. To address this we temporarily remove the frame
        when we need to access the widgets and add it afterwards.
        """
        self.layout.removeWidget(self.grid_frame)

    def add_frame(self):
        self.layout.addWidget(self.grid_frame, 0, 0, -1, -1)
        
    
    def add_visualization(self, image, session, day):
        # There is a chance that the visualization already exists. If so, just update the visualization
        if f"{session[1:]}:{day[1:]}" in self.images:
            self.images[f"{session[1:]}:{day[1:]}"].update_visualization(image)
            return
        image_viewer = Viewer(self.main_ref, image, self.group, self.mouseID, session, day)
        aspect_maintain = AspectRatioWidget(image_viewer, aspect_ratio=1.0)
        day = int(day[1:])
        session = int(session[1:])
        self.images[f"{session}:{day}"] = aspect_maintain

        if day not in self.days:
            bisect.insort(self.days, day)
        if session not in self.sessions:
            bisect.insort(self.sessions, session)
        if day not in self.day_labels:
            self.day_labels[day] = GridQLabel("D" + str(day))
        if session not in self.session_labels:
            self.session_labels[session] = GridQLabel("S" + str(session))
        self.redraw_grid()
        
    def remove_visualization(self, session, day):
        # Temporarily remove the frame
        self.remove_frame()
        day = int(day[1:])
        session = int(session[1:])
        session_index = self.sessions.index(session)
        day_index = self.days.index(day)
        wid = self.layout.itemAtPosition(session_index+1, day_index+1).widget()
        self.layout.removeWidget(wid)
        wid.setParent(None)
        wid.deleteLater()
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

        self.redraw_grid()

        if self.layout.count() == 1:
            wid = self.layout.itemAtPosition(0, 0).widget()
            self.layout.removeWidget(wid)
            wid.setParent(None)
            wid.deleteLater()
        
        else:
            self.add_frame()

    def is_empty(self):
        return self.layout.count()


            
    def redraw_grid(self):
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
    
    def is_empty(self):
        return self.layout.count() == 0
        



class Viewer(QGraphicsView):
    def __init__(self, main_ref, image, group, mouseID, session, day, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.m_pixmapItem = self.scene().addPixmap(QPixmap())
        self.setAlignment(Qt.AlignCenter)
        self.main_ref = main_ref

        self.p = self.palette()
        self.p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(self.p)
        
        self.selected = False
        self.mouseReleaseEvent=self.update_params
        self.mouseDoubleClickEvent=self.inspect
        self.group = group
        self.mouseID = mouseID
        self.session = session
        self.day = day
        self.image = image
        self.create_pixmap()

    def update_visualization(self, image):
        self.image = image
        self.create_pixmap()

    def create_pixmap(self):
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

    def change_to_white(self):
        self.p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(self.p)
        self.selected = False

    def change_to_red(self):
        self.p.setColor(self.backgroundRole(), Qt.red)
        self.setPalette(self.p)
        self.selected = True

    def update_params(self, _):
        self.main_ref.activate_params(self)
        
    def update_visualization(self, image):
        ov = (image*255).astype('uint8')
        qimg = QImage(ov, ov.shape[1], ov.shape[0], ov.shape[1] * 3, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)

    def inspect(self, _):
        self.main_ref.start_inspection(self)

    def __eq__(self, other):
        return (self.group, self.session, self.day, self.mouseID) == (other.group, other.session, other.day, other.mouseID)

    def return_info(self):
        return self.group, self.session, self.day, self.mouseID

class AspectRatioWidget(QWidget):
    def __init__(self, widget, parent=None, aspect_ratio=1.0):
        super().__init__(parent)
        self.aspect_ratio = aspect_ratio
        self.widget = widget
        self.setLayout(QBoxLayout(QBoxLayout.LeftToRight, self))
        self.layout().addItem(QSpacerItem(0, 0))
        self.layout().addWidget(self.widget)
        self.layout().addItem(QSpacerItem(0, 0))

    def update_visualization(self, image):
        self.widget.update_visualization(image)

    def resizeEvent(self, e):
        w = e.size().width()
        h = e.size().height()

        if w / h > self.aspect_ratio:
            self.layout().setDirection(QBoxLayout.LeftToRight)
            widget_stretch = h * self.aspect_ratio
            outer_stretch = (w - widget_stretch) / 2 + 0.5
        else:
            self.layout().setDirection(QBoxLayout.TopToBottom)
            widget_stretch = w / self.aspect_ratio
            outer_stretch = (h - widget_stretch) / 2 + 0.5

        self.layout().setStretch(0, int(outer_stretch))
        self.layout().setStretch(1, int(widget_stretch))
        self.layout().setStretch(2, int(outer_stretch)) 

    def reset(self):
        self.widget.reset()
class GridQLabel(QLabel):
    def __init__(self, parent=None, *args):
        super().__init__(parent, *args)

        self.setAlignment(Qt.AlignCenter)
        self.setContentsMargins(8, 8, 0, 0)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setBaseSize(300,300)
        self.setStyleSheet("font-size: 14pt;")
        self.setLineWidth(2)