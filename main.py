from PyQt5.QtWidgets import (QApplication, QMainWindow, QStyle, QFileDialog, QMessageBox, QAction,
                            QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QWidget,
                            QFrame, QCheckBox)
from PyQt5.QtCore import (QThreadPool)
from custom_widgets import LoadingDialog, ParamDialog, VisualizeClusterWidget, Viewer, ToolWidget
from PyQt5 import Qt
import sys
import os
import json
sys.path.insert(0, ".")
from backend import SessionFeature

class MainWindow(QMainWindow):


    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Cell Clustering Tool")
        self.setMinimumSize(600, 400)

        # Threading
        self.threadpool = QThreadPool()

        # Data stuff
        self.sessions = {}
        self.sessions['saline'] = {}
        self.sessions['cocaine'] = {}
        self.path_list = {}

        # Menu Bar
        pixmapi_folder = QStyle.StandardPixmap.SP_DirIcon
        button_folder = QAction(self.style().standardIcon(pixmapi_folder), "&Load Data", self)
        button_folder.setStatusTip("Select a Folder to load in data")
        button_folder.triggered.connect(self.onMyToolBarButtonClick)
        pixmapi_save = QStyle.StandardPixmap.SP_DialogSaveButton
        button_save = QAction(self.style().standardIcon(pixmapi_save), "&Save", self)
        button_save.setStatusTip("Save current state")
        button_save.triggered.connect(self.save)
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_folder)
        file_menu.addAction(button_save)

        # Tool Widgets

        
        self.current_selection = None
        self.w_tools = ToolWidget()
        self.w_tools.setEnabled(False)


        # Layouts
        layout_central = QHBoxLayout()
        layout_cluster = QVBoxLayout()
        self.cluster_viz = VisualizeClusterWidget()


        layout_cluster.addWidget(self.cluster_viz)
        layout_central.addLayout(layout_cluster)
        layout_central.addWidget(self.w_tools)

        widget = QWidget()
        widget.setLayout(layout_central)
        self.setCentralWidget(widget)




        self.show()

        if os.path.isfile('paths.json'):
            if os.path.getsize('paths.json') != 0:
                dlg = LoadingDialog()
                if dlg.exec():
                    self.setWindowTitle("Loading...")
                    with open('data.json', 'w') as f:
                        json.dump(self.path_list, f)
                    
                    for fname, results in self.path_list.values():
                        self.load_session(fname, results)

                    self.setWindowTitle("Cell Clustering Tool")

    def activateParams(self, viewer: Viewer):
        if self.current_selection is None:
            self.current_selection = viewer
            self.current_selection.changeToRed()
            self.w_tools.setEnabled(True)
            self.updateParams()
        elif self.current_selection == viewer:
            if viewer.selected == True:
                viewer.changeToWhite()
                self.w_tools.setEnabled(False)
            else:
                viewer.selected = True
                viewer.changeToRed()
                self.w_tools.setEnabled(True)
                self.updateParams()
        else:
            self.current_selection.changeToWhite()

            self.current_selection = viewer
            self.current_selection.changeToRed()
            self.w_tools.setEnabled(True)
            self.updateParams()

    def updateParams(self):
        group, x, y, mouseID = self.current_selection.returnInfo()
        session = self.sessions[group][mouseID][f"{x}:{y}"]
        result = self.path_list[session.dpath]

        self.w_tools.update(result)

    def updateCluster(self, result):
        self.setWindowTitle("Loading...")
        group, x, y, mouseID = self.current_selection.returnInfo()
        session = self.sessions[group][mouseID][f"{x}:{y}"]
        old_result = self.path_list[session.dpath]
        session.load_events(result.keys())
        for event in result:
            delay, window = result[event]["delay"], result[event]["window"]
            session.events[event].set_delay_and_duration(delay, window)
            session.events[event].set_values()        
        result["group"] = old_result["group"]
        session.set_vector()
        session.compute_clustering()
        self.path_list[session.dpath] = result

        # Visualisation stuff
        mouseID, x, y, group, cl_result = session.get_vis_info()
        self.current_selection.updateVisualization(cl_result)
        self.setWindowTitle("Cell Clustering Tool")
        
    def startInspection(self, current_selection=None):
        current_selection = self.current_selection if current_selection is None else current_selection
        dlg = QMessageBox(self)
        dlg.setWindowTitle("I have a question!")
        dlg.setText("This is a question dialog")
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dlg.setIcon(QMessageBox.Question)
        button = dlg.exec()

        if button == QMessageBox.Yes:
            print("Yes!")
        else:
            print("No!")
        

    def printError(self, s):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("I have a question!")
        if isinstance(s, tuple):
            text = f"For path {s[1]} the following error occurred:\n {s[0]}"
        else:
            text = s
        dlg.setText(text)
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.exec()


    def addData(self):
        #self.data.append(data)
        print("finished")


    def onMyToolBarButtonClick(self, s):
        fname = QFileDialog.getExistingDirectory(
            self,
            "Open File",
        )

        pdg = ParamDialog()
        if pdg.exec():
            result = pdg.get_result()
        else:
            return



        # Cannot do the stuff below as it segfaults :(
        '''
        worker = Worker(open_minian, fname, self.data)
        self.threadpool.start(worker)
        worker.signals.error.connect(self.printError)
        worker.signals.finished.connect(self.addData)

        # Execute
        self.threadpool.start(worker)
        '''
        if fname != '' and fname not in self.path_list:
            self.load_session(fname, result)
            

            
        

    def load_session(self, fname, result):
        self.setWindowTitle("Loading...")
        events = list(result.keys())
        events.remove("group")
        session = SessionFeature(fname, events)
        for event in events:
            delay, window = result[event]["delay"], result[event]["window"]
            session.events[event].set_delay_and_duration(delay, window)
            session.events[event].set_values()

        session.set_group(result["group"])
        session.set_vector()
        session.compute_clustering()

        self.path_list[fname] = result
        

        # Visualisation stuff
        mouseID, x, y, group, cl_result = session.get_vis_info()
        if session.mouseID not in self.sessions[group]:
            self.sessions[group][f"{session.mouseID}"] = {}
            self.cluster_viz.grids[group].addGrid(mouseID)
        self.sessions[group][f"{session.mouseID}"][f"{x}:{y}"] = session
        # Generate the Grid
        self.cluster_viz.grids[group].addVisualization(group, mouseID, cl_result, x, y)
        self.setWindowTitle("Cell Clustering Tool")

    
    def save(self):
        if self.path_list:
            with open('paths.json', 'w') as f:
                json.dump(self.path_list, f)

app = QApplication([])
window = MainWindow()
app.exec()