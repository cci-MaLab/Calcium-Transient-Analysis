from PyQt5.QtWidgets import (QApplication, QMainWindow, QStyle, QFileDialog, QMessageBox, QAction,
                            QVBoxLayout, QHBoxLayout, QWidget, QTabWidget)
from gui.main_widgets import (UpdateDialog, ParamDialog, VisualizeInstanceWidget, Viewer, ClusteringToolWidget,
                            GAToolWidget, ExplorationToolWidget)

from gui.genetic_algorithm_widgets import GAWindowWidget
from gui.exploration_widgets import ExplorationWidget
import sys
import os
import json
sys.path.insert(0, ".")
from core.backend import DataInstance
from gui.clustering_inspection_widgets import InspectionWidget
import dask

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Cell Exploration Tool")
        self.setMinimumSize(600, 800)
        
        self.windows = {}

        dask.config.set({"array.slicing.split_large_chunks": True})

        # Data stuff
        self.instances = {}
        self.path_list = {}

        # Event defaults:
        self.event_defaults = {"ALP": {"window": 20, "delay": 0},
                               "IALP": {"window": 20, "delay": 0},
                               "RNFS": {"window": 20, "delay": 0},
                               "ALP_Timeout": {"window": 20, "delay": 0},
                               "distance_metric": "euclidean"}

        # Menu Bar
        pixmapi_folder = QStyle.StandardPixmap.SP_DirIcon
        button_folder = QAction(self.style().standardIcon(pixmapi_folder), "&Load Data", self)
        button_folder.setStatusTip("Select a Folder to load in data")
        button_folder.triggered.connect(self.load_data)
        pixmapi_save = QStyle.StandardPixmap.SP_DialogSaveButton
        button_save = QAction(self.style().standardIcon(pixmapi_save), "&Save", self)
        button_save.setStatusTip("Save current state")
        button_save.triggered.connect(self.save)
        pixmapi_load = QStyle.StandardPixmap.SP_FileDialogStart
        button_load = QAction(self.style().standardIcon(pixmapi_load), "&Load Saved State", self)
        button_load.setStatusTip("Load previously saved state")
        button_load.triggered.connect(self.load_saved_state)
        pixmapi_update = QStyle.StandardPixmap.SP_BrowserReload
        button_update = QAction(self.style().standardIcon(pixmapi_update), "&Update Default Parameters", self)
        button_update.setStatusTip("Update the default parameters of the events")
        button_update.triggered.connect(self.update_defaults)
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_folder)
        file_menu.addAction(button_save)
        file_menu.addAction(button_load)
        file_menu.addAction(button_update)

        # Tool Widgets        
        self.current_selection = None
        self.cl_tools = ClusteringToolWidget(self, self.event_defaults)
        self.ga_tools = GAToolWidget(self)
        self.e_tools = ExplorationToolWidget(self)
        self.cl_tools.setEnabled(False)
        self.e_tools.setEnabled(False)
        self.ga_tools.setEnabled(False)

        # Layouts and tabs
        layout_central = QHBoxLayout()
        layout_cluster = QVBoxLayout()
        tabs = QTabWidget()
        tabs.setFixedWidth(320)
        self.instance_viz = VisualizeInstanceWidget(self)

        tabs.addTab(self.e_tools, "Exploration")
        tabs.addTab(self.cl_tools, "Clustering")
        tabs.addTab(self.ga_tools, "Genetic Algorithm")

        layout_cluster.addWidget(self.instance_viz)
        layout_central.addLayout(layout_cluster)
        layout_central.addWidget(tabs)

        widget = QWidget()
        widget.setLayout(layout_central)
        self.setCentralWidget(widget)

        self.show()

    def activate_params(self, viewer: Viewer):
        if self.current_selection is None:
            self.current_selection = viewer
            self.current_selection.change_to_red()
            self.cl_tools.setEnabled(True)
            self.e_tools.setEnabled(True)
            self.ga_tools.setEnabled(True)
            self.update_params()
        elif self.current_selection == viewer:
            if viewer.selected == True:
                viewer.change_to_white()
                self.cl_tools.setEnabled(False)
                self.e_tools.setEnabled(False)
                self.ga_tools.setEnabled(False)
            else:
                viewer.selected = True
                viewer.change_to_red()
                self.cl_tools.setEnabled(True)
                self.e_tools.setEnabled(True)
                self.ga_tools.setEnabled(True)
                self.update_params()
        else:
            self.current_selection.change_to_white()

            self.current_selection = viewer
            self.current_selection.change_to_red()
            self.cl_tools.setEnabled(True)
            self.e_tools.setEnabled(True)
            self.ga_tools.setEnabled(True)
            self.update_params()

    def update_params(self):
        group, session, day, mouseID = self.current_selection.return_info()
        instance = self.instances[group][mouseID][f"{session}:{day}"]
        result = self.path_list[instance.dpath]
        if result is not None:
            self.cl_tools.display_params()
            result["no_of_clusters"] = instance.no_of_clusters
            self.cl_tools.update(result, instance.data["unit_ids"])
        else:
            self.cl_tools.display_default()

    def update_defaults(self):
        pdg = UpdateDialog(self.event_defaults)
        if pdg.exec():
            result = pdg.get_result()
        else:
            return
        self.event_defaults = result
        self.cl_tools.update_defaults(self.event_defaults)

    def start_exploration(self, current_selection=None):
        current_selection = self.current_selection if current_selection is None else current_selection
        group, session, day, mouseID = current_selection.return_info()
        instance = self.instances[group][mouseID][f"{session}:{day}"]

        name = f"{instance.mouseID} {instance.day} {instance.session} Exploration"

        if name not in self.windows:
            wid = ExplorationWidget(instance, name, self)
            if wid is not None:
                wid.setWindowTitle(name)
                self.windows[name] = wid
                wid.show()

    def start_ga(self, ga):
        name = "Genetic Algorithm"
        if name not in self.windows:
            wid = GAWindowWidget(self,ga = ga)
            wid.setWindowTitle(name)
            self.windows[name] = wid
            wid.show()


    def update_cluster(self, result):
        no_of_clusters = result.pop("no_of_clusters")
        outliers = result.pop("outliers")
        distance_metric = result.pop("distance_metric")
        self.setWindowTitle("Loading...")
        group, session, day, mouseID = self.current_selection.return_info()
        instance = self.instances[group][mouseID][f"{session}:{day}"]
        instance.set_outliers(outliers)
        instance.set_distance_metric(distance_metric)
        instance.load_events(result.keys())
        for event in result:
            delay, window = result[event]["delay"], result[event]["window"]
            instance.events[event].set_delay_and_duration(delay, window)
            instance.events[event].set_values()     
        result["outliers"] = instance.outliers_list
        result["distance_metric"] = instance.distance_metric
        instance.set_vector()
        instance.set_no_of_clusters(no_of_clusters)
        instance.compute_clustering()
        self.path_list[instance.dpath] = result

        # Visualisation stuff
        mouseID, session, day, group, cl_result = instance.get_vis_info()
        self.current_selection.update_visualization(cl_result)
        self.setWindowTitle("Cell Exploration Tool")

        # Check if there is an active subwindow and update it
        name = f"{instance.mouseID} {instance.day} {instance.session}"
        if name in self.windows:
            self.windows[name].refresh()

        
    def start_inspection(self, current_selection=None):
        current_selection = self.current_selection if current_selection is None else current_selection
        group, session, day, mouseID = current_selection.return_info()
        instance = self.instances[group][mouseID][f"{session}:{day}"]

        name = f"{instance.mouseID} {instance.day} {instance.session} Inspection"

        # Clustering result will only contain basic if it wasn't computed
        if name not in self.windows and "all" in instance.clustering_result:
            wid = InspectionWidget(instance, self)
            wid.setWindowTitle(name)
            self.windows[name] = wid
            wid.show()
    
    def delete_selection(self):
        group, session, day, mouseID = self.current_selection.return_info()
        self.instance_viz.remove_visualization(group, mouseID, session, day)
        path = self.instances[group][mouseID][f"{session}:{day}"].dpath
        # Remove it from all references
        del self.path_list[path]
        del self.instances[group][mouseID][f"{session}:{day}"]
        if not self.instances[group][mouseID]:
            del self.instances[group][mouseID]
        self.cl_tools.setEnabled(False)
        self.e_tools.setEnabled(False)
        self.ga_tools.setEnabled(False)
        self.current_selection = None



    def remove_window(self, name):
        del self.windows[name]

    def print_error(self, s):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("I have a question!")
        if isinstance(s, tuple):
            text = f"For path {s[1]} the following error occurred:\n {s[0]}"
        else:
            text = s
        dlg.setText(text)
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.exec()


    def load_data(self, _):
        fname = QFileDialog.getOpenFileName(
            self,
            "Select ini File",
        )
        fname = fname[0]
        if fname != '' and fname not in self.path_list:
            self.load_instance(fname)

    def load_clustering_params(self):
        pdg = ParamDialog(self.event_defaults)
        if pdg.exec():
            result = pdg.get_result()
        else:
            return    
        self.load_clustering(result)        

    def load_saved_state(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Open File",
        )
        fname = fname[0]
        if fname[-4:] == "json":
            if os.path.getsize(fname) != 0:
                self.setWindowTitle("Loading...")
                with open(fname, 'r') as f:
                    self.path_list = json.load(f)

                if "defaults" in self.path_list:
                    self.event_defaults = self.path_list.pop("defaults")
                    if "distance_metric" not in self.event_defaults:
                        self.event_defaults["distance_metric"] = "euclidean"
                    self.cl_tools.update_defaults(self.event_defaults)

                for path in self.path_list.keys():
                    results = self.path_list[path]
                    self.load_instance(path, results)

                self.setWindowTitle("Cell Exploration Tool")      
        

    def load_clustering(self, result):
        self.setWindowTitle("Loading...")

        current_selection = self.current_selection
        group, session, day, mouseID = current_selection.return_info()
        instance = self.instances[group][mouseID][f"{session}:{day}"]

        events = list(result.keys())
        
        if "distance_metric" in result:
            events.remove("distance_metric")
        if "outliers" in result:
            events.remove("outliers")
        no_of_clusters = None
        if "no_of_clusters" in events:
            no_of_clusters = result["no_of_clusters"]
            events.remove("no_of_clusters")
        instance.load_events(events)
        if no_of_clusters is not None:
            instance.no_of_clusters = no_of_clusters
        for event in events:
            delay, window = result[event]["delay"], result[event]["window"]
            instance.events[event].set_delay_and_duration(delay, window)
            instance.events[event].set_values()
        if "outliers" in result:
            instance.set_outliers(result["outliers"])
        if "distance_metric" in result:
            instance.set_distance_metric(result["distance_metric"])
        instance.set_vector()
        instance.compute_clustering()

        self.path_list[instance.dpath] = result
        

        # Visualisation stuff
        mouseID, session, day, group, cl_result = instance.get_vis_info()
        # Generate the Grid
        self.instance_viz.add_visualization(group, mouseID, cl_result, session, day)
        self.cl_tools.display_params()
        self.current_selection.change_to_red()
        self.update_params()
        self.setWindowTitle("Cell Exploration Tool")

    def load_instance(self, fname):
        self.setWindowTitle("Loading...")
        instance = DataInstance(fname)
        # Check if the instance is valid
        if instance.check_essential_data():
            mouseID, session, day, group, viz_result = instance.get_vis_info()
            if group not in self.instances:
                self.instances[group] = {}
            if instance.mouseID not in self.instances[group]:
                self.instances[group][f"{instance.mouseID}"] = {}
            self.instances[group][f"{instance.mouseID}"][f"{session}:{day}"] = instance
            self.instance_viz.add_visualization(group, mouseID, viz_result, session, day)
            self.path_list[fname] = None
            # Initially there might be no E and DFF data, check and create it if necessary
            instance.check_E()
            instance.check_DFF()
        self.setWindowTitle("Cell Exploration Tool")
    
    def save(self):
        default_dir = os.getcwd()
        default_filename = os.path.join(default_dir, "paths.json")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save State", default_filename, "JSON Files (*.json)"
        )
        if filename:
            if self.path_list:
                extended_json = self.path_list.copy()
                extended_json["defaults"] = self.event_defaults
                with open(filename, 'w') as f:
                    json.dump(extended_json, f)
    

app = QApplication([])
app.setStyle('Fusion')
window = MainWindow()
app.exec()