"""
The following file will be used for doing a deeper dive into the selected session
"""
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QAction, QStyle, 
                            QSlider, QLabel, QListWidget, QAbstractItemView, QLineEdit, QSplitter,
                            QApplication, QStyleFactory, QFrame, QTabWidget, QMenuBar, QCheckBox,
                            QTextEdit, QComboBox, QGraphicsTextItem, QMessageBox, QFileDialog,
                            QScrollArea, QListWidgetItem)
from PyQt5.QtCore import (Qt, QTimer)
from PyQt5 import QtCore
from PyQt5.QtGui import (QIntValidator, QDoubleValidator, QFont)
from pyqtgraph import (PlotItem, PlotCurveItem, ScatterPlotItem, InfiniteLine)
import pyqtgraph as pg
import numpy as np
from scipy.signal import find_peaks
from skimage.segmentation import flood_fill
from skimage.feature import canny
from skimage.measure import find_contours
from core.exploration_statistics import (GeneralStatsWidget, LocalStatsWidget, MetricsWidget)
from core.pyqtgraph_override import ImageViewOverride
from gui.cofiring_2d_widgets import Cofiring2DWidget
from gui.sda_widgets import (MayaviQWidget, base_visualization)
import os
import matplotlib.pyplot as plt
import pickle
from mayavi.core.lut_manager import lut_mode_list


try:
    import torch
    torch_imported = True
except ImportError:
    torch_imported = False

class ExplorationWidget(QWidget):
    def __init__(self, session, name, main_window_ref, timestamps=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.name = name
        self.main_window_ref = main_window_ref
        self.timestamps = timestamps
        self.gen_stats_window = None
        self.metrics_window = None
        self.local_stats_windows = {}
        self.select_missed_mode = False
        self.missed_cell_indices = set()
        self.missed_cells_selection = set()
        self.prev_video_tab_idx = 0
        self.show_justification = False
        self.rejected_justification = self.session.load_justifications()
        self.savgol_params = {}
        self.noise_params = {}
        self.hovered_cells = {} # id, item
        self.temp_picks = {}
        self.show_temp_picks = True
        self.pre_images = None
        self.pre_bimages = None
        self.windows = {}



        # Set up main view
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.imv_cell = ImageViewOverride()
        self.imv_behavior = ImageViewOverride()
        self.imv_behavior.setVisible(False)
        self.visualization_3D = MayaviQWidget(self.session)
        self.visualization_3D.setVisible(False)
        self.visualization_3D.point_signal.connect(self.point_selection)


        self.session.load_videos()
        if not self.session.video_data:
            print("Missing Videos")
            return None
        self.current_video = self.session.video_data["varr"]
        self.video_length = self.current_video.shape[0]
        self.mask = np.ones((self.current_video.shape[1], self.current_video.shape[2]))
        # We need two seperate masks here. One for the missed cells we confirmed and one for drawing a new missed cell
        self.video_missed_mask = np.zeros(self.mask.shape)
        self.video_missed_mask_candidate = np.zeros(self.mask.shape)
        self.current_frame = 0
        self.imv_cell.setImage(self.current_video.sel(frame=self.current_frame).values)
        if "behavior_video" in self.session.video_data:
            self.imv_behavior.setImage(self.session.video_data["behavior_video"].sel(frame=self.current_frame).values[0])

        # Add Context Menu Action
        self.video_to_title = {"varr": "Original", "Y_fm_chk": "Processed"}
        self.submenu_videos = self.imv_cell.getView().menu.addMenu('&Video Format')
        for type in self.session.video_data.keys():
            if type ==  "Y_hw_chk" or type == "behavior_video":
                continue
            button_video_type = QAction(f"&{self.video_to_title[type]}", self.submenu_videos)
            button_video_type.triggered.connect(lambda state, x=type: self.change_cell_video(x))
            button_video_type.setCheckable(True)
            if type == "varr":
                button_video_type.setChecked(True)
            else:
                button_video_type.setChecked(False)
            self.submenu_videos.addAction(button_video_type)
        
        submenu_add_group = self.imv_cell.getView().menu.addMenu('&Add Group')
        button_add_group_rect = QAction("Rectangle", submenu_add_group)
        submenu_add_group.addAction(button_add_group_rect)
        button_add_group_rect.triggered.connect(lambda: self.highlight_roi_selection(type="rectangle"))
        button_add_group_ellipse = QAction("Ellipse", submenu_add_group)
        submenu_add_group.addAction(button_add_group_ellipse)
        button_add_group_ellipse.triggered.connect(lambda: self.highlight_roi_selection(type="ellipse"))
        

        # Menu Bar for statistics
        menu = QMenuBar()
        pixmapi_tools = QStyle.StandardPixmap.SP_FileDialogListView
        btn_general_stats = QAction(self.style().standardIcon(pixmapi_tools), "&General Statistics", self)
        btn_general_stats.setStatusTip("Produce General Statistics")
        btn_general_stats.triggered.connect(self.generate_gen_stats)

        stats_menu = menu.addMenu("&Statistics")
        stats_menu.addAction(btn_general_stats)

        btn_max_projection_processed = QAction(self.style().standardIcon(pixmapi_tools), "&Max Projection", self)
        btn_max_projection_processed.setStatusTip("Save Max Projection")
        btn_max_projection_processed.triggered.connect(self.save_max_projection)

        self.chkbox_cell_video = QAction("Cell Video", self)
        self.chkbox_cell_video.setCheckable(True)
        self.chkbox_cell_video.setChecked(True)
        self.chkbox_cell_video.triggered.connect(self.toggle_videos)
        self.chkbox_behavior_video = QAction("Behavior Video", self)
        self.chkbox_behavior_video.setCheckable(True)
        self.chkbox_behavior_video.setChecked(False)
        self.chkbox_behavior_video.triggered.connect(self.toggle_videos)
        self.chkbox_3D = QAction("3D Visualization", self)
        self.chkbox_3D.setCheckable(True)
        self.chkbox_3D.setChecked(False)
        self.chkbox_3D.triggered.connect(self.toggle_videos)

        util_menu = menu.addMenu("&Utilities")
        util_menu.addAction(btn_max_projection_processed)

        video_menu = menu.addMenu("&Select Videos/Visualizations")
        video_menu.addAction(self.chkbox_cell_video)
        video_menu.addAction(self.chkbox_behavior_video)
        video_menu.addAction(self.chkbox_3D)

        

        # We'll load in a copy of the visualization of the cells we are monitoring
        self.A = self.session.A.copy()
        for outlier in self.session.outliers_list:
            self.A.pop(outlier)


        self.A_pos_to_missed_cell = {}
        self.A_pos_to_cell = {}
        for cell_id, cell_ROI in self.A.items():
            indices = np.argwhere(cell_ROI.values > 0)
            for pair in indices:
                if tuple(pair) in self.A_pos_to_cell:
                    self.A_pos_to_cell[tuple(pair)].append(cell_id)
                else:
                    self.A_pos_to_cell[tuple(pair)] = [cell_id]
                    


        # Select Cells
        w_cell_label = QLabel("Pick which cells to focus (Hold ctrl/shift for multi):")
        self.list_cell = QListWidget()
        self.list_cell.setMaximumHeight(600)
        self.list_cell.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.btn_cell_focus = QPushButton("Focus Selection")
        self.btn_cell_focus.clicked.connect(self.focus_mask)
        self.btn_cell_reset = QPushButton("Reset Mask")
        self.btn_cell_reset.clicked.connect(self.reset_mask)
        cell_highlight_mode_label = QLabel("Highlight Mode:")
        self.cmb_cell_highlight_mode = QComboBox()
        self.cmb_cell_highlight_mode.addItems(["Outline", "Color", "Clear"])
        self.cmb_cell_highlight_mode.setCurrentIndex(0)
        self.cmb_cell_highlight_mode.currentIndexChanged.connect(self.refresh_image)
        self.btn_add_to_group = QPushButton("Add to Group")
        self.btn_add_to_group.clicked.connect(self.add_to_group)
        self.btn_remove_from_group = QPushButton("Remove from Group")
        self.btn_remove_from_group.clicked.connect(self.remove_from_group)
        self.btn_cell_reject = QPushButton("Reject Cell(s)")
        self.btn_cell_reject.clicked.connect(self.reject_cells)

        self.tabs_video_tools_parent = QScrollArea()
        self.tabs_video_tools = QTabWidget()
        self.tabs_video_tools_parent.setFixedWidth(340)
        self.tabs_video_tools_parent.setWidgetResizable(True)
        self.tabs_video_tools_parent.setWidget(self.tabs_video_tools)

        self.tabs_visualization = QTabWidget()

        self.tabs_video = QTabWidget()
        self.tabs_video.setFixedWidth(330)

        tabs_signal_parent = QScrollArea()
        tabs_signal = QTabWidget()
        tabs_signal_parent.setWidgetResizable(True)
        tabs_signal_parent.setWidget(tabs_signal)
        tabs_signal_parent.setFixedWidth(350)
        self.tabs_global_cell_switch = QTabWidget() # This is for the bottom half of the screen

        tabs_cofiring_options = QTabWidget()


        # Rejected Cells
        w_rejected_cell_label = QLabel("Rejected Cells:")
        self.list_rejected_cell = QListWidget()
        self.list_rejected_cell.setMaximumHeight(600)
        self.list_rejected_cell.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_rejected_cell.itemSelectionChanged.connect(lambda: self.enable_disable_justification(True))
        
        self.btn_cell_return = QPushButton("Return Cell")
        self.btn_cell_return.clicked.connect(self.approve_cells)
        self.btn_justification_start = QPushButton("Show/Justify Rejection")
        self.btn_justification_start.setEnabled(False)
        self.btn_justification_start.clicked.connect(self.start_justification)
        self.btn_justification_save = QPushButton("Save")
        self.btn_justification_save.clicked.connect(self.backup_text)
        self.btn_justification_save.hide()
        self.btn_justification_cancel = QPushButton("Cancel")
        self.btn_justification_cancel.clicked.connect(lambda: self.enable_disable_justification(False))
        self.btn_justification_cancel.hide()

        self.input_justification = QTextEdit()
        self.input_justification.hide()

        # Missed Cells
        w_missed_cell_label = QLabel("Missed Cells:")
        self.list_missed_cell = QListWidget()
        self.list_missed_cell.setMaximumHeight(600)
        self.list_missed_cell.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.btn_missed_select = QPushButton("Enable Select Cell Mode")
        self.btn_missed_select.clicked.connect(self.switch_missed_cell_mode)
        self.btn_missed_remove = QPushButton("Remove Cell(s)")
        self.btn_missed_remove.clicked.connect(self.remove_missed_cells)

        btn_missed_clear = QPushButton("Clear Selected Pixels")
        btn_missed_clear.setStyleSheet("background-color: yellow")
        btn_missed_clear.clicked.connect(self.clear_selected_pixels)
        btn_missed_confirm = QPushButton("Confirm Selected Pixels")
        btn_missed_confirm.setStyleSheet("background-color: green")
        btn_missed_confirm.clicked.connect(self.confirm_selected_pixels)
        layout_missed_utility = QHBoxLayout()
        layout_missed_utility.addWidget(btn_missed_clear)
        layout_missed_utility.addWidget(btn_missed_confirm)
        self.w_missed_utility = QWidget()
        self.w_missed_utility.setLayout(layout_missed_utility)
        self.w_missed_utility.hide()

        btn_missed_reset_mask = QPushButton("Reset Mask")
        btn_missed_reset_mask.clicked.connect(self.reset_mask)
        btn_missed_focus_mask = QPushButton("Focus Selection")
        btn_missed_focus_mask.clicked.connect(self.focus_mask)

        # Plot utility
        self.auto_label = QLabel("Automatic Transient Detection")
        self.manual_label = QLabel("Manual Transient Detection")
        self.min_height_label = QLabel("Peak Threshold (ΔF/F)")
        local_stats_label = QLabel("Local Statistics")
        self.min_height_input = QLineEdit()
        self.min_height_input.setValidator(QDoubleValidator(0, 1000, 3))
        self.min_height_input.setText("0")

        self.dist_label = QLabel("Interval Threshold (frame)")
        self.dist_input = QLineEdit()
        self.dist_input.setValidator(QIntValidator(0, 1000))
        self.dist_input.setText("10")

        self.snr_label = QLabel("SNR Threshold")
        self.snr_input = QLineEdit()
        self.snr_input.setValidator(QDoubleValidator(0, 1000, 3))
        self.snr_input.setText("0")

        btn_algo_event = QPushButton("Calculate Events")
        btn_algo_event.clicked.connect(self.update_peaks)
        btn_clear_selected_events = QPushButton("Clear Selected Events")
        btn_clear_selected_events.clicked.connect(self.clear_selected_events)
        btn_create_event = QPushButton("Create Event")
        btn_create_event.clicked.connect(self.create_event)
        btn_verified = QPushButton("Verify/Unverify")
        btn_verified.clicked.connect(self.verification_state_changed)

        btn_stats_amp = QPushButton("Amplitude Frequency Histogram")
        btn_stats_amp.clicked.connect(lambda: self.generate_stats(type="amplitude"))
        btn_stats_iei = QPushButton("IEI Frequency Histogram")
        btn_stats_iei.clicked.connect(lambda: self.generate_stats(type="iei"))

        btn_generate_stats = QPushButton("Generate Local Statistics")
        btn_generate_stats.clicked.connect(self.generate_local_stats)

        model_name_label = QLabel("Model")
        self.cmb_model_name = QComboBox()
        self.name_to_path = {}
        # Check if there is a model available in ./ml_training/output
        if os.path.exists("./ml_training/output"):
            for root, dirs, files in os.walk("./ml_training/output"):
                for file in files:
                    if file.endswith(".pth") and "val" in file:
                        model = file.split(".")[0]
                        self.cmb_model_name.addItem(model)
                        self.name_to_path[model] = (os.path.join(root, file))

        if self.cmb_model_name.count() > 0:
            btn_run_model = QPushButton("Run Model")
            btn_run_model.clicked.connect(self.run_model)
        else:
            btn_run_model = QPushButton("No Model Available")
            btn_run_model.setEnabled(False)

        model_conf_threshold_label = QLabel("Model Confidence Threshold")
        self.model_conf_threshold_input = QLineEdit()
        self.model_conf_threshold_input.setValidator(QDoubleValidator(0, 1, 3))
        self.model_conf_threshold_input.setText("0.5")

        # Temp Picks Utility
        self.btn_toggle_temp_picks = QPushButton("Toggle Temp Picks")
        self.btn_toggle_temp_picks.clicked.connect(self.show_hide_picks)
        self.btn_show_metrics = QPushButton("Show Evaluation Metrics")
        self.btn_show_metrics.clicked.connect(self.show_metrics)
        self.cmb_confirmation_type = QComboBox()
        self.cmb_confirmation_type.addItems(["Accept Incoming Only", "Accept Overlapping Only", "Accept All"])
        self.btn_confirm_temp_picks = QPushButton("Confirm Temp Picks")
        self.btn_confirm_temp_picks.clicked.connect(self.confirm_picks)
        self.btn_confirm_temp_picks.setStyleSheet("background-color: green")
        self.btn_clear_temp_picks = QPushButton("Clear Temp Picks")
        self.btn_clear_temp_picks.clicked.connect(self.discard_picks)
        self.btn_clear_temp_picks.setStyleSheet("background-color: red")

        # Force Transients Utility
        label_force_transient = QLabel("Force/Readjust Transient Event")
        label_force_start = QLabel("Start")
        self.input_force_start = QLineEdit()
        self.input_force_start.setValidator(QIntValidator(0, 1000000))
        label_force_end = QLabel("End")
        self.input_force_end = QLineEdit()
        self.input_force_end.setValidator(QIntValidator(0, 1000000))
        self.btn_force_transient = QPushButton("No Plot Selected")
        self.btn_force_transient.clicked.connect(self.force_adjust_transient)
        self.btn_force_transient.setEnabled(False)
        

        # SavGol Utility
        self.savgol_label = QLabel("SavGol Parameters")
        self.savgol_win_len_label = QLabel("Window Length")
        self.savgol_win_len_input = QLineEdit()
        self.savgol_win_len_input.setValidator(QIntValidator(1, 100))
        self.savgol_win_len_input.setText("5")
        self.savgol_poly_order_label = QLabel("Polynomial Order")
        self.savgol_poly_order_input = QLineEdit()
        self.savgol_poly_order_input.setValidator(QIntValidator(1, 100))
        self.savgol_poly_order_input.setText("3")
        self.savgol_deriv_label = QLabel("Derivative")
        self.savgol_deriv_input = QLineEdit()
        self.savgol_deriv_input.setValidator(QIntValidator(1, 100))
        self.savgol_deriv_input.setText("0")
        self.savgol_delta_label = QLabel("Delta")
        self.savgol_delta_input = QLineEdit()
        self.savgol_delta_input.setValidator(QDoubleValidator(0, 100, 3))
        self.savgol_delta_input.setText("1.0")
        btn_savgol = QPushButton("Update SavGol")
        btn_savgol.clicked.connect(self.update_savgol)

        # Noise Utility
        self.noise_label = QLabel("Noise Parameters")
        self.noise_win_len_label = QLabel("Window Length")
        self.noise_win_len_input = QLineEdit()
        self.noise_win_len_input.setValidator(QIntValidator(1, 1000))
        self.noise_win_len_input.setText("10")
        self.noise_type_label = QLabel("Type")
        self.noise_type_combobox = QComboBox()
        self.noise_type_combobox.addItems(["None", "Mean", "Median", "Max"])
        self.noise_type_combobox.setCurrentIndex(0)
        self.noise_cap_label = QLabel("Cap")
        self.noise_cap_input = QLineEdit()
        self.noise_cap_input.setValidator(QDoubleValidator(0, 1, 4))
        self.noise_cap_input.setText("0.1")
        btn_noise = QPushButton("Update Noise")
        btn_noise.clicked.connect(self.update_noise)

        # View Utility
        self.view_y_start_label = QLabel("Y Axis Start")
        self.view_y_start_input = QLineEdit()
        self.view_y_start_input.setValidator(QDoubleValidator(-5, 0, 3))
        self.view_y_start_input.setText("-1")
        self.view_y_end_label = QLabel("Y Axis End")
        self.view_y_end_input = QLineEdit()
        self.view_y_end_input.setValidator(QDoubleValidator(0, 50, 3))
        self.view_y_end_input.setText("10")
        self.view_window_label = QLabel("Window Size")
        self.view_window_input = QLineEdit()
        self.view_window_input.setValidator(QIntValidator(100, 100000))
        self.view_window_input.setText("1000")
        self.view_btn_update = QPushButton("Update View")
        self.view_btn_update.clicked.connect(self.update_plot_view)



        self.chkbox_plot_options_C = QCheckBox("C Signal")
        self.chkbox_plot_options_C.setStyleSheet("background-color: white; border: 1px solid black; width: 15px; height: 15px;")
        self.chkbox_plot_options_S = QCheckBox("S Signal")
        self.chkbox_plot_options_S.setStyleSheet("background-color: magenta; border: 1px solid black; width: 15px; height: 15px;")
        self.chkbox_plot_options_YrA = QCheckBox("Raw Signal")
        self.chkbox_plot_options_YrA.setStyleSheet("background-color: cyan; border: 1px solid black; width: 15px; height: 15px;")
        self.chkbox_plot_options_dff = QCheckBox("ΔF/F")
        self.chkbox_plot_options_dff.setStyleSheet("background-color: yellow; border: 1px solid black; width: 15px; height: 15px;")
        self.chkbox_plot_options_savgol = QCheckBox("SavGol Filter (ΔF/F)")
        self.chkbox_plot_options_savgol.setStyleSheet("background-color: rgb(154,205,50); border: 1px solid black; width: 15px; height: 15px;")
        self.chkbox_plot_options_noise = QCheckBox("Noise")
        self.chkbox_plot_options_noise.setStyleSheet("background-color: rgb(0,191,255); border: 1px solid black; width: 15px; height: 15px;")
        self.chkbox_plot_options_snr = QCheckBox("SNR")
        self.chkbox_plot_options_snr.setStyleSheet("background-color: rgb(255,105,180); border: 1px solid black; width: 15px; height: 15px;")
        self.btn_reset_view = QPushButton("Reset View")
        self.btn_reset_view.clicked.connect(lambda: self.visualize_signals(reset_view=True))
        self.chkbox_plot_options_C.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        self.chkbox_plot_options_S.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        self.chkbox_plot_options_YrA.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        self.chkbox_plot_options_dff.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        self.chkbox_plot_options_savgol.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        self.chkbox_plot_options_noise.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        self.chkbox_plot_options_snr.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        self.chkbox_plot_options_C.clicked.connect(self.enable_disable_event_buttons)
        self.chkbox_plot_options_C.setChecked(True)

        # Global View Utility
        self.chkbox_plot_global_C = QCheckBox("C Signal")
        self.chkbox_plot_global_C.setStyleSheet("background-color: white; border: 1px solid black; width: 15px; height: 15px;")
        self.chkbox_plot_global_S = QCheckBox("S Signal")
        self.chkbox_plot_global_S.setStyleSheet("background-color: magenta; border: 1px solid black; width: 15px; height: 15px;")
        self.chkbox_plot_global_YrA = QCheckBox("Raw Signal")
        self.chkbox_plot_global_YrA.setStyleSheet("background-color: cyan; border: 1px solid black; width: 15px; height: 15px;")
        self.chkbox_plot_global_dff = QCheckBox("ΔF/F")
        self.chkbox_plot_global_dff.setStyleSheet("background-color: yellow; border: 1px solid black; width: 15px; height: 15px;")
        self.btn_global_reset_view = QPushButton("Reset View")
        # Which group of cells to visualize
        label_global_which_cells = QLabel("Which Cells to Visualize")
        self.list_global_which_cells = QComboBox()
        self.list_global_which_cells.addItems(["All Cells", "Verified Cells"])
        unique_groups = self.session.get_group_ids()
        self.list_global_which_cells.addItems([f"Group {group}" for group in unique_groups])
        self.list_global_which_cells.currentIndexChanged.connect(lambda: self.visualize_global_signals(reset_view=False))
        # Input for window size
        global_window_size_label = QLabel("Window Size")
        self.global_window_size_input = QLineEdit()
        self.global_window_size_input.setValidator(QIntValidator(1, 1000))
        self.global_window_size_input.setText("1")
        self.global_window_preview_chkbox = QCheckBox("Preview")
        self.global_window_preview_chkbox.clicked.connect(lambda: self.visualize_global_signals(reset_view=False))
        self.global_window_size_btn = QPushButton("Update Size")
        self.global_window_size_btn.clicked.connect(lambda: self.visualize_global_signals(reset_view=False))
        self.layout_global_window_size = QHBoxLayout()

        self.layout_global_window_size.addWidget(global_window_size_label)
        self.layout_global_window_size.addWidget(self.global_window_size_input)
        self.layout_global_window_size.addWidget(self.global_window_preview_chkbox)
        self.layout_global_window_size.addWidget(self.global_window_size_btn)
        # Set which method for gathering averages
        layout_global_avg_method = QHBoxLayout()
        global_avg_method_label = QLabel("Averaging Method:")
        self.global_avg_method = QComboBox()
        self.global_avg_method.addItems(["Rolling", "Coarse"])
        self.global_avg_method.setCurrentIndex(0)
        self.global_avg_method.currentIndexChanged.connect(lambda: self.visualize_global_signals(reset_view=False))
        layout_global_avg_method.addWidget(global_avg_method_label)
        layout_global_avg_method.addWidget(self.global_avg_method)
        # CheckBoxes for global signals
        self.btn_global_reset_view.clicked.connect(lambda: self.visualize_global_signals(reset_view=True))
        self.chkbox_plot_global_C.clicked.connect(lambda: self.visualize_global_signals(reset_view=False))
        self.chkbox_plot_global_S.clicked.connect(lambda: self.visualize_global_signals(reset_view=False))
        self.chkbox_plot_global_YrA.clicked.connect(lambda: self.visualize_global_signals(reset_view=False))
        self.chkbox_plot_global_dff.clicked.connect(lambda: self.visualize_global_signals(reset_view=False))

        if "RNF" in self.session.data:
            self.chkbox_plot_options_RNF = QCheckBox("RNF")
            self.chkbox_plot_options_RNF.setStyleSheet("background-color: rgb(250, 200, 20); border: 1px solid black; width: 15px; height: 15px;")
            self.chkbox_plot_options_RNF.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        if "ALP" in self.session.data:
            self.chkbox_plot_options_ALP = QCheckBox("ALP")
            self.chkbox_plot_options_ALP.setStyleSheet("background-color: rgb(100, 50, 150); border: 1px solid black; width: 15px; height: 15px;")
            self.chkbox_plot_options_ALP.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        if "ILP" in self.session.data:
            self.chkbox_plot_options_ILP = QCheckBox("ILP")
            self.chkbox_plot_options_ILP.setStyleSheet("background-color: rgb(50, 150, 100); border: 1px solid black; width: 15px; height: 15px;")
            self.chkbox_plot_options_ILP.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        if "ALP_Timeout" in self.session.data:
            self.chkbox_plot_options_ALP_Timeout = QCheckBox("ALP Timeout")
            self.chkbox_plot_options_ALP_Timeout.setStyleSheet("background-color: rgb(60, 200, 250); border: 1px solid black; width: 15px; height: 15px;")
            self.chkbox_plot_options_ALP_Timeout.clicked.connect(lambda: self.visualize_signals(reset_view=False))
        

        # Plot Colors
        self.color_mapping = {
            "C": "w",
            "S": "m",
            "YrA": "c",
            "DFF": "y",
            "SavGol": (154,205,50), # Greenish/Yellow
            "noise": (0,191,255), # Deep Sky Blue
            "SNR": (255,105,180), # Hot Pink
            "RNF": (250, 200, 20),
            "ALP": (100, 50, 150),
            "ILP": (50, 150, 100),
            "ALP_Timeout": (60, 200, 250)
        }

        # Populate cell list
        self.refresh_cell_list()

        
        # Tools for video
        self.pixmapi_play = QStyle.StandardPixmap.SP_MediaPlay
        self.pixmapi_pause = QStyle.StandardPixmap.SP_MediaPause
        self.btn_play = QPushButton("")
        self.btn_play.setCheckable(True)
        self.btn_play.setIcon(self.style().standardIcon(self.pixmapi_play))
        self.btn_play.setFixedSize(30, 30)
        self.btn_play.clicked.connect(self.play_pause)
        
        self.pixmapi_forward = QStyle.StandardPixmap.SP_ArrowRight
        self.btn_forward = QPushButton("")
        self.btn_forward.setIcon(self.style().standardIcon(self.pixmapi_forward))
        self.btn_forward.setFixedSize(30, 30)
        self.btn_forward.clicked.connect(self.clicked_next)

        self.pixmapi_backward = QStyle.StandardPixmap.SP_ArrowLeft
        self.btn_backward = QPushButton("")
        self.btn_backward.setIcon(self.style().standardIcon(self.pixmapi_backward))
        self.btn_backward.setFixedSize(30, 30)
        self.btn_backward.clicked.connect(self.clicked_prev)

        self.scroll_video = QSlider(Qt.Orientation.Horizontal)
        self.scroll_video.setRange(0, self.video_length)
        self.scroll_video.sliderMoved.connect(self.pause_video)
        self.scroll_video.sliderReleased.connect(self.slider_update)
        self.scroll_video.valueChanged.connect(self.update_plot_lines)

        self.video_timer_label = QLabel("00:00:00")
        self.video_timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_timer_label.setFixedSize(80, 30)

        # Video interaction elements
        self.imv_cell.scene.sigMouseClicked.connect(self.video_click)
        self.video_cell_selection = set()
        self.video_cell_mask = np.zeros((self.current_video.shape[1], self.current_video.shape[2]))

        # Visualize signals selected in video
        self.w_signals = pg.GraphicsLayoutWidget()
        self.w_global_signals = pg.GraphicsLayoutWidget()
        

        # Layouts
        layout_video_cells = QHBoxLayout()
        layout_video_cells_visualize = QHBoxLayout()

        layout_video_tools = QHBoxLayout()
        layout_video_tools.addWidget(self.btn_backward)
        layout_video_tools.addWidget(self.btn_play)
        layout_video_tools.addWidget(self.btn_forward)
        layout_video_tools.addWidget(self.scroll_video)
        layout_video_tools.addWidget(self.video_timer_label)

        widget_video_subvideos = QSplitter(Qt.Orientation.Horizontal)
        widget_video_subvideos.setFrameShape(QFrame.StyledPanel)
        widget_video_subvideos.setStyleSheet(            
            "QSplitter::handle{background-color: gray; width: 5px; border: 1px dotted gray}"
        )

        widget_video_subvideos.addWidget(self.imv_cell)
        widget_video_subvideos.addWidget(self.imv_behavior)
        widget_video_subvideos.addWidget(self.visualization_3D)

        layout_video = QVBoxLayout()
        layout_video.addWidget(widget_video_subvideos)
        layout_video.addLayout(layout_video_tools)

        layout_highlight_mode = QHBoxLayout()
        layout_highlight_mode.addWidget(cell_highlight_mode_label)
        layout_highlight_mode.addWidget(self.cmb_cell_highlight_mode)

        layout_add_remove_group = QHBoxLayout()
        layout_add_remove_group.addWidget(self.btn_add_to_group)
        layout_add_remove_group.addWidget(self.btn_remove_from_group)

        layout_cells = QVBoxLayout()
        layout_cells.addWidget(w_cell_label)
        layout_cells.addWidget(self.list_cell)
        layout_cells.addWidget(self.btn_cell_focus)
        layout_cells.addWidget(self.btn_cell_reset)
        layout_cells.addLayout(layout_highlight_mode)
        layout_cells.addLayout(layout_add_remove_group)
        layout_cells.addWidget(self.btn_cell_reject)
        layout_cells.addWidget(btn_verified)
        w_cells = QWidget()
        w_cells.setLayout(layout_cells)

        # 3D Visualization Tools
        visualization_3D_layout = QVBoxLayout()
        
        label_3D_which_cells = QLabel("Which Cells to Visualize")
        self.list_3D_which_cells = QComboBox()
        self.list_3D_which_cells.addItems(["All Cells", "Verified Cells"])
        self.list_3D_which_cells.addItems([f"Group {group}" for group in unique_groups])
        label_3D_functions = QLabel("3D Visualization Functions")
        self.dropdown_3D_functions = QComboBox()
        self.dropdown_3D_functions.addItems(["Raw Visualization", "Transient Visualization"])
        self.dropdown_3D_functions.currentIndexChanged.connect(self.changed_3D_function)
        self.dropdown_3D_data_types = QComboBox()
        self.dropdown_3D_data_types.addItems(["C", "DFF", "Binary Transient"])
        self.dropdown_3D_data_types.currentIndexChanged.connect(self.changed_3D_data_type)

        self.layout_3D_chkbox_parent = QWidget()
        layout_3D_chkbox = QHBoxLayout(self.layout_3D_chkbox_parent)
        self.chkbox_3D_cumulative = QCheckBox("Cumulative")
        self.chkbox_3D_normalize = QCheckBox("Normalize")
        self.chkbox_3D_average = QCheckBox("Average")
        layout_3D_chkbox.addWidget(self.chkbox_3D_cumulative)
        layout_3D_chkbox.addWidget(self.chkbox_3D_normalize)
        layout_3D_chkbox.addWidget(self.chkbox_3D_average)
        self.layout_3D_chkbox_parent.hide()

        # Cofiring Tools
        cofiring_layout = QVBoxLayout()
        cofiring_layout.addWidget(QLabel("Cofiring Tools"))
        cofiring_window_layout = QHBoxLayout()
        self.cofiring_window_size = QLineEdit()
        self.cofiring_window_btn = QPushButton("Update Cofiring Window")
        self.cofiring_window_btn.clicked.connect(self.update_cofiring_window)
        cofiring_window_layout.addWidget(self.cofiring_window_size)
        cofiring_window_layout.addWidget(self.cofiring_window_btn)
        self.cofiring_window_size.setValidator(QIntValidator(1, 1000))
        self.cofiring_window_size.setText("30")
        self.cofiring_chkbox = QCheckBox("Show Cofiring")
        self.cofiring_shareA_chkbox = QCheckBox("Share A")
        self.cofiring_shareA_chkbox.setChecked(True)
        self.cofiring_shareA_chkbox.clicked.connect(lambda: self.update_cofiring_window(reset_list=True))
        self.cofiring_shareB_chkbox = QCheckBox("Share B")
        self.cofiring_shareB_chkbox.setChecked(True)
        self.cofiring_shareB_chkbox.clicked.connect(lambda: self.update_cofiring_window(reset_list=True))
        self.cofiring_direction_dropdown = QComboBox()
        self.cofiring_direction_dropdown.addItems(["Bidirectional", "Forward", "Backward"])
        self.cofiring_direction_dropdown.currentIndexChanged.connect(lambda: self.update_cofiring_window(reset_list=True))
        self.cofiring_chkbox.clicked.connect(self.visualize_cofiring)
        self.cofiring_list = QListWidget()
        self.cofiring_list.itemChanged.connect(lambda: self.update_cofiring_window(reset_list=False))
        self.cofiring_individual_cell_list = QListWidget()
        self.cofiring_individual_cell_list.itemChanged.connect(lambda: self.update_cofiring_window(reset_list=False))
        btn_cofiring_2d_show = QPushButton("Show 2D Representation")
        btn_cofiring_2d_show.clicked.connect(self.show_2D_cofiring)

        # Smoothing
        frame_smoothing = QFrame()
        frame_smoothing.setFrameShape(QFrame.StyledPanel)
        frame_smoothing.setFrameShadow(QFrame.Raised)
        frame_smoothing.setLineWidth(3)
        layout_smoothing = QVBoxLayout(frame_smoothing)
        label_smoothing = QLabel("Smoothing")
        layout_smoothing_type = QHBoxLayout()
        label_smoothing_type = QLabel("Type:")
        self.dropdown_smoothing_type = QComboBox()
        self.dropdown_smoothing_type.addItems(["Mean"])
        layout_smoothing_type.addWidget(label_smoothing_type)
        layout_smoothing_type.addWidget(self.dropdown_smoothing_type)
        layout_smoothing_size = QHBoxLayout()
        label_smoothing_size = QLabel("Size:")
        self.input_smoothing_size = QLineEdit()
        self.input_smoothing_size.setValidator(QIntValidator(1, 1000))
        self.input_smoothing_size.setText("1")
        layout_smoothing_size.addWidget(label_smoothing_size)
        layout_smoothing_size.addWidget(self.input_smoothing_size)
        layout_smoothing.addWidget(label_smoothing)
        layout_smoothing.addLayout(layout_smoothing_type)
        layout_smoothing.addLayout(layout_smoothing_size)

        # Window Size
        self.frame_3D_window_size = QFrame()
        self.frame_3D_window_size.setFrameShape(QFrame.StyledPanel)
        self.frame_3D_window_size.setFrameShadow(QFrame.Raised)
        self.frame_3D_window_size.setLineWidth(3)
        self.frame_3D_window_size.hide()
        layout_window_size = QHBoxLayout(self.frame_3D_window_size)
        label_window_size = QLabel("Window Size:")
        self.input_3D_window_size = QLineEdit()
        self.input_3D_window_size.setValidator(QIntValidator(1, 10000))
        self.input_3D_window_size.setText("1")
        layout_window_size.addWidget(label_window_size)
        layout_window_size.addWidget(self.input_3D_window_size)


        # 3D Z Axis Scaling
        frame_3D_scaling = QFrame()
        frame_3D_scaling.setFrameShape(QFrame.StyledPanel)
        frame_3D_scaling.setFrameShadow(QFrame.Raised)
        frame_3D_scaling.setLineWidth(3)
        layout_3D_scaling = QVBoxLayout(frame_3D_scaling)
        label_3D_slider = QLabel("Scale Z Axis")
        self.slider_value = QLabel("1")
        self.slider_value.setFixedWidth(30)
        self.slider_3D_scaling = QSlider(Qt.Orientation.Horizontal)
        self.slider_3D_scaling.setRange(1, 1000)
        self.slider_3D_scaling.setValue(1)
        self.slider_3D_scaling.valueChanged.connect(lambda: self.slider_value.setText(str(self.slider_3D_scaling.value())))
        layout_3D_slider = QHBoxLayout()
        layout_3D_slider.addWidget(self.slider_3D_scaling)
        layout_3D_slider.addWidget(self.slider_value)
        layout_3D_scaling.addWidget(label_3D_slider)
        layout_3D_scaling.addLayout(layout_3D_slider)

        self.btn_3D_visualize = QPushButton("Visualize")
        self.btn_3D_visualize.clicked.connect(self.visualize_3D)

        # Color Mapping
        layout_colormap = QHBoxLayout()
        layout_colormap.addWidget(QLabel("Colormap:"))
        dropdown_3D_colormap = QComboBox()
        dropdown_3D_colormap.addItems(lut_mode_list())
        # Set index to whatever hot is
        dropdown_3D_colormap.setCurrentIndex(lut_mode_list().index("hot"))
        dropdown_3D_colormap.currentIndexChanged.connect(lambda: self.visualization_3D.change_colormap(dropdown_3D_colormap.currentText()))
        layout_colormap.addWidget(dropdown_3D_colormap)


        visualization_3D_layout.addWidget(label_3D_functions)
        visualization_3D_layout.addWidget(self.dropdown_3D_functions)
        visualization_3D_layout.addWidget(self.dropdown_3D_data_types)
        visualization_3D_layout.addWidget(self.layout_3D_chkbox_parent)
        visualization_3D_layout.addWidget(frame_smoothing)
        visualization_3D_layout.addWidget(self.frame_3D_window_size)
        visualization_3D_layout.addWidget(frame_3D_scaling)
        visualization_3D_layout.addLayout(layout_colormap)
        visualization_3D_layout.addWidget(self.btn_3D_visualize)
        visualization_3D_layout.addStretch()
        visualization_3D_tools = QWidget()
        visualization_3D_tools.setLayout(visualization_3D_layout)

        # Co-Firing Checkbox layout
        cofiring_chkbox_layout = QHBoxLayout()
        cofiring_chkbox_layout.addWidget(self.cofiring_chkbox)
        cofiring_chkbox_layout.addWidget(self.cofiring_shareA_chkbox)
        cofiring_chkbox_layout.addWidget(self.cofiring_shareB_chkbox)

        # Tab Co-Firing Options
        tabs_cofiring_options.addTab(self.cofiring_list, "Group Co-Firing")
        tabs_cofiring_options.addTab(self.cofiring_individual_cell_list, "Individual Cells")

        # Co-Firing Tools
        cofiring_layout.addLayout(cofiring_window_layout)
        cofiring_layout.addLayout(cofiring_chkbox_layout)
        cofiring_layout.addWidget(self.cofiring_direction_dropdown)
        cofiring_layout.addWidget(tabs_cofiring_options)
        cofiring_layout.addWidget(btn_cofiring_2d_show)
        cofiring_layout.addStretch()
        cofiring_tools = QWidget()
        cofiring_tools.setLayout(cofiring_layout)

        # Rejected
        layout_rejected_justification_utility = QHBoxLayout()
        layout_rejected_justification_utility.addWidget(self.btn_justification_save)
        layout_rejected_justification_utility.addWidget(self.btn_justification_cancel)
        layout_rejected_cells = QVBoxLayout()
        layout_rejected_cells.addWidget(w_rejected_cell_label)
        layout_rejected_cells.addWidget(self.list_rejected_cell)
        layout_rejected_cells.addWidget(self.btn_justification_start)
        layout_rejected_cells.addWidget(self.input_justification)
        layout_rejected_cells.addLayout(layout_rejected_justification_utility)
        layout_rejected_cells.addWidget(self.btn_cell_return)
        w_rejected_cells = QWidget()
        w_rejected_cells.setLayout(layout_rejected_cells)

        layout_missed_cells = QVBoxLayout()
        layout_missed_cells.addWidget(w_missed_cell_label)
        layout_missed_cells.addWidget(self.list_missed_cell)
        layout_missed_cells.addWidget(self.btn_missed_select)
        layout_missed_cells.addWidget(self.w_missed_utility)
        layout_missed_cells.addWidget(btn_missed_focus_mask)
        layout_missed_cells.addWidget(btn_missed_reset_mask)
        layout_missed_cells.addWidget(self.btn_missed_remove)
        w_missed_cells = QWidget()
        w_missed_cells.setLayout(layout_missed_cells)    

        self.tabs_video.addTab(w_cells, "Approved Cells")
        self.tabs_video.addTab(w_rejected_cells, "Rejected Cells")
        self.tabs_video.addTab(w_missed_cells, "Missed Cells")
        self.tabs_video.currentChanged.connect(self.switched_tabs)

        tabs_visualization_layout = QVBoxLayout()
        tabs_visualization_layout.addWidget(label_3D_which_cells)
        tabs_visualization_layout.addWidget(self.list_3D_which_cells)
        tabs_visualization_layout.addWidget(self.tabs_visualization)
        tabs_visualization_parent = QWidget()
        tabs_visualization_parent.setLayout(tabs_visualization_layout)
        

        self.tabs_visualization.addTab(visualization_3D_tools, "Signal Settings")
        self.tabs_visualization.addTab(cofiring_tools, "Co-Firing")

        self.tabs_video_tools.addTab(self.tabs_video, "Cell Video")
        self.tabs_video_tools.addTab(tabs_visualization_parent, "3D Visualization")

        # General plot utility
        layout_plot_utility = QVBoxLayout()
        layout_plot_utility.addStretch()
        layout_plot_utility.setDirection(3)

        # Clear Traces Button
        btn_clear_traces = QPushButton("Clear Selected Traces")
        btn_clear_traces.clicked.connect(self.clear_selected_traces)

        # Clear Events Button
        btn_clear_events = QPushButton("Clear All Events")
        btn_clear_events.clicked.connect(self.clear_all_events)



        # Event Generation Algorithm
        layout_height = QHBoxLayout()
        layout_height.addWidget(self.min_height_label)
        layout_height.addWidget(self.min_height_input)
        layout_dist = QHBoxLayout()
        layout_dist.addWidget(self.dist_label)
        layout_dist.addWidget(self.dist_input)
        layout_auc = QHBoxLayout()
        layout_auc.addWidget(self.snr_label)
        layout_auc.addWidget(self.snr_input)

        self.frame_algo_events = QFrame()
        self.frame_algo_events.setFrameShape(QFrame.Box)
        self.frame_algo_events.setFrameShadow(QFrame.Raised)
        self.frame_algo_events.setLineWidth(3)
        layout_algo_events = QVBoxLayout(self.frame_algo_events)
        layout_algo_events.addWidget(self.auto_label)
        layout_algo_events.addLayout(layout_height)
        layout_algo_events.addLayout(layout_dist)
        layout_algo_events.addLayout(layout_auc)
        layout_algo_events.addWidget(btn_algo_event)
        layout_algo_events.addStretch()

        # Machine Learning Event Generation
        frame_ml_events = QFrame()
        frame_ml_events.setFrameShape(QFrame.Box)
        frame_ml_events.setFrameShadow(QFrame.Raised)
        frame_ml_events.setLineWidth(3)

        layout_ml_name = QHBoxLayout()
        layout_ml_name.addWidget(model_name_label)
        layout_ml_name.addWidget(self.cmb_model_name)
        layout_ml_threshold = QHBoxLayout()
        layout_ml_threshold.addWidget(model_conf_threshold_label)
        layout_ml_threshold.addWidget(self.model_conf_threshold_input)
        layout_ml = QVBoxLayout(frame_ml_events)
        layout_ml.addLayout(layout_ml_name)
        layout_ml.addLayout(layout_ml_threshold)
        layout_ml.addWidget(btn_run_model)

        # Machine Learning Results
        frame_ml_results = QFrame()
        frame_ml_results.setFrameShape(QFrame.Box)
        frame_ml_results.setFrameShadow(QFrame.Raised)
        frame_ml_results.setLineWidth(3)

        layout_ml_results = QVBoxLayout(frame_ml_results)
        # Extract the actual name from the window
        self.actual_name = "".join(self.name.split(" ")[0:3])
        self.actual_name = self.actual_name[:5] + "_" + self.actual_name[5:]
        self.cmb_experiment = QComboBox()
        self.cmb_experiment.addItems(['cross_animal', 'cross_day_cross_session', 'cross_day_same_session', 'cross_session_same_day', 'within_session'])
        self.cmb_experiment.currentIndexChanged.connect(self.check_if_results_exist)
        self.cmb_testing_set = QComboBox()
        self.cmb_testing_set.addItems(["PL010_D1S1", "AA058_D1S1", "AA036_D2S1", "AA034_D1S1"])
        self.cmb_testing_set.currentIndexChanged.connect(self.check_if_results_exist)
        label_no_of_cells = QLabel("No of Cells")
        self.cmb_no_of_cells = QComboBox()
        self.cmb_no_of_cells.addItems(["1", "2", "5", "10", "15", "20"])
        self.cmb_no_of_cells.setEnabled(False)
        self.cmb_no_of_cells.currentIndexChanged.connect(self.changed_no_cells)
        label_which_run = QLabel("Which Run")
        self.cmb_which_run = QComboBox()
        self.cmb_which_run.addItems(["1", "2", "3", "4", "5"])
        self.cmb_which_run.setEnabled(False)
        self.btn_generate_ml_results = QPushButton("Generate ML Results")
        self.btn_generate_ml_results.clicked.connect(self.visualize_ml_test_results)
        self.btn_generate_ml_results.setEnabled(False)


        # Temp toggling/confirming layout
        frame_temp_picks = QFrame()
        frame_temp_picks.setFrameShape(QFrame.Box)
        frame_temp_picks.setFrameShadow(QFrame.Raised)
        frame_temp_picks.setLineWidth(3)

        layout_temp_picks = QVBoxLayout(frame_temp_picks)
        layout_temp_picks.addWidget(self.btn_toggle_temp_picks)
        layout_temp_picks.addWidget(self.btn_show_metrics)
        layout_temp_picks.addWidget(self.cmb_confirmation_type)
        layout_temp_picks.addWidget(self.btn_confirm_temp_picks)
        layout_temp_picks.addWidget(self.btn_clear_temp_picks)

        
        layout_auto_ml = QVBoxLayout()
        layout_auto_ml.addWidget(self.frame_algo_events)
        # Auto and ML into one widget if torch exists
        if torch_imported:
            layout_auto_ml.addWidget(frame_ml_events)
        layout_auto_ml.addWidget(frame_temp_picks)
        layout_auto_ml.addStretch()
        w_auto_ml = QWidget()
        w_auto_ml.setLayout(layout_auto_ml)


        # Machine Learning Results
        # This should be removed later down the line
        w_ml_results = QWidget()
        w_ml_results.setLayout(layout_ml_results)
        layout_ml_results.addWidget(self.cmb_experiment)
        layout_ml_results.addWidget(self.cmb_testing_set)
        layout_ml_results.addWidget(label_no_of_cells)
        layout_ml_results.addWidget(self.cmb_no_of_cells)
        layout_ml_results.addWidget(label_which_run)
        layout_ml_results.addWidget(self.cmb_which_run)
        layout_ml_results.addWidget(self.btn_generate_ml_results)


        # Manual Event Generation
        self.frame_manual_events = QFrame()
        self.frame_manual_events.setFrameShape(QFrame.Box)
        self.frame_manual_events.setFrameShadow(QFrame.Raised)
        self.frame_manual_events.setLineWidth(3)
        layout_manual_events = QVBoxLayout(self.frame_manual_events)
        layout_manual_events.addWidget(self.manual_label)
        layout_manual_events.addWidget(btn_create_event)
        layout_manual_events.addWidget(btn_clear_selected_events) 
        

        # Force Event Generation
        self.frame_force_events = QFrame()
        self.frame_force_events.setFrameShape(QFrame.Box)
        self.frame_force_events.setFrameShadow(QFrame.Raised)
        self.frame_force_events.setLineWidth(3)
        layout_force_events = QVBoxLayout(self.frame_force_events)
        layout_force_events.addWidget(label_force_transient)
        layout_force_start = QHBoxLayout()
        layout_force_start.addWidget(label_force_start)
        layout_force_start.addWidget(self.input_force_start)
        layout_force_end = QHBoxLayout()
        layout_force_end.addWidget(label_force_end)
        layout_force_end.addWidget(self.input_force_end)
        layout_force_events.addLayout(layout_force_start)
        layout_force_events.addLayout(layout_force_end)
        layout_force_events.addWidget(self.btn_force_transient)

        # Force and Manual into one widget
        layout_force_manual = QVBoxLayout()
        layout_force_manual.addWidget(self.frame_manual_events)
        layout_force_manual.addWidget(self.frame_force_events)
        layout_force_manual.addStretch()
        w_force_manual = QWidget()
        w_force_manual.setLayout(layout_force_manual)


        # Event Generation Tab
        tab_transient_detection = QTabWidget()
        tab_transient_detection.addTab(w_auto_ml, "Automatic")
        tab_transient_detection.addTab(w_force_manual, "Manual")
        if os.path.exists("test_cells_dict.pkl") and os.path.exists("test_results.pkl"):
            with open("test_cells_dict.pkl", "rb") as f:
                self.test_cells_dict = pickle.load(f)
            with open("test_results.pkl", "rb") as f:
                self.test_results = pickle.load(f)
            tab_transient_detection.addTab(w_ml_results, "ML Results")
            self.check_if_results_exist()

        # Statistics buttons
        frame_stats = QFrame()
        frame_stats.setFrameShape(QFrame.Box)
        frame_stats.setFrameShadow(QFrame.Raised)
        frame_stats.setLineWidth(3)
        layout_stats = QVBoxLayout(frame_stats)
        layout_stats.addStretch()
        layout_stats.setDirection(3)
        layout_stats.addWidget(btn_generate_stats)
        layout_stats.addWidget(local_stats_label)

        # SavGol Tab
        savgol_utility = QFrame()
        savgol_utility.setFrameShape(QFrame.Box)
        savgol_utility.setFrameShadow(QFrame.Raised)
        savgol_utility.setLineWidth(3)

        # Noise Tab
        noise_utility = QFrame()
        noise_utility.setFrameShape(QFrame.Box)
        noise_utility.setFrameShadow(QFrame.Raised)
        noise_utility.setLineWidth(3)

        # View Utility Frame
        view_utility = QFrame()
        view_utility.setFrameShape(QFrame.Box)
        view_utility.setFrameShadow(QFrame.Raised)
        view_utility.setLineWidth(3)     

        # SavGol Layouts
        layout_savgol = QVBoxLayout(savgol_utility)
        layout_savgol.addWidget(self.savgol_label)
        layout_savgol_win_len = QHBoxLayout()
        layout_savgol_win_len.addWidget(self.savgol_win_len_label)
        layout_savgol_win_len.addWidget(self.savgol_win_len_input)
        layout_savgol.addLayout(layout_savgol_win_len)
        layout_savgol_poly_order = QHBoxLayout()
        layout_savgol_poly_order.addWidget(self.savgol_poly_order_label)
        layout_savgol_poly_order.addWidget(self.savgol_poly_order_input)
        layout_savgol.addLayout(layout_savgol_poly_order)
        layout_savgol_deriv = QHBoxLayout()
        layout_savgol_deriv.addWidget(self.savgol_deriv_label)
        layout_savgol_deriv.addWidget(self.savgol_deriv_input)
        layout_savgol.addLayout(layout_savgol_deriv)
        layout_savgol_delta = QHBoxLayout()
        layout_savgol_delta.addWidget(self.savgol_delta_label)
        layout_savgol_delta.addWidget(self.savgol_delta_input)
        layout_savgol.addLayout(layout_savgol_delta)
        layout_savgol.addWidget(btn_savgol)

        # Noise Layouts
        layout_noise = QVBoxLayout(noise_utility)
        layout_noise.addWidget(self.noise_label)
        layout_noise_win_len = QHBoxLayout()
        layout_noise_win_len.addWidget(self.noise_win_len_label)
        layout_noise_win_len.addWidget(self.noise_win_len_input)
        layout_noise.addLayout(layout_noise_win_len)
        layout_noise_type = QHBoxLayout()
        layout_noise_type.addWidget(self.noise_type_label)
        layout_noise_type.addWidget(self.noise_type_combobox)
        layout_noise_cap = QHBoxLayout()
        layout_noise_cap.addWidget(self.noise_cap_label)
        layout_noise_cap.addWidget(self.noise_cap_input)
        layout_noise.addLayout(layout_noise_type)
        layout_noise.addLayout(layout_noise_cap)
        layout_noise.addWidget(btn_noise)
        layout_noise.addStretch()

        # View Layout
        layout_plot_view = QVBoxLayout(view_utility)
        layout_plot_view_y_start = QHBoxLayout()
        layout_plot_view_y_start.addWidget(self.view_y_start_label)
        layout_plot_view_y_start.addWidget(self.view_y_start_input)
        layout_plot_view_y_end = QHBoxLayout()
        layout_plot_view_y_end.addWidget(self.view_y_end_label)
        layout_plot_view_y_end.addWidget(self.view_y_end_input)
        layout_plot_view_window = QHBoxLayout()
        layout_plot_view_window.addWidget(self.view_window_label)
        layout_plot_view_window.addWidget(self.view_window_input)
        layout_plot_view.addLayout(layout_plot_view_y_start)
        layout_plot_view.addLayout(layout_plot_view_y_end)
        layout_plot_view.addLayout(layout_plot_view_window)
        layout_plot_view.addWidget(self.view_btn_update)
        layout_plot_view.addStretch()
        
        
        
        # Param Tabs
        tab_params = QTabWidget()
        tab_params.addTab(savgol_utility, "SavGol")
        tab_params.addTab(noise_utility, "Noise")
        tab_params.addTab(view_utility, "View")

        # Plot options
        frame_plot_options = QFrame()
        frame_plot_options.setFrameShape(QFrame.Box)
        frame_plot_options.setFrameShadow(QFrame.Raised)
        frame_plot_options.setLineWidth(3)
        layout_plot_options = QVBoxLayout(frame_plot_options)
        layout_plot_options.addStretch()
        layout_plot_options.setDirection(3)
        layout_plot_options.addWidget(self.btn_reset_view)
        if "RNF" in self.session.data:
            layout_plot_options.addWidget(self.chkbox_plot_options_RNF)
        if "ILP" in self.session.data:
            layout_plot_options.addWidget(self.chkbox_plot_options_ILP)
        if "ALP" in self.session.data:
            layout_plot_options.addWidget(self.chkbox_plot_options_ALP)
        if "ALP_Timeout" in self.session.data:
            layout_plot_options.addWidget(self.chkbox_plot_options_ALP_Timeout)
        layout_plot_options.addWidget(self.chkbox_plot_options_snr)
        layout_plot_options.addWidget(self.chkbox_plot_options_noise)
        layout_plot_options.addWidget(self.chkbox_plot_options_savgol)
        layout_plot_options.addWidget(self.chkbox_plot_options_dff)
        layout_plot_options.addWidget(self.chkbox_plot_options_S)
        layout_plot_options.addWidget(self.chkbox_plot_options_C)
        layout_plot_options.addWidget(self.chkbox_plot_options_YrA)

        # Global plot options
        frame_global_plot_options = QFrame()
        frame_global_plot_options.setFrameShape(QFrame.Box)
        frame_global_plot_options.setFrameShadow(QFrame.Raised)
        frame_global_plot_options.setLineWidth(3)
        frame_global_plot_options.setMaximumWidth(300)
        layout_global_plot_options = QVBoxLayout(frame_global_plot_options)
        layout_global_plot_options.addStretch()
        layout_global_plot_options.setDirection(3)
        layout_global_plot_options.addWidget(self.btn_global_reset_view)
        layout_global_plot_options.addLayout(layout_global_avg_method)
        layout_global_plot_options.addLayout(self.layout_global_window_size)
        layout_global_plot_options.addWidget(self.list_global_which_cells)
        layout_global_plot_options.addWidget(label_global_which_cells)
        layout_global_plot_options.addWidget(self.chkbox_plot_global_C)
        layout_global_plot_options.addWidget(self.chkbox_plot_global_S)
        layout_global_plot_options.addWidget(self.chkbox_plot_global_YrA)
        layout_global_plot_options.addWidget(self.chkbox_plot_global_dff)


        layout_plot_utility.addWidget(tab_params)
        layout_plot_utility.addWidget(frame_plot_options)
        widget_plot_utility = QWidget()
        widget_plot_utility.setLayout(layout_plot_utility)
        widget_plot_utility.setMaximumWidth(320)

        tabs_signal.addTab(widget_plot_utility, "Params")
        tabs_signal.addTab(tab_transient_detection, "Event Detection")
        tabs_signal.addTab(frame_stats, "Local Stats")

        layout_video_cells.addLayout(layout_video)
        layout_video_cells.addWidget(self.tabs_video_tools_parent)
        self.widget_video_cells = QWidget()
        self.widget_video_cells.setLayout(layout_video_cells)

        layout_video_cells_visualize.addWidget(self.widget_video_cells)

        layout_plot_utility = QVBoxLayout()
        layout_plot_utility.addWidget(btn_clear_traces)
        layout_plot_utility.addWidget(btn_clear_events)
        layout_plot_utility.addWidget(tabs_signal_parent)

        layout_plot = QHBoxLayout()
        layout_plot.addWidget(self.w_signals)
        layout_plot.addLayout(layout_plot_utility)
        widget_plot = QWidget()
        widget_plot.setLayout(layout_plot)
        layout_video_cells_visualize.addWidget(widget_plot)
        widget_video_cells_visualize = QWidget()
        widget_video_cells_visualize.setLayout(layout_video_cells_visualize)

        layout_global_plot_utility = QVBoxLayout()
        layout_global_plot_utility.addWidget(frame_global_plot_options)

        layout_global_cell_switch = QHBoxLayout()
        layout_global_cell_switch.addWidget(self.w_global_signals)
        layout_global_cell_switch.addLayout(layout_global_plot_utility)
        widget_global_cell_switch = QWidget()
        widget_global_cell_switch.setLayout(layout_global_cell_switch)
        self.tabs_global_cell_switch.addTab(widget_video_cells_visualize, "Per Cell View")
        self.tabs_global_cell_switch.addTab(widget_global_cell_switch, "Global View")
        

        layout = QVBoxLayout()
        main_widget = QSplitter(Qt.Orientation.Vertical)
        main_widget.setFrameShape(QFrame.StyledPanel)
        main_widget.setStyleSheet(            
            "QSplitter::handle{background-color: gray; width: 5px; border: 1px dotted gray}"
        )
        main_widget.addWidget(self.widget_video_cells)
        main_widget.addWidget(self.tabs_global_cell_switch)
        layout.addWidget(main_widget)
        layout.setMenuBar(menu)

        self.setLayout(layout)
        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))

        self.video_timer = QTimer()
        interval = self.session.get_video_interval()
        self.video_timer.setInterval(interval)
        self.video_timer.timeout.connect(self.next_frame)

        self.missed_cell_init()

        self.imv_cell.scene.sigMouseMoved.connect(self.detect_cell_hover)

    def highlight_roi_selection(self, type="ellipse"):
        # Get mouse position
        pos = self.imv_cell.last_pos
        if type == "ellipse":
            roi = pg.EllipseROI(pos, [40, 40], pen=pg.mkPen(color='r', width=2), removable=True)
        elif type == "rectangle":
            roi = pg.RectROI(pos, [40, 40], pen=pg.mkPen(color='r', width=2), removable=True)
        else:
            raise ValueError("Invalid ROI type")
        
        def remove_roi(roi):
            self.imv_cell.removeItem(roi)

        def add_group_roi(roi, verified=False):
            # From the ROI we'll get the values of the position, height and width and angle to calculate whether the centroids are contained
            # in the ROI
            pos = roi.pos()
            size = roi.size()
            angle = roi.angle()

            # Now check if we're dealing with an ellipse or a rectangle
            roi_type = roi.__class__.__name__
            ids = set()
            for centroid in self.session.centroids_to_cell_ids.keys():
                id = self.session.centroids_to_cell_ids[centroid]
                if roi_type == "EllipseROI":
                    if self.within_ellipse(centroid, pos, size, angle):
                        ids.add(id)
                elif roi_type == "RectROI":
                    if self.within_rectangle(centroid, pos, size, angle):
                        ids.add(id)

            if verified:
                # Prune out non-verified cells
                ids = self.session.prune_non_verified(ids)


            # Now that we have the ids we will clear the selections in the cell list and select the cells that are within the ROI
            self.list_cell.clearSelection()
            for i in range(self.list_cell.count()):
                item = self.list_cell.item(i)
                cell_id = int(item.text().split(" ")[0])
                if cell_id in ids:
                    item.setSelected(True)
            
            # Now Focus on the cells
            self.focus_mask()
                

        self.imv_cell.addItem(roi)

        roi.sigRemoveRequested.connect(remove_roi)

        # Add action to the ROI menu
        menu = roi.getMenu()
        action_verified = QAction("Highlight Selection Verified", self)
        action_verified.triggered.connect(lambda: add_group_roi(roi, verified=True))
        menu.addAction(action_verified)
        action_standard = QAction("Highlight Selection", self)
        action_standard.triggered.connect(lambda: add_group_roi(roi, verified=False))
        menu.addAction(action_standard)



    def within_rectangle(self, point, upper_left, size, angle):
        """
        Check if a point is within a rectangle.
        
        Parameters:
        - point: The point to check (x, y).
        - upper_left: The point indicating the top left corner of the ROI.
        - size: The size of the rectangle (width, height).
        - angle: The rotation angle of the rectangle in degrees.
        
        Returns:
        - True if the point is within the rectangle, False otherwise.
        """
        point = [point[1], point[0]]
        angle = np.radians(angle)

        point_local = np.array(point) - upper_left

        # Create the rotation matrix for the inverse rotation (to unrotate the point)
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])

        # Apply the rotation to the translated point
        rotated_point = np.dot(rotation_matrix, point_local)

        # Check if the point is within the axis-aligned rectangle (after unrotating)
        if 0 <= rotated_point[0] <= size[0] and 0 <= rotated_point[1] <= size[1]:
            return True
        return False
    
    def within_ellipse(self, point, upper_left, size, angle):
        """
        Check if a point is within an ellipse.
        
        Parameters:
        - point: The point to check (x, y).
        - upper_left: The point indicating the top left corner of the ROI.
        - size: The size of the ellipse (width, height).
        - angle: The rotation angle of the ellipse in degrees.
        
        Returns:
        - True if the point is within the ellipse, False otherwise.
        """
        point = np.array([point[1], point[0]])
        angle = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])
        
        # Step 1: Calculate the center of the ellipse (not upper_left)
        center = np.array(upper_left) + np.array(size) / 2.0

        # Step 1.1: Rotate the center where upper_left is the origin
        center_local = center - upper_left
        center_rotated = np.dot(rotation_matrix, center_local)
        center_adjusted = center_rotated + upper_left

        # Step 1.2: Get the vector from center to center_adjusted
        center_vector = center_adjusted - center

        
        # Step 2: Translate the point relative to the center of the bounding box
        point_local = point - center

        # Step 3: Apply the rotation to the translated point
        rotated_point = np.dot(rotation_matrix, point_local)
        rotated_point += center_vector

        # Step 4: Treat it like a rectangle until now, but now apply the ellipse boundary check
        semi_major = size[0] / 2.0  # a (semi-major axis)
        semi_minor = size[1] / 2.0  # b (semi-minor axis)

        x_prime, y_prime = rotated_point
        ellipse_eq = (x_prime / semi_major) ** 2 + (y_prime / semi_minor) ** 2

        # Step 5: Check if the point lies within the ellipse
        return ellipse_eq <= 1
            


    def visualize_cofiring(self):
        visualize_cofiring = self.cofiring_chkbox.isChecked()
        
        if not visualize_cofiring:
            self.visualization_3D.remove_cofiring()
            return
        else:
            self.update_cofiring_window()

    def update_cofiring_window(self, reset_list=True):
        window_size = int(self.cofiring_window_size.text())
        visualize_cofiring = self.cofiring_chkbox.isChecked()

        cells_for_cofiring = self.list_3D_which_cells.currentText()
        shareA = self.cofiring_shareA_chkbox.isChecked()
        shareB = self.cofiring_shareB_chkbox.isChecked()
        direction = self.cofiring_direction_dropdown.currentText().lower()

        # Get items that are checked
        cofiring_nums = set()
        for i in range(self.cofiring_list.count()):
            item = self.cofiring_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                cofiring_nums.add(int(item.text().split(" ")[1]))

        # When we reset the list we want to visualize all cofiring connections
        if reset_list:
            cofiring_nums.add("all")
        
        cofiring_cells = set()
        for i in range(self.cofiring_individual_cell_list.count()):
            item = self.cofiring_individual_cell_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                cofiring_cells.add(int(item.text().split(" ")[1]))

        kwargs = {"nums_to_visualize": cells_for_cofiring, "visualize": visualize_cofiring, "cofiring_nums": cofiring_nums,
                   "shareA": shareA, "shareB": shareB, "direction": direction, "cofiring_cells": cofiring_cells}

        precalculated_values = self.visualization_3D.change_cofiring_window(window_size, **kwargs)
        if reset_list:
            self.reset_list(precalculated_values)
            

    def reset_list(self, precalculated_values):
        cofiring_nums = list(precalculated_values["number"].keys())
        # Populate the list
        self.cofiring_list.clear()
        # Add cofiring numbers to the list and make them checkable
        cofiring_nums.sort()
        # Remove 0 if it exists because we are not interested in no cofiring connections.
        cofiring_nums = cofiring_nums[1:] if cofiring_nums[0] == 0 else cofiring_nums
        self.cofiring_list.itemChanged.disconnect()
        for num in cofiring_nums:
            item = QListWidgetItem(f"Cofiring {num}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)           
            self.cofiring_list.addItem(item)
        self.cofiring_list.itemChanged.connect(lambda: self.update_cofiring_window(reset_list=False))

        self.cofiring_individual_cell_list.clear()
        cells_to_visualize = self.list_global_which_cells.currentText()
        cells = self.session.get_cell_ids(cells_to_visualize, verified=True)
        for cell in cells:
            item = QListWidgetItem(f"Cell {cell}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)           
            self.cofiring_individual_cell_list.addItem(item)
        self.cofiring_individual_cell_list.itemChanged.connect(lambda: self.update_cofiring_window(reset_list=False))



    def show_2D_cofiring(self):
        window_size = int(self.cofiring_window_size.text())
        visualize_cofiring = self.cofiring_chkbox.isChecked()

        cells_for_cofiring = self.list_3D_which_cells.currentText()
        shareA = self.cofiring_shareA_chkbox.isChecked()
        shareB = self.cofiring_shareB_chkbox.isChecked()
        direction = self.cofiring_direction_dropdown.currentText().lower()

        # Get items that are checked
        cofiring_nums = set()
        for i in range(self.cofiring_list.count()):
            item = self.cofiring_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                cofiring_nums.add(int(item.text().split(" ")[1]))

        kwargs = {"nums_to_visualize": cells_for_cofiring, "visualize": visualize_cofiring,
                  "cofiring_nums": cofiring_nums, "shareA": shareA, "shareB": shareB, 
                  "direction": direction}
        
        cofiring_data = self.visualization_3D.extract_cofiring_data(window_size, **kwargs)

        kwargs["cofiring_data"] = cofiring_data
        
        cofiring2d_window = Cofiring2DWidget(self.session, self.name, parent=self, **kwargs)

        if cofiring2d_window.name not in self.windows:
            self.windows[cofiring2d_window.name ] = cofiring2d_window        
            cofiring2d_window.show()



    def changed_3D_data_type(self):
        if self.dropdown_3D_data_types.currentText() == "Transient Count":
            self.chkbox_3D_average.hide()
        else:
            self.chkbox_3D_average.show()

    def changed_3D_function(self):
        if self.dropdown_3D_functions.currentText() == "Raw Visualization":
            self.layout_3D_chkbox_parent.hide()
            self.frame_3D_window_size.hide()
            self.dropdown_3D_data_types.clear()
            self.dropdown_3D_data_types.addItems(["C", "DFF", "Binary Transient"])
        else:
            self.layout_3D_chkbox_parent.show()
            self.frame_3D_window_size.show()
            self.dropdown_3D_data_types.clear()
            self.dropdown_3D_data_types.addItems(["C", "DFF", "Transient Count"])

    def visualize_3D(self):
        visualization_function = self.dropdown_3D_functions.currentText()
        visualization_type = self.dropdown_3D_data_types.currentText()
        scaling = self.slider_3D_scaling.value()
        cells_to_visualize = self.list_3D_which_cells.currentText()
        # Clamp values between 1 and 1000
        smoothing_size = int(self.input_smoothing_size.text())
        smoothing_size = max(1, min(smoothing_size, 1000))
        smoothing_type = self.dropdown_smoothing_type.currentText().lower()
        window_size = int(self.input_3D_window_size.text()) if visualization_function == "Transient Visualization" else 1
        normalize = self.chkbox_3D_normalize.isChecked()
        average = self.chkbox_3D_average.isChecked()
        cumulative = self.chkbox_3D_cumulative.isChecked()

        if visualization_type in ["C", "DFF"]:
            if visualization_function == "Transient Visualization" and cumulative:
                visualization_type = self.dropdown_3D_data_types.currentText() + "_cumulative"
            elif visualization_function == "Transient Visualization":
                visualization_type = self.dropdown_3D_data_types.currentText() + "_transient"
            
        if visualization_type == "Binary Transient":
            visualization_type = "E"

        self.visualization_3D.change_func(base_visualization, data_type=visualization_type, scaling=scaling, cells_to_visualize=cells_to_visualize,
                                          smoothing_size=smoothing_size, smoothing_type=smoothing_type, window_size=window_size, normalize=normalize, average=average, cumulative=cumulative)
        
        self.cofiring_chkbox.setChecked(False)
        self.visualization_3D.remove_cofiring()

    def check_if_results_exist(self):
        idx_to_cells = {"0":"1", "1":"2", "2":"5", "3":"10", "4":"15", "5":"20"}
        experiment = self.cmb_experiment.currentText()
        testing_set = self.cmb_testing_set.currentText()

        observed_signals = []
        idx = 0
        while self.w_signals.getItem(idx,0) is not None:
            item = self.w_signals.getItem(idx,0)
            if isinstance(item, PlotItemEnhanced):
                if item.cell_type == "Standard":
                    observed_signals.append(item.id)

            idx += 1        

        try:
            test_cells = self.test_cells_dict[experiment][testing_set][self.actual_name]

            self.no_cells_to_runs  = {}

            for i, test_cell in enumerate(test_cells):
                for observed_signal in observed_signals:
                    if observed_signal in test_cell:
                        
                        no_cells = i // 5
                        which_run = i % 5

                        if no_cells not in self.no_cells_to_runs:
                            self.no_cells_to_runs[str(no_cells)] = []
                        self.no_cells_to_runs[str(no_cells)].append(str(which_run))

            self.cmb_no_of_cells.clear()
            self.cmb_no_of_cells.addItems([idx_to_cells[key] for key in self.no_cells_to_runs.keys()])
            selection = self.cmb_no_of_cells.currentText()
            self.cmb_which_run.clear()
            self.cmb_which_run.addItems(self.no_cells_to_runs[selection])
            self.cmb_no_of_cells.setEnabled(True)
            self.cmb_which_run.setEnabled(True)
            self.btn_generate_ml_results.setEnabled(True)
            

        except:
            self.cmb_no_of_cells.clear()
            self.cmb_no_of_cells.setEnabled(False)
            self.cmb_which_run.clear()
            self.cmb_which_run.setEnabled(False)
            self.btn_generate_ml_results.setEnabled(False)
            return

    def changed_no_cells(self):
        cells_to_idx = {"1":"0", "2":"1", "5":"2", "10":"3", "15":"4", "20":"5"}
        if self.no_cells_to_runs:
            selection = self.cmb_no_of_cells.currentText()
            self.cmb_which_run.clear()
            self.cmb_which_run.addItems(self.no_cells_to_runs[cells_to_idx[selection]])

    def visualize_ml_test_results(self):
        cells_to_idx = {"1":0, "2":1, "5":2, "10":3, "15":4, "20":5}
        experiment = self.cmb_experiment.currentText()
        testing_set = self.cmb_testing_set.currentText()
        no_cells = cells_to_idx[self.cmb_no_of_cells.currentText()]
        which_run = int(self.cmb_which_run.currentText())

        pos = no_cells * 5 + which_run
        test_cells = self.test_cells_dict[experiment][testing_set][self.actual_name][pos]
        test_result = self.test_results[experiment][testing_set]


        observed_signals = []
        idx = 0
        while self.w_signals.getItem(idx,0) is not None:
            item = self.w_signals.getItem(idx,0)
            if isinstance(item, PlotItemEnhanced):
                if item.cell_type == "Standard":
                    observed_signals.append(item.id)

            idx += 1  

        sub_pos = {}

        for id in observed_signals:
            if id in test_cells:
                sub_pos[id] = test_cells.index(id)

        for id, pos in sub_pos.items():
            preds = test_result[2*(no_cells*5+which_run)+1][26999*pos:26999*(pos+1)]
            self.temp_picks[id] = preds


        self.visualize_signals(reset_view=False)


    def run_model(self):
        from ml_training import config
        from ml_training.dataset import extract_data
        from ml_training.ml_util import sequence_to_predictions
        model_path = self.name_to_path.get(self.cmb_model_name.currentText(), None)
        confidence = float(self.model_conf_threshold_input.text()) if self.model_conf_threshold_input.text() else 0.5

        if model_path:
            model = torch.load(model_path, map_location=torch.device(config.DEVICE))
            model.eval()
            with torch.no_grad():
                i = 0
                while self.w_signals.getItem(i,0) is not None:
                    item = self.w_signals.getItem(i,0)
                    if isinstance(item, PlotItemEnhanced):
                        if item.cell_type == "Standard":
                            unit_id = item.id
                            if "hidden" in self.cmb_model_name.currentText():
                                input_data = model.inputs
                                inputs = []
                                for input_type in input_data:
                                    data = self.session.data[input_type].sel(unit_id=unit_id).values
                                    data /= np.max(data)
                                    inputs.append(data)

                                x = np.stack(inputs).T
                                x = torch.as_tensor(x).to(torch.float32).to(config.DEVICE)

                                pred = model(x)
                                pred = torch.sigmoid(pred).cpu().detach().numpy().flatten()
                                pred[pred >= confidence] = 1
                                pred[pred < confidence] = 0

                            else:                                
                                input_data, _ = extract_data(self.session.data, unit_id, model.slack)

                                pred = sequence_to_predictions(model, input_data, config.ROLLING, voting="min")

                                pred[pred >= confidence] = 1
                                pred[pred < confidence] = 0

                            # Prediction made now update the events
                            self.temp_picks[unit_id] = pred
                            
                    i += 1

            self.visualize_signals(reset_view=False)

    def confirm_picks(self):
        """
        Iterate through the current visible plots and save the picks if they are stored in temp
        """
        ["Accept Incoming Only", "Accept Overlapping Only", "Accept All"]
        i = 0
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                if item.id in self.temp_picks and item.cell_type == "Standard":
                    confirmation_type = self.cmb_confirmation_type.currentText()
                    self.session.update_and_save_E(item.id, self.temp_picks[item.id], confirmation_type)
                    del self.temp_picks[item.id]
            i += 1

        self.visualize_signals(reset_view=False)

    def discard_picks(self):
        """
        Discard the picks that are in view
        """
        i = 0
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                if item.id in self.temp_picks and item.cell_type == "Standard":
                    del self.temp_picks[item.id]
            i += 1

        self.visualize_signals(reset_view=False)

    def show_hide_picks(self):
        self.show_temp_picks = not self.show_temp_picks
        self.visualize_signals(reset_view=False)

    def show_metrics(self):
        i = 0
        ids = []
        ground_truths = []
        predictions = []
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                if item.id in self.temp_picks and item.cell_type == "Standard":
                    ids.append(item.id)
                    ground_truths.append(self.session.data['E'].sel(unit_id=item.id).values)
                    predictions.append(self.temp_picks[item.id])
            i += 1
        
        if ids:
            self.metrics_window = MetricsWidget(ids, ground_truths, predictions)
            self.metrics_window.setWindowTitle("Evluation Metrics")
            self.metrics_window.show()


    def selected_event_change(self, ev):
        '''
        This method is not the most efficient but at least I'll have certainty that 
        the selected events are tracked correctly. This should still be quite fast
        regardless, due to looping through only a few plots.

        This method should be called whenever:
        1.) Cell Selection is made.
        2.) Event Selection/Deselection is made.
        3.) When Plots are cleared.
        '''
        # Iterate through all the plots and make sure the number of selected events is 1
        deselected = ev
        selected_event = None
        too_many_selected = False
        i = 0
        plot_count = 0
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                plot_count += 1
                if plot_count > 1:
                    too_many_selected = True
                    break
                if item.selected_events:
                    if len(item.selected_events) > 1 or selected_event is not None:
                        too_many_selected = True
                        break
                    else:
                        selected_event = next(iter(item.selected_events))
                
            i += 1
        """
        Case 1.) No events are selected -> Enable the button and "Force Transient" and clear values.
        Case 2.) One event is selected -> Enable the button, set the values to the selected event and "Adjust Selected Transient"
        Case 3.) More than one event is selected -> Disable the button and set the text to "Too Many Selections"

        There is an optional case that can occur in any of the above cases: if deselected then clear the values
        """
        if plot_count == 0:
            self.btn_force_transient.setEnabled(False)
            self.btn_force_transient.setText("No Plots")
            self.input_force_start.setText("")
            self.input_force_end.setText("")
        if deselected:
            self.input_force_start.setText("")
            self.input_force_end.setText("")
        if selected_event is None and not too_many_selected:
            self.btn_force_transient.setEnabled(True)
            self.btn_force_transient.setText("Force Transient")
            self.input_force_start.setText("")
            self.input_force_end.setText("")
        elif selected_event is not None and not too_many_selected:
            self.btn_force_transient.setEnabled(True)
            self.btn_force_transient.setText("Adjust Transient")
            x = selected_event.xData
            start, end = int(x[0]), int(x[-1])
            self.input_force_start.setText(str(start))
            self.input_force_end.setText(str(end))
        elif too_many_selected:
            self.btn_force_transient.setEnabled(False)
            self.btn_force_transient.setText("Too Many Plots/Selections")



    def force_adjust_transient(self):
        '''
        This is for the two cases where we want to either create a new transient or adjust an existing one.
        '''
        if not self.input_force_start.text() or not self.input_force_end.text():
            print("Invalid Range")
            return
        start = int(self.input_force_start.text())
        end = int(self.input_force_end.text())
        if start >= end:
            print("Invalid Range")
            return
        
        # Clear the events
        item = self.w_signals.getItem(0, 0)
        id = next(iter(self.video_cell_selection))

        if item.selected_events:
            # Remove the currently selected transient from data
            removed_transient = {id: item.clear_selected_events_local()}
            self.session.remove_from_E(removed_transient)
        # For the current selected ranges, add to E the new event and redraw the events
        new_transient = {id: np.arange(start, end+1)}
        self.session.add_to_E(new_transient)


        item.clear_event_curves()

        # Redraw the events
        events = self.session.data['E'].sel(unit_id=id).values
        events = np.nan_to_num(events, nan=0) # Sometimes saving errors can cause NaNs
        indices = events.nonzero()[0]
        if indices.any():
            # Split up the indices into groups
            indices = np.split(indices, np.where(np.diff(indices) != 1)[0]+1)
            # Now Split the indices into pairs of first and last indices
            indices = [(indices_group[0], indices_group[-1]+1) for indices_group in indices]
            item.draw_event_curves(indices)

        self.selected_event_change(False)



    def update_plot_view(self):
        y_start = float(self.view_y_start_input.text())
        y_end = float(self.view_y_end_input.text())
        window = int(self.view_window_input.text())
        i = 0
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                item.getViewBox().setYRange(y_start, y_end, padding=0)
                xs, _ = item.getViewBox().viewRange()
                midpoint = int((xs[0] + xs[1]) / 2)
                x_start = midpoint - int(window / 2)
                x_end = midpoint + int(window / 2)
                item.getViewBox().setXRange(x_start, x_end, padding=0)
            i += 1

    def detect_cell_hover(self, event):
        point = self.imv_cell.getImageItem().mapFromScene(event)
        x, y = point.x(), point.y()
        pos_rounded = (round(y-0.5), round(x-0.5))
        if pos_rounded in self.A_pos_to_cell:
            potential_ids = set(self.A_pos_to_cell[pos_rounded])
            new_ids = potential_ids.difference(self.hovered_cells.keys())
            delete = set(self.hovered_cells.keys()).difference(potential_ids)

            for id in new_ids:
                y, x = self.session.centroids[id]
                text = pg.TextItem(text=str(id), anchor=(0.4,0.4), color=(255, 0, 0, 255))
                self.imv_cell.addItem(text)
                self.hovered_cells[id] = text
                text.setFont(QFont('Times', 7))
                text.setPos(round(x), round(y))
            
            for id in delete:
                self.imv_cell.removeItem(self.hovered_cells[id])
                self.hovered_cells.pop(id)

        else:
            for id in list(self.hovered_cells.keys()):
                text = self.hovered_cells[id]
                self.imv_cell.removeItem(text)
                self.hovered_cells.pop(id)

                


    def keyReleaseEvent(self, event):
        
        action_view = {Qt.Key_A: "start", Qt.Key_F: "end", Qt.Key_D: "next", Qt.Key_S: "prev"}.get(event.key(), None)
        action_trace = {Qt.Key_W: "toggle_dff"}.get(event.key(), None)

        if self.w_signals and action_view is not None:
            i = 0
            while self.w_signals.getItem(i,0) is not None:
                item = self.w_signals.getItem(i,0)
                if isinstance(item, PlotItemEnhanced):
                    if len(item.listDataItems()) > 1: # When it's empty there is only one empty within the list
                        xs, _ = item.getViewBox().viewRange()
                        length = self.session.data["C"].shape[1]
                        window = xs[1] - xs[0]
                        jump = int(window * 0.5)
                        if action_view == "start":
                            item.getViewBox().setXRange(0, window, padding=0)
                        elif action_view == "end":
                            item.getViewBox().setXRange(length - window, length, padding=0)
                        elif action_view == "next":
                            if xs[1] + jump > length:
                                item.getViewBox().setXRange(length - window, length, padding=0)
                            else:
                                item.getViewBox().setXRange(xs[0] + jump, xs[1] + jump, padding=0)
                        elif action_view == "prev":
                            if xs[0] - jump < 0:
                                item.getViewBox().setXRange(0, window, padding=0)
                            else:
                                item.getViewBox().setXRange(xs[0] - jump, xs[1] - jump, padding=0)
                i += 1
        if self.w_signals and action_trace == "toggle_dff":
            self.chkbox_plot_options_dff.setChecked(not self.chkbox_plot_options_dff.isChecked())
            self.visualize_signals(reset_view=False)


    def switched_tabs(self):
        '''
        This function is necessary due to the fact that the video will have different functionality on click depending
        on the tab, missed cells vs. others.
        '''
        if self.tabs_video.currentIndex() != 2:
            self.select_missed_mode = False
            self.btn_missed_select.setText("Enable Select Cell Mode")
            self.imv_cell.getView().setMenuEnabled(True)
            self.imv_cell.getView().setMouseEnabled(x=True, y=True)
            if self.prev_video_tab_idx == 2: # Switching between 0 and 1 should not reset the state
                self.reset_state()
                self.missed_cell_signals_disabled()
                self.video_missed_mask_candidate = np.zeros(self.mask.shape)
                self.missed_cell_indices = set()
                self.current_frame -= 1
                self.next_frame()
        else:
            self.reset_state()
        self.prev_video_tab_idx = self.tabs_video.currentIndex()

    def reset_state(self):
        '''
        Clear selected cells, reset mask, and clear signals.
        '''
        self.video_cell_selection = set()
        self.reset_mask()
        self.w_signals.clear()
        self.selected_events = {}
        self.selected_event_change(True)

    def start_justification(self):
        self.show_justification = True
        self.btn_justification_start.hide()
        self.btn_justification_save.show()
        self.btn_justification_cancel.show()
        self.input_justification.show()
        id = ''.join(filter(str.isdigit, self.list_rejected_cell.selectedItems()[0].text()))
        self.input_justification.setText(self.rejected_justification[id])

    def update_savgol(self, _):
        self.savgol_params["win_len"] = int(self.savgol_win_len_input.text())
        self.savgol_params["poly_order"] = int(self.savgol_poly_order_input.text())
        self.savgol_params["deriv"] = int(self.savgol_deriv_input.text())
        self.savgol_params["delta"] = float(self.savgol_delta_input.text())
        if self.savgol_params["poly_order"] >= self.savgol_params["win_len"]:
            print("Polynomial Order should be less than Window Length")
            self.poly_order_input.setText(str(self.savgol_params["win_len"]-1))
            self.savgol_params["poly_order"] = self.savgol_params["win_len"]-1
        else:
            self.visualize_signals(reset_view=False)
        
    def update_noise(self, _):
        self.noise_params["win_len"] = int(self.noise_win_len_input.text())
        self.noise_params["type"] = self.noise_type_combobox.currentText()
        self.noise_params["cap"] = float(self.noise_cap_input.text())
        self.visualize_signals(reset_view=False)

    def clear_all_events(self):
        # First make sure that the user wants to clear all events
        reply = QMessageBox.question(self, 'Warning!', "Are you sure you want to clear all events from the visible traces?\n This step is not reversible?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            i = 0
            while self.w_signals.getItem(i,0) is not None:
                item = self.w_signals.getItem(i,0)
                if isinstance(item, PlotItemEnhanced):
                    if item.cell_type == "Standard":
                        id = item.id
                        item.clear_event_curves()
                        self.session.clear_E(id)
                i += 1
            self.selected_event_change(False)

    def clear_selected_traces(self):
        # Clear only the selected signals
        i = 0
        to_remove = []
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                if item.selected:
                    to_remove.append((item.cell_type, item.id))
            i += 1

        for cell_type, id in to_remove:
            if cell_type == "Missed":
                self.missed_cells_selection.discard(id)
                self.video_missed_mask -= self.session.data["M"].sel(missed_id=id).values

            else:
                self.video_cell_selection.discard(id)
                self.video_cell_mask -= self.session.data["A"].sel(unit_id=id).values
        
        self.visualize_signals(reset_view=False)
        if not self.btn_play.isChecked():
            self.current_frame -= 1
            self.next_frame()

        self.selected_event_change(False)
        self.visualization_3D.update_selected_cells(self.video_cell_selection)


    def switch_missed_cell_mode(self):
        if self.select_missed_mode:
            self.select_missed_mode = False
            self.btn_missed_select.setText("Enable Select Cell Mode")
            self.imv_cell.getView().setMenuEnabled(True)
            self.imv_cell.getView().setMouseEnabled(x=True, y=True)
            self.missed_cell_indices = set()
            self.video_missed_mask_candidate = np.zeros(self.mask.shape)
            self.current_frame -= 1
            self.next_frame()
            self.missed_cell_signals_disabled()
            self.w_missed_utility.hide()
        else:
            self.select_missed_mode = True
            self.btn_missed_select.setText("Disable Missed Cell Mode")
            self.imv_cell.getView().setMenuEnabled(False)
            self.imv_cell.getView().setMouseEnabled(x=False, y=False)
            self.video_missed_mask_candidate = np.zeros(self.mask.shape)
            self.missed_cell_signals_enabled()
            self.w_missed_utility.show()

    def missed_cell_signals_enabled(self):
        self.missed_cell_signals_disabled()
        self.imv_cell.scene.sigMousePressMove.connect(self.draw_trace)
        self.imv_cell.scene.sigMousePressAltMove.connect(self.remove_trace)
        self.imv_cell.scene.sigMouseRelease.connect(self.finished_trace)

    def missed_cell_signals_disabled(self):
        try:
            self.imv_cell.scene.sigMousePressMove.disconnect(self.draw_trace)
            self.imv_cell.scene.sigMousePressAltMove.disconnect(self.remove_trace)
            self.imv_cell.scene.sigMouseRelease.disconnect(self.finished_trace)
        except:
            pass


    def remove_missed_cells(self):
        # Extract the cell ids but remove non-numeric characters
        cell_ids = []
        for item in self.list_missed_cell.selectedItems():
            id = int(''.join(filter(str.isdigit, item.text())))
            cell_ids.append(id)
        # Deselect the cells and remove the mask
        for id in cell_ids:
            self.missed_cells_selection.discard(id)

        self.mask[self.session.data["M"].sel(missed_id=cell_ids).values.sum(axis=0) > 0] = 3
        self.video_missed_mask  = np.sum(self.session.data["M"].sel(missed_id=list(self.missed_cells_selection)).values, axis=0)
        self.visualize_signals(reset_view=False)
        if not self.btn_play.isChecked():
            self.current_frame -= 1
            self.next_frame()

        self.session.remove_missed(cell_ids)
        self.refresh_missed_list()


    def refresh_missed_list(self):
        if self.session.data["M"] is not None:
            missed_ids = self.session.data["M"].coords["missed_id"].values
            self.list_missed_cell.clear()
            for missed_id in missed_ids:
                self.list_missed_cell.addItem(f"Missing Cell {missed_id}")
        else:
            self.list_missed_cell.clear()


    def missed_cell_init(self):
        if self.session.data["M"] is not None:    
            M = self.session.data["M"].load()
            missed_ids = M.coords["missed_id"].values
            for missed_id in missed_ids:
                indices = np.argwhere(M.sel(missed_id=missed_id).values == 1)
                for pair in indices:
                    if tuple(pair) in self.A_pos_to_missed_cell:
                        self.A_pos_to_missed_cell[tuple(pair)].append(missed_id)
                    else:
                        self.A_pos_to_missed_cell[tuple(pair)] = [missed_id]

            self.refresh_missed_list()

    def clear_selected_pixels(self):
        self.missed_cell_indices = set()
        self.video_missed_mask_candidate = np.zeros(self.mask.shape)
        if not self.btn_play.isChecked():
            self.current_frame -= 1
            self.next_frame()

    def confirm_selected_pixels(self):
        # Extract all the current indices from the mask
        indices = np.argwhere(self.video_missed_mask_candidate == 1)
        id = self.session.add_missed(self.video_missed_mask_candidate)
        for pair in indices:
            if tuple(pair) in self.A_pos_to_missed_cell:
                self.A_pos_to_missed_cell[tuple(pair)].append(id)
            else:
                self.A_pos_to_missed_cell[tuple(pair)] = [id]

        self.refresh_missed_list()
        self.switch_missed_cell_mode()
        self.clear_selected_pixels()

    def save_max_projection(self):
        path, ext = QFileDialog.getSaveFileName(self, "Save Max Projection", "", "PDF (*.pdf);; SVG (*.svg)")
        # Window prompt for paramter toggle
        msg = QMessageBox()
        msg.setText("Display cell ids on the max projection?")
        msg.setWindowTitle("Display Cell IDs")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        ret = msg.exec_()
        display_text = ret == QMessageBox.Yes
        if path:
            max_proj = (self.session.video_data["Y_fm_chk"].max(dim="frame") * self.session.data["A"].sum("unit_id")).values

            contours = []
            indices = []
            unit_ids = self.session.data["unit_ids"]
            for unit_id in unit_ids:
                footprint = (self.session.data["A"].sel(unit_id=unit_id).values)
                thresholded_roi = 1 * footprint > (np.mean(footprint) + 5 * np.std(footprint))
                contours.append(find_contours(thresholded_roi, 0)[0])
                indices.append(np.argwhere(footprint > 0))

            fig, ax = plt.subplots(figsize=(12.8, 9.6))
            ax.imshow(max_proj * 0, cmap='gray')
            for outline, indices, unit_id in zip(contours, indices, unit_ids):
                ax.plot(outline[:, 1], outline[:, 0], color='xkcd:azure', alpha=0.5)
                # Extract the values from max_proj to get the alpha values
                alphas = max_proj[indices[:, 0], indices[:, 1]]
                alphas /= np.max(alphas)
                rgba_colors = np.ones((len(indices), 4))     
                rgba_colors[:, 3] = alphas                  
                ax.scatter(indices[:, 1], indices[:, 0], marker=',', color=rgba_colors, s=(92./fig.dpi)**2, lw=0)
                if display_text:
                    ax.text(np.mean(outline[:, 1]), np.mean(outline[:, 0]), unit_id, color='xkcd:azure',
                        ha='center', va='center', fontsize=4)
            ax.axis('off')
            
            # For some reason the extension is not added by default
            ext = ".pdf" if "pdf" in ext else ".svg"
            if not path.endswith(ext):
                path += ext

            fig.savefig(path)
            

    def generate_gen_stats(self):
        self.gen_stats_window = GeneralStatsWidget(self.session)
        self.gen_stats_window.setWindowTitle("General Statistics")
        self.gen_stats_window.show()

    def find_subplot(self, event):
        if event.double():
            # We need to identify which Plot Item is under the click
            items = self.w_signals.scene().items(event.scenePos())
            plot_item = None
            for item in items:
                if isinstance(item, PlotItemEnhanced):
                    plot_item = item
                    break
            if plot_item is not None and isinstance(event.currentItem, PlotCurveItemEnhanced):
                plot_item.add_point(event)

    def backup_text(self):
        if len(self.list_rejected_cell.selectedItems()) == 1:
            id = ''.join(filter(str.isdigit, self.list_rejected_cell.selectedItems()[0].text()))
            self.rejected_justification[id] = self.input_justification.toPlainText()
            self.session.save_justifications(self.rejected_justification)

    def draw_trace(self, event):
        point = self.imv_cell.getImageItem().mapFromScene(event)
        pos_rounded = (round(point.y()-0.5), round(point.x()-0.5)) # Switch x and y due to transpose

        self.missed_cell_indices.add(pos_rounded)
        x, y = pos_rounded

        self.video_missed_mask_candidate[x, y] = 1
        if not self.btn_play.isChecked():
            self.current_frame -= 1
            self.next_frame()
            

    def remove_trace(self, event):
        point = self.imv_cell.getImageItem().mapFromScene(event)
        x, y = (round(point.y()-0.5), round(point.x()-0.5))
        if self.video_missed_mask_candidate[x, y] == 1:
            self.video_missed_mask_candidate[x, y] = 0
            if not self.btn_play.isChecked():
                self.current_frame -= 1
                self.next_frame()


    def finished_trace(self, event):
        # Once the trace is finished let's check if the area can be filled
        copy_mask = self.video_missed_mask_candidate.copy()
        if self.missed_cell_indices:
            # Get the mean index of the missed cell
            mean = np.mean([[x, y] for x, y in self.missed_cell_indices], axis=0).astype(int)
            # Flood fill
            filled = flood_fill(copy_mask, tuple(mean), 1, connectivity=1)
            # It could be that the filling might not be successful and fill the entire image
            # therefore if the sum exceeds 500 we'll use the original mask.
            if np.sum(filled) < 500:
                self.video_missed_mask_candidate = filled
            
            self.missed_cell_indices = set()
            self.current_frame -= 1
            self.next_frame()

    def point_selection(self, x, y):
        if (x, y) in self.A_pos_to_cell:
            temp_ids = set()
            for cell_id in self.A_pos_to_cell[(x, y)]:
                temp_ids.add(cell_id)

            # We add selected cells and deactivate already selected cells
            self.video_cell_selection = (self.video_cell_selection | temp_ids) - (self.video_cell_selection & temp_ids)
            self.video_cell_mask = np.zeros(self.mask.shape)
            for id in self.video_cell_selection:
                self.video_cell_mask  += self.A[id].values
            self.video_cell_mask[self.video_cell_mask  > 0] = 1
            self.visualize_signals(reset_view=False)
            if not self.btn_play.isChecked():
                self.current_frame -= 1
                self.next_frame()
            
            self.selected_event_change(False)
            self.visualization_3D.update_selected_cells(self.video_cell_selection)
        
        if (x, y) in self.A_pos_to_missed_cell:
            temp_ids = set()
            for missed_id in self.A_pos_to_missed_cell[(x, y)]:
                temp_ids.add(missed_id)

            self.missed_cells_selection = (self.missed_cells_selection | temp_ids) - (self.missed_cells_selection & temp_ids)
            self.video_missed_mask  = np.sum(self.session.data["M"].sel(missed_id=list(self.missed_cells_selection)).values, axis=0)
            if not self.btn_play.isChecked():
                self.current_frame -= 1
                self.next_frame()
            self.visualize_signals(reset_view=False)

    def video_click(self, event):       
        point = self.imv_cell.getImageItem().mapFromScene(event.pos())
        x, y = point.x(), point.y()
        converted_point = (round(y - 0.5), round(x - 0.5)) # Switch x and y due to transpose
        self.point_selection(*converted_point)
        
            
            
    def enable_disable_justification(self, enable=True):
        if len(self.list_rejected_cell.selectedItems()) == 1 and enable:
            self.btn_justification_start.setEnabled(True)
            id = ''.join(filter(str.isdigit, self.list_rejected_cell.selectedItems()[0].text()))
            if id not in self.rejected_justification:
                self.rejected_justification[id] = ""
            else:
                self.input_justification.setText(self.rejected_justification[id])
        else:
            if len(self.list_rejected_cell.selectedItems()) == 1:
                self.btn_justification_start.setEnabled(True)
            else:
                self.btn_justification_start.setEnabled(False)
            self.btn_justification_start.show()
            self.btn_justification_cancel.hide()
            self.btn_justification_save.hide()
            self.input_justification.hide()


    def enable_disable_event_buttons(self):
        if self.chkbox_plot_options_C.isChecked():
            self.frame_algo_events.setEnabled(True)
            self.frame_manual_events.setEnabled(True)
        else:
            self.frame_algo_events.setEnabled(False)
            self.frame_manual_events.setEnabled(False)

    def get_selected_data_type_global(self):
        selected_data_type = []
        if self.chkbox_plot_global_C.isChecked():
            selected_data_type.append('C')
        if self.chkbox_plot_global_S.isChecked():
            selected_data_type.append('S')
        if self.chkbox_plot_global_YrA.isChecked():
            selected_data_type.append('YrA')
        if self.chkbox_plot_global_dff.isChecked():
            selected_data_type.append('DFF')
        
        return selected_data_type

    def get_selected_data_type(self):
        selected_data_type = []
        if self.chkbox_plot_options_C.isChecked():
            selected_data_type.append('C')
        if self.chkbox_plot_options_S.isChecked():
            selected_data_type.append('S')
        if self.chkbox_plot_options_YrA.isChecked():
            selected_data_type.append('YrA')
        if self.chkbox_plot_options_dff.isChecked():
            selected_data_type.append('DFF')
        if self.chkbox_plot_options_savgol.isChecked():
            selected_data_type.append('SavGol')
        if self.chkbox_plot_options_noise.isChecked():
            selected_data_type.append('noise')
        if self.chkbox_plot_options_snr.isChecked():
            selected_data_type.append('SNR')

        return selected_data_type
    
    def get_selected_events(self):
        selected_events = []
        if "RNF" in self.session.data:
            if self.chkbox_plot_options_RNF.isChecked():
                selected_events.append('RNF')
        if "ALP" in self.session.data:
            if self.chkbox_plot_options_ALP.isChecked():
                selected_events.append('ALP')
        if "ILP" in self.session.data:
            if self.chkbox_plot_options_ILP.isChecked():
                selected_events.append('ILP')
        if "ALP_Timeout" in self.session.data:
            if self.chkbox_plot_options_ALP_Timeout.isChecked():
                selected_events.append('ALP_Timeout')

        return selected_events

    def visualize_global_signals(self, reset_view=False):
        self.w_global_signals.clear()
        # Depending on the selected data types we'll visualize the data as averages
        p = PlotItemEnhanced(id="Global", cell_type="Standard")
        p.plotLine.setPos(self.scroll_video.value())
        p.plotLine.sigDragged.connect(self.pause_video)
        p.plotLine.sigPositionChangeFinished.connect(self.update_slider_pos)
        p.setTitle("Averaged Signals")

        self.w_global_signals.addItem(p, row=0, col=0)

        global_window_size = int(self.global_window_size_input.text())
        selected_types = self.get_selected_data_type_global()
        for data_type in selected_types:
            custom_indices = None # Due to Coarsening
            data = self.session.data[data_type]
            cells_to_visualize = self.list_global_which_cells.currentText()
            units = self.session.get_cell_ids(cells_to_visualize)
            data = data.sel(unit_id=units).mean(dim="unit_id")
            if global_window_size > 1:
                if self.global_avg_method.currentText() == "Rolling":
                    data = data.rolling(frame=global_window_size, center=True).mean()
                elif self.global_avg_method.currentText() == "Coarse":
                    data = data.coarsen(frame=global_window_size, boundary="trim").mean()
                    custom_indices = np.arange(global_window_size//2, self.session.data["C"].shape[1]-global_window_size//2, global_window_size)
            
            data = data.values


            global_window_size = int(self.global_window_size_input.text()) if self.global_window_preview_chkbox.isChecked() else 1

            p.add_main_curve(data, custom_indices=custom_indices, is_C=False, pen=self.color_mapping[data_type], window_preview=global_window_size)


    def visualize_signals(self, reset_view=False):
        cell_ids = self.video_cell_selection
        missed_ids = self.missed_cells_selection
        # Before clear the plots and get viewRect
        idx = 0
        views = {"Standard": {}, "Missed": {}} # We'll store the viewRect for each cell type
        if not reset_view:
            i = 0
            while self.w_signals.getItem(i,0) is not None:
                item = self.w_signals.getItem(i,0)
                if isinstance(item, PlotItemEnhanced):
                    # Don't store if there are no plots
                    if len(item.listDataItems()) > 1: # When it's empty there is only one empty within the list
                        views[item.cell_type][item.id] = item.getViewBox().viewRange()
                i += 1
        self.w_signals.clear()   
        self.selected_events = {}            
                    

        
        last_i = 1
        try:
            self.w_signals.scene().sigMouseClicked.disconnect(self.find_subplot)
        except:
            pass
        if cell_ids:
            self.w_signals.scene().sigMouseClicked.connect(self.find_subplot)
            for i, id in enumerate(cell_ids):
                p = PlotItemEnhanced(id=id, cell_type="Standard")
                p.signalChangedSelection.connect(self.selected_event_change)
                p.plotLine.setPos(self.scroll_video.value())
                p.plotLine.sigDragged.connect(self.pause_video)
                p.plotLine.sigPositionChangeFinished.connect(self.update_slider_pos)
                p.setTitle(f"Cell {id}")
                self.w_signals.addItem(p, row=i, col=0)
                selected_types = self.get_selected_data_type()
                for data_type in selected_types:
                    if data_type in self.session.data:
                        data = self.session.data[data_type].sel(unit_id=id).values
                    elif data_type == 'SavGol' or data_type == 'noise' or data_type == 'SNR':
                        data = self.session.get_savgol(id, self.savgol_params)
                        if data_type == 'noise' or data_type == 'SNR':
                            sav_data = data
                            data = self.session.get_noise(sav_data, id, self.noise_params)
                            if data_type == 'SNR':
                                noise = data
                                data = self.session.get_SNR(sav_data, noise)

                    p.add_main_curve(data, is_C=(data_type == 'C'), pen=self.color_mapping[data_type])
                    if 'E' in self.session.data and data_type == 'C':
                        events = self.session.data['E'].sel(unit_id=id).values
                        events = np.nan_to_num(events, nan=0) # Sometimes saving errors can cause NaNs
                        indices = events.nonzero()[0]
                        if indices.any():
                            # Split up the indices into groups
                            indices = np.split(indices, np.where(np.diff(indices) != 1)[0]+1)
                            # Now Split the indices into pairs of first and last indices
                            indices = [(indices_group[0], indices_group[-1]+1) for indices_group in indices]
                            p.draw_event_curves(indices)
                        if id in self.temp_picks and self.show_temp_picks:
                            events_temp = self.temp_picks[id]
                            indices_temp = events_temp.nonzero()[0]
                            if indices_temp.any():
                                indices_temp = np.split(indices_temp, np.where(np.diff(indices_temp) != 1)[0]+1)
                                indices_temp = [(indices_group[0], indices_group[-1]+1) for indices_group in indices_temp]
                                p.draw_temp_curves(indices_temp, indices)


                    selected_events = self.get_selected_events()
                    for event_type in selected_events:
                        if event_type in self.session.data:
                            events = self.session.data[event_type].values
                            # Get indices where == 1
                            indices = np.argwhere(events == 1)
                            p.draw_behavior_events(indices, self.color_mapping[event_type])

                if selected_types and id in views["Standard"]:
                    p.getViewBox().setRange(xRange=views["Standard"][id][0], yRange=views["Standard"][id][1], padding=0)


                last_i += 1

        if missed_ids:
            for i, id in enumerate(missed_ids):
                p = PlotItemEnhanced(id=id, cell_type="Missed")
                p.plotLine.setPos(self.scroll_video.value())
                p.plotLine.sigDragged.connect(self.pause_video)
                p.plotLine.sigPositionChangeFinished.connect(self.update_slider_pos)
                p.setTitle(f"Missed Cell {id}")
                self.w_signals.addItem(p, row=i+last_i, col=0)
                if "YrA" in self.get_selected_data_type():
                    data = self.session.get_missed_signal(id)
                    p.add_main_curve(data)
                if id in views["Missed"]:
                    p.getViewBox().setRange(xRange=views["Missed"][id][0], yRange=views["Missed"][id][1], padding=0)


    def generate_local_stats(self):
        # Iterate through the current plots and generate local statistics windows
        cell_ids = []
        i = 0
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                cell_ids.append(item.id)
            i += 1
        
        for cell_id in cell_ids:
            if cell_id not in self.local_stats_windows:
                self.local_stats_windows[cell_id] = LocalStatsWidget(self.session, cell_id, self)
                self.local_stats_windows[cell_id].setWindowTitle(f"Statistics for Cell {cell_id}")
                self.local_stats_windows[cell_id].show()

    def delete_local_stats_win(self, cell_id):
        del self.local_stats_windows[cell_id]


    def verification_state_changed(self):
        cell_ids = [self.extract_id(item) for item in self.list_cell.selectedItems()]
        self.session.update_verified(cell_ids)
        self.refresh_cell_list()   


    def focus_mask(self):
        
        if self.tabs_video.currentIndex() == 0:            
            cell_ids = [self.extract_id(item) for item in self.list_cell.selectedItems()]
            new_mask = np.zeros(self.mask.shape)
            if cell_ids:
                for cell_id in cell_ids:
                    new_mask += self.A[cell_id].values
            
            new_mask[new_mask > 0] = 1
            new_mask[new_mask == 0] = 3
            self.mask = new_mask
            if not self.btn_play.isChecked():
                self.current_frame -= 1
                self.next_frame()
        elif self.tabs_video.currentIndex() == 2:
            missed_ids = [item.text() for item in self.list_missed_cell.selectedItems()]
            missed_ids = [int(''.join(filter(str.isdigit, id))) for id in missed_ids]
            new_mask = np.zeros(self.mask.shape)
            if missed_ids:
                new_mask += np.sum(self.session.data["M"].sel(missed_id=missed_ids).values, axis=0)
                new_mask[new_mask > 0] = 1
            
            new_mask[new_mask == 0] = 3
            self.mask = new_mask
            if not self.btn_play.isChecked():
                self.current_frame -= 1
                self.next_frame()

    def reset_mask(self):
        self.mask = np.ones(self.mask.shape)
        if not self.btn_play.isChecked():
            self.current_frame -= 1
            self.next_frame()


    def slider_update(self):
        self.current_frame = self.scroll_video.value() - 1
        self.next_frame()

    def clicked_next(self):
        self.pause_video()
        self.next_frame()
    
    def clicked_prev(self):
        self.pause_video()
        self.prev_frame()

    def play_pause(self):
        if self.btn_play.isChecked():
            self.start_video()
        else:
            self.pause_video()            


    def next_frame(self):
        self.current_frame = (1 + self.current_frame) % self.video_length
        self.scroll_video.setValue(self.current_frame)
        image, bimage = self.generate_image()
        if image is not None:
            self.imv_cell.setImage(image, autoRange=False, autoLevels=False)
        if bimage is not None:
            self.imv_behavior.setImage(bimage, autoRange=False, autoLevels=False)
        
        self.video_timer_label.setText(f"{self.session.frame_to_time(self.current_frame)}")
        

    def prev_frame(self):
        self.current_frame = (self.current_frame - 1) % self.video_length
        self.scroll_video.setValue(self.current_frame)
        image, bimage = self.generate_image()
        if image is not None:
            self.imv_cell.setImage(image, autoRange=False, autoLevels=False)
        if bimage is not None:
            self.imv_behavior.setImage(bimage, autoRange=False, autoLevels=False)

    def check_preload_image(self):
        chunk_length = self.current_video.chunks[0][0] * 10
        if self.pre_images is None:
            # Check which chunk the current frame is in
            chunk_idx = self.current_frame // chunk_length
            self.pre_images = self.current_video.sel(frame=slice(chunk_idx*chunk_length, (chunk_idx+1)*chunk_length)).load()

        else:
            frames = self.pre_images.coords["frame"].values
            if frames[0] <= self.current_frame <= frames[-1]:
                return
            else:
                chunk_idx = self.current_frame // chunk_length
                self.pre_images = self.current_video.sel(frame=slice(chunk_idx*chunk_length, (chunk_idx+1)*chunk_length)).load()

    def check_preload_bimage(self, current_frame):
        chunk_length = self.session.video_data["behavior_video"].chunks[0][0]
        if self.pre_bimages is None:
            # Check which chunk the current frame is in
            chunk_idx = current_frame // chunk_length
            self.pre_bimages = self.session.video_data["behavior_video"].sel(frame=slice(chunk_idx*chunk_length, (chunk_idx+1)*chunk_length)).load()
        
        else:
            frames = self.pre_bimages.coords["frame"].values
            if frames[0] <= current_frame <= frames[-1]:
                return
            else:
                chunk_idx = current_frame // chunk_length
                self.pre_bimages = self.session.video_data["behavior_video"].sel(frame=slice(chunk_idx*chunk_length, (chunk_idx+1)*chunk_length)).load()


        

    def generate_image(self):
        image = None
        bimage = None
        if self.chkbox_cell_video.isChecked():
            self.check_preload_image()
            image = self.pre_images.sel(frame=self.current_frame).values // self.mask
            if self.video_cell_selection or self.missed_cells_selection or self.select_missed_mode:
                image = np.stack((image,)*3, axis=-1)
                if self.cmb_cell_highlight_mode.currentText() == "Color":
                    image[:,:,0][self.video_cell_mask == 1] = 0
                    image[:,:,1][self.video_missed_mask == 1] = 0
                elif self.cmb_cell_highlight_mode.currentText() == "Outline":
                    # Use Canny filter to get the edges
                    if self.video_cell_mask.any():
                        edges = canny(self.video_cell_mask, sigma=2)
                        image[edges == 1] = np.array([0, 255, 255])
                    if self.video_missed_mask.any():
                        edges = canny(self.video_missed_mask, sigma=2)
                        image[edges == 1] = np.array([255, 0, 255])
                if self.select_missed_mode:
                    image[:,:,1][self.video_missed_mask_candidate == 1] = 0
        if self.chkbox_behavior_video.isChecked():
            _, bframes, bheight, bwidth = self.session.video_data["behavior_video"].shape
            vframes = self.current_video.shape[0]
            bcurrent_frame = int(self.current_frame * bframes / vframes)
            self.check_preload_bimage(bcurrent_frame)
            bimage = self.pre_bimages.sel(frame=bcurrent_frame).values[0]
        if self.chkbox_3D.isChecked():
            self.visualization_3D.set_frame(self.current_frame)

        return image, bimage
    
    def refresh_image(self):
        image, bimage = self.generate_image()
        if image is not None:
            self.imv_cell.setImage(image, autoRange=False, autoLevels=False)
        if bimage is not None:
            self.imv_behavior.setImage(bimage, autoRange=False, autoLevels=False)

    def refresh_cell_list(self):
        self.list_cell.clear()
        self.list_rejected_cell.clear()
        cell_ids_to_groups = self.session.cell_ids_to_groups
        good_bad_cells = self.session.data['E']['good_cells'].values
        reject_size = 0
        for i, cell_id in enumerate(self.session.data['E']['unit_id'].values):
            if good_bad_cells[i]:
                if cell_id in cell_ids_to_groups:
                    self.list_cell.addItem(f"{cell_id} G{cell_ids_to_groups[cell_id]}")
                else:
                    self.list_cell.addItem(str(cell_id))
                if self.session.data['E']['verified'].loc[{'unit_id': cell_id}].values.item():
                    self.list_cell.item(i-reject_size).setBackground(Qt.green)
            else:
                self.list_rejected_cell.addItem(str(cell_id))
                reject_size += 1

    def extract_id(self, item):
        if "G" in item.text():
            # Remove everything just after " "
            return int(item.text().split(" ")[0])
        else:
            return int(item.text())


    def add_to_group(self):
        cell_ids = [self.extract_id(item) for item in self.list_cell.selectedItems()]

        self.session.add_cell_id_group(cell_ids)
        unique_groups = self.session.get_group_ids()
        self.list_3D_which_cells.clear()
        self.list_3D_which_cells.addItems(["All Cells", "Verified Cells"])
        self.list_3D_which_cells.addItems([f"Group {group_id}" for group_id in unique_groups])
        self.list_global_which_cells.clear()
        self.list_global_which_cells.addItems(["All Cells", "Verified Cells"])
        self.list_global_which_cells.addItems([f"Group {group_id}" for group_id in unique_groups])
        self.refresh_cell_list()
    
    def remove_from_group(self):
        cell_ids = [self.extract_id(item) for item in self.list_cell.selectedItems()]
        self.session.remove_cell_id_group(cell_ids)
        unique_groups = self.session.get_group_ids()
        self.list_3D_which_cells.clear()
        self.list_3D_which_cells.addItems(["All Cells", "Verified Cells"])
        self.list_3D_which_cells.addItems([f"Group {group_id}" for group_id in unique_groups])
        self.list_global_which_cells.clear()
        self.list_global_which_cells.addItems(["All Cells", "Verified Cells"])
        self.list_global_which_cells.addItems([f"Group {group_id}" for group_id in unique_groups])
        self.refresh_cell_list()

    def reject_cells(self):
        cell_ids = [self.extract_id(item) for item in self.list_cell.selectedItems()]
        # Update good_cell list in E values
        self.session.reject_cells(cell_ids)
        self.refresh_cell_list()

    def approve_cells(self):
        cell_ids = [self.extract_id(item) for item in self.list_rejected_cell.selectedItems()]
        self.session.approve_cells(cell_ids)
        self.refresh_cell_list()

        
    def update_slider_pos(self, event):
        self.scroll_video.setValue(round(event.value()))
        self.current_frame = self.scroll_video.value() - 1
        self.next_frame()
    
    def pause_video(self):
        self.video_timer.stop()
        self.btn_play.setIcon(self.style().standardIcon(self.pixmapi_play))
        self.btn_play.setChecked(False)
    
    def start_video(self):
        self.video_timer.start()
        self.btn_play.setIcon(self.style().standardIcon(self.pixmapi_pause))

    def toggle_videos(self):
        if self.chkbox_cell_video.isChecked():
            self.imv_cell.setVisible(True)
        else:
            self.imv_cell.setVisible(False)
        if self.chkbox_behavior_video.isChecked():
            self.imv_behavior.setVisible(True)
        else:
            self.imv_behavior.setVisible(False)
        if self.chkbox_3D.isChecked():
            self.visualization_3D.set_frame(self.current_frame)
            self.visualization_3D.setVisible(True)
        else:
            self.visualization_3D.setVisible(False)
        if not self.chkbox_cell_video.isChecked() and not self.chkbox_behavior_video.isChecked() and not self.chkbox_3D.isChecked():
            self.widget_video_cells.setVisible(False)
        else:
            self.widget_video_cells.setVisible(True)
        self.current_frame -= 1
        self.next_frame()

    def change_cell_video(self, type):
        self.current_video = self.session.video_data[type]
        self.pre_images = None
        for action in self.submenu_videos.actions():
            if action.text() == "&Behavior Video":
                continue
            if action.text() == f"&{self.video_to_title[type]}":
                action.setChecked(True)
            else:
                action.setChecked(False)
        self.imv_cell.setImage(self.current_video.sel(frame=self.current_frame).values, autoRange=False)
        self.current_frame -= 1
        self.next_frame()

    def update_plot_lines(self):
        i = 0
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                item.plotLine.setPos(self.scroll_video.value())
            i += 1

    def closeEvent(self, event):
        windows_to_close = list(self.windows.values())
        for window in windows_to_close:
            window.close()
        self.main_window_ref.remove_window(self.name)
        event.accept()

    def remove_cofire_window(self, name):
        del self.windows[name]
    
    def clear_selected_events(self):
        accumulated_selected_events = {}
        i = 0
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                accumulated_selected_events[item.id] = item.clear_selected_events_local()
            i += 1
        
        self.session.remove_from_E(accumulated_selected_events)

        self.selected_event_change(False)

    def create_event(self):
        accumulated_created_events = {}
        i = 0
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                event_local = item.create_event_local()
                if event_local is not None:
                    accumulated_created_events[item.id] = event_local
            i += 1

        self.session.add_to_E(accumulated_created_events)

    def update_peaks(self):
        min_height = int(self.min_height_input.text()) if self.min_height_input.text() else 0
        distance = float(self.dist_input.text()) if self.dist_input.text() else 10        
        snr_thresh = float(self.snr_input.text()) if self.snr_input.text() else 0
        
        idx = 0
        while self.w_signals.getItem(idx,0) is not None:
            item = self.w_signals.getItem(idx,0)
            if isinstance(item, PlotItemEnhanced):
                C_signal = self.session.data['C'].sel(unit_id=item.id).values
                S_signal = self.session.data['S'].sel(unit_id=item.id).values
                DFF_signal = self.session.data['DFF'].sel(unit_id=item.id).values
                savgol_data = self.session.get_savgol(item.id, self.savgol_params)
                noise_data = self.session.get_noise(savgol_data, item.id, self.noise_params)
                SNR_data = self.session.get_SNR(savgol_data, noise_data)
                peaks, _ = find_peaks(C_signal)
                spikes = []
                final_peaks = []
                # We must now determine when the beginning of the spiking occurs. It must satisfy the following conditions:
                # 1.) Use the C signal to detect all potential peaks.
                # 2.) Going from left to right start evaluating the distance between peaks. If the next peak is close enough and greater than the current, delete the current peak and allocate the S values to the next peak.
                # 3.) Check the AUC of all allocated S values of the observed peak. If its less than a threshold then delete it.
                # 4.) Exclude all peaks whose C value (amplitude) is smaller than a threshold) - I decided to do this last in case we want to allocate the S value to another peak.
                culminated_s_indices = set() # We make use of a set to avoid duplicates
                for i, current_peak in enumerate(peaks):
                    # Look at the next peak and see if it is close enough to the current peak
                    peak_height = DFF_signal[current_peak]
                    # Allocate the overlapping S values to the next peak
                    if S_signal[current_peak] == 0:
                        continue # This indicates no corresponding S signal
                    culminated_s_indices.add(self.get_S_dimensions(S_signal, current_peak))
                    if i < len(peaks) - 1 and DFF_signal[peaks[i+1]] > peak_height:
                        diff = peaks[i+1] - current_peak if self.timestamps is None else self.timestamps[peaks[i+1]] - self.timestamps[current_peak]
                        if diff <= distance:
                            continue

                    # Now check the AUC of the current peak we will use the accumulated S values and also keep track of the earliest
                    # index.
                    beg, end = len(S_signal), 0
                    for beg_temp, end_temp in culminated_s_indices:
                        beg = beg_temp if beg_temp < beg else beg
                        end = end_temp if end_temp > end else end
                    
                    culminated_s_indices = set()

                    if DFF_signal[beg:current_peak+1].max() - DFF_signal[beg:current_peak+1].min() < min_height:
                        continue
                
                    if SNR_data[beg:current_peak+1].max() < snr_thresh:
                        continue



                    # Compensate for the fact that S a frame after the beginning of the spike.
                    beg = max(0, beg-1)

                    spikes.append([beg, current_peak+1])
                    final_peaks.append([current_peak])
                # Convert spikes to a numpy array
                spike_arr = np.zeros(len(C_signal), dtype=int)
                for beg, end in spikes:
                    spike_arr[beg:end] = 1
                self.temp_picks[item.id] = spike_arr
            idx += 1
        
        self.visualize_signals(reset_view=False)




    def get_S_dimensions(self, S_signal, idx):
        '''
        This is a helper function for update_peaks. It returns the beginning and end indices of the S signal for a given peak.
        It will make use of numpy methods to make it quick as possible, as looping through the S signal will be slow.
        '''
        # First get the final index of the S signal we'll assume for the time being that an S signal is no longer than 200 frames.
        # If by a small chance the S signal is longer than 200 frames, then we'll keep doubling the frame length until we find the end of the S signal.
        frame_length = 200
        end = -1
        start = -1
        while end == -1:
            reached_end = False
            if idx + frame_length > len(S_signal):
                frame_length = idx + frame_length - len(S_signal)
                reached_end = True
            values = np.where(S_signal[idx:idx+frame_length] == 0)[0]
            end = idx + values[0] if values.any() else -1
            if end == -1:
                if reached_end:
                    end = len(S_signal)
                else:
                    frame_length *= 2
        
        # Now get the beginning index of the S signal we'll do the same as above except backwards
        frame_length = 200
        while start == -1:
            reached_beg = False
            if idx - frame_length < 0:
                frame_length = idx
                reached_beg = True
            values = np.where(S_signal[idx-frame_length:idx+1][::-1] == 0)[0] # Little hack to reverse it
            start = idx - values[0] + 1 if values.any() else -1
            if start == -1:
                if reached_beg:
                    start = 0
                else:
                    frame_length *= 2
        
        return (start, end)

class PlotItemEnhanced(PlotItem):
    signalChangedSelection = QtCore.Signal(object)
    def __init__(self, **kwargs):
        super(PlotItemEnhanced, self).__init__(**kwargs)
        self.C_signal = None
        self.id = kwargs["id"] if "id" in kwargs else None
        self.cell_type = kwargs["cell_type"] if "cell_type" in kwargs else None
        self.plotLine = InfiniteLine(pos=0, angle=90, pen='g', movable=True)
        self.addItem(self.plotLine)
        self.selected_events = set()
        self.clicked_points = []
        self.selected = False
        

    def clear_event_curves(self):
        for item in self.listDataItems():
            if isinstance(item, PlotCurveItemEnhanced):
                if item.is_event:
                    self.removeItem(item)

    def draw_event_curves(self, spikes):
        for beg, end in spikes:
            event_curve = PlotCurveItemEnhanced(np.arange(beg, end), self.C_signal[beg:end], pen='r', is_event=True, main_plot=self)
            self.addItem(event_curve)
    
    def draw_behavior_events(self, indices, color):
        indices = indices.flatten()
        for idx in indices:
            # draw verical line
            self.addItem(InfiniteLine(pos=idx, angle=90, pen=color))

    def draw_temp_curves(self, spikes, indices=None):
        if not indices:
            for beg, end in spikes:
                event_curve = PlotCurveItemEnhanced(np.arange(beg, end), self.C_signal[beg:end], pen=(255,140,0), is_event=False, main_plot=self)
                self.addItem(event_curve)
        # If indices are provided then overlapping events are colored in red
        else:
            for beg, end in spikes:
                color = (255,140,0)
                for i, (beg_temp, end_temp) in enumerate(indices):
                    if beg_temp <= beg <= end_temp or beg_temp <= end <= end_temp or beg <= beg_temp <= end or beg <= end_temp <= end: # This can be optimized!
                        color = 'g'
                        break
                event_curve = PlotCurveItemEnhanced(np.arange(beg, end), self.C_signal[beg:end], pen=color, is_event=False, main_plot=self)
                self.addItem(event_curve)

    def add_main_curve(self, data, custom_indices=None, is_C=False, pen='w', window_preview=1):
        if is_C:
            self.C_signal = data
        if custom_indices is None:
            curve = PlotCurveItemEnhanced(np.arange(len(data)), data, pen=pen, is_event=False)
        else:
            curve = PlotCurveItemEnhanced(custom_indices, data, pen=pen, is_event=False)
        if window_preview > 1:
            total_frames = len(data) if custom_indices is None else custom_indices[-1]
            # Add vertical lines every window_preview frames in red
            for i in range(0, total_frames + window_preview, window_preview):
                self.addItem(InfiniteLine(pos=i, angle=90, pen='r'))
        self.addItem(curve)

    def clear_selected_events_local(self):
        accumulated_selected_events = np.array([], dtype=int)
        for item in self.selected_events:
            accumulated_selected_events = np.concatenate([accumulated_selected_events, item.xData])
            self.removeItem(item)
        self.selected_events.clear()

        return accumulated_selected_events
        

    def mousePressEvent(self, ev):
        super(PlotItemEnhanced, self).mousePressEvent(ev)
        clicked_items = self.scene().items(ev.scenePos())
        if clicked_items:
            if isinstance(clicked_items[0], QGraphicsTextItem):
                if clicked_items[0].toPlainText() == f"Cell {self.id}" or clicked_items[0].toPlainText() == f"Missed Cell {self.id}":
                    title = clicked_items[0].toPlainText()
                    if not self.selected:
                        self.setTitle(title, color="#0000FF")
                        self.selected = True
                    else:
                        self.setTitle(title, color="#FFFFFF")
                        self.selected = False


    def add_point(self, event):
        # Map to this items coordinates
        point = event.pos()
        x, y = point.x(), point.y()
        # Draw an x at the point
        if len(self.clicked_points) == 2:
            for point in self.clicked_points:
                self.removeItem(point)
            self.clicked_points = []
        
        point = ScatterPlotItem([x], [y], pen='b', symbol='x', size=10)
        self.addItem(point)
        self.clicked_points.append(point)

    def add_selection(self, item):
        self.selected_events.add(item)
        self.signalChangedSelection.emit(False)
    
    def remove_selection(self, item):
        self.selected_events.remove(item)
        self.signalChangedSelection.emit(True)
        
    def create_event_local(self):
        event_curve = None
        if len(self.clicked_points) == 2:
            x1 = self.clicked_points[0].data[0][0]
            x2 = self.clicked_points[1].data[0][0]
            # Flip x1 and x2 if x1 is greater than x2
            if x1 > x2:
                x1, x2 = x2, x1
            # Check if any of the x values have been extended to avoid unnecessary computation
            extended_x1 = False
            extended_x2 = False
            # Now check the other events to see if they overlap
            for item in self.listDataItems():
                if isinstance(item, PlotCurveItemEnhanced):
                    if item.is_event:
                        x1_temp = item.getData()[0].min()
                        x2_temp = item.getData()[0].max()
                        # Check if the two events overlap
                        if x1_temp <= x1 <= x2_temp or x1_temp <= x2 <= x2_temp:
                            # Extend the x values if necessary
                            if not extended_x1:
                                if x1_temp < x1:
                                    x1 = x1_temp
                                    extended_x1 = True
                            if not extended_x2:
                                if x2_temp > x2:
                                    x2 = x2_temp
                                    extended_x2 = True
                            # Remove the event
                            self.removeItem(item)
                            # Remove the event from the selected events
                            self.selected_events.discard(item)
                        # if temp_x1 is greater than x2 then we can break
                        if x1_temp > x2:
                            break
                if extended_x1 and extended_x2:
                    break
            # If either hasn't been extended then we need to find the local minima and maxima
            x1, x2 = round(x1), round(x2)
            if not extended_x1:
                # Just grab the lowest point within 20 frames of x1
                lower_bound = x1 - 20 if x1 - 20 >= 0 else 0
                upper_bound = x1 + 20 if x1 + 20 <= len(self.C_signal) else len(self.C_signal)
                x1 = np.argmin(self.C_signal[lower_bound:upper_bound]) + lower_bound
            if not extended_x2:
                # Just grab the highest point within 20 frames of x2
                lower_bound = x2 - 20 if x2 - 20 >= 0 else 0
                upper_bound = x2 + 20 if x2 + 20 <= len(self.C_signal) else len(self.C_signal)
                x2 = np.argmax(self.C_signal[lower_bound:upper_bound]) + lower_bound
            # x1 can be set a bit too far to the left, push it right until you find a higher value
            while self.C_signal[x1] == self.C_signal[x1+1]:
                x1 += 1
            # Now we can draw the event
            event_curve = PlotCurveItemEnhanced(np.arange(x1, x2+1), self.C_signal[x1:x2+1], pen='r', is_event=True, main_plot=self)
            self.addItem(event_curve)


        for point in self.clicked_points:
            self.removeItem(point)
        self.clicked_points = []

        if event_curve is not None:
            return event_curve.xData
        else:
            return None
        
    




class PlotCurveItemEnhanced(PlotCurveItem):
    def __init__(self, *args, **kwargs):
        super(PlotCurveItemEnhanced, self).__init__(*args, **kwargs)
        self.is_event = kwargs["is_event"] if "is_event" in kwargs else False
        self.main_plot = kwargs["main_plot"] if "main_plot" in kwargs else None
        self.selected = False
        self.setClickable(True)

        if self.is_event:
            self.sigClicked.connect(self.clicked)
    
    def clicked(self, _, event):
        if not event.double():
            self.selected = not self.selected
            if self.selected:
                self.setPen('b')
                self.main_plot.add_selection(self)
            else:
                self.setPen('r')
                self.main_plot.remove_selection(self)