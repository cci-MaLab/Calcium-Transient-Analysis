"""
The following file will be used for doing a deeper dive into the selected session
"""
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QAction, QStyle, 
                            QSlider, QLabel, QListWidget, QAbstractItemView, QLineEdit, QSplitter,
                            QApplication, QStyleFactory, QFrame, QTabWidget, QMenuBar, QCheckBox,
                            QTextEdit, QComboBox, QGraphicsTextItem, QMessageBox)
from PyQt5.QtCore import (Qt, QTimer)
from PyQt5 import QtCore
from PyQt5.QtGui import (QIntValidator, QDoubleValidator, QFont)
from pyqtgraph import (PlotItem, PlotCurveItem, ScatterPlotItem)
import pyqtgraph as pg
import numpy as np
from pyqtgraph import InfiniteLine
from scipy.signal import find_peaks
from core.exploration_statistics import (GeneralStatsWidget, LocalStatsWidget)
from core.pyqtgraph_override import ImageViewOverride
from skimage.segmentation import flood_fill
from skimage.feature import canny
import os


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

        # Set up main view
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.imv = ImageViewOverride()

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
        self.imv.setImage(self.current_video.sel(frame=self.current_frame).values)

        # Add Context Menu Action
        self.video_to_title = {"varr": "Original", "Y_fm_chk": "Processed"}
        self.submenu_videos = self.imv.getView().menu.addMenu('&Video Format')
        for type in self.session.video_data.keys():
            if type ==  "Y_hw_chk":
                continue
            button_video_type = QAction(f"&{self.video_to_title[type]}", self.submenu_videos)
            button_video_type.triggered.connect(lambda state, x=type: self.change_video(x))
            button_video_type.setCheckable(True)
            if type == "varr":
                button_video_type.setChecked(True)
            else:
                button_video_type.setChecked(False)
            self.submenu_videos.addAction(button_video_type)   

        # Menu Bar for statistics
        menu = QMenuBar()
        pixmapi_tools = QStyle.StandardPixmap.SP_FileDialogListView
        btn_general_stats = QAction(self.style().standardIcon(pixmapi_tools), "&General Statistics", self)
        btn_general_stats.setStatusTip("Produce General Statistics")
        btn_general_stats.triggered.connect(self.generate_gen_stats)
        stats_menu = menu.addMenu("&Statistics")
        stats_menu.addAction(btn_general_stats)
        

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
        self.list_cell.setMaximumSize(250, 600)
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
        self.btn_cell_reject = QPushButton("Reject Cell(s)")
        self.btn_cell_reject.clicked.connect(self.reject_cells)

        self.tabs_video = QTabWidget()
        self.tabs_video.setFixedWidth(330)
        tabs_signal = QTabWidget()
        tabs_signal.setFixedWidth(320)

        # Rejected Cells
        w_rejected_cell_label = QLabel("Rejected Cells:")
        self.list_rejected_cell = QListWidget()
        self.list_rejected_cell.setMaximumSize(320, 600)
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
        self.list_missed_cell.setMaximumSize(320, 600)
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
        self.min_height_label = QLabel("Height Threshold (ΔF/F)")
        local_stats_label = QLabel("Local Statistics")
        self.min_height_input = QLineEdit()
        self.min_height_input.setValidator(QDoubleValidator(0, 1000, 3))
        self.min_height_input.setText("0")

        self.dist_label = QLabel("Min IEI")
        self.dist_input = QLineEdit()
        self.dist_input.setValidator(QIntValidator(0, 1000))
        self.dist_input.setText("10")

        self.snr_label = QLabel("SNR Threshold")
        self.snr_input = QLineEdit()
        self.snr_input.setValidator(QDoubleValidator(0, 1000, 3))
        self.snr_input.setText("0")

        btn_algo_event = QPushButton("Calculate Events")
        btn_algo_event.clicked.connect(self.update_peaks)
        btn_clear_events = QPushButton("Clear Selected Events")
        btn_clear_events.clicked.connect(self.clear_selected_events)
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

        # Plot Colors
        self.color_mapping = {
            "C": "w",
            "S": "m",
            "YrA": "c",
            "DFF": "y",
            "SavGol": (154,205,50), # Greenish/Yellow
            "noise": (0,191,255), # Deep Sky Blue
            "SNR": (255,105,180) # Hot Pink
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

        # Video interaction elements
        self.imv.scene.sigMouseClicked.connect(self.video_click)
        self.video_cell_selection = set()
        self.video_cell_mask = np.zeros((self.current_video.shape[1], self.current_video.shape[2]))

        # Visualize signals selected in video
        self.w_signals = pg.GraphicsLayoutWidget()
        

        # Layouts
        layout_video_cells = QHBoxLayout()
        layout_video_cells_visualize = QHBoxLayout()

        layout_video_tools = QHBoxLayout()
        layout_video_tools.addWidget(self.btn_backward)
        layout_video_tools.addWidget(self.btn_play)
        layout_video_tools.addWidget(self.btn_forward)
        layout_video_tools.addWidget(self.scroll_video)

        layout_video = QVBoxLayout()
        layout_video.addWidget(self.imv)
        layout_video.addLayout(layout_video_tools)

        layout_highlight_mode = QHBoxLayout()
        layout_highlight_mode.addWidget(cell_highlight_mode_label)
        layout_highlight_mode.addWidget(self.cmb_cell_highlight_mode)

        layout_cells = QVBoxLayout()
        layout_cells.addWidget(w_cell_label)
        layout_cells.addWidget(self.list_cell)
        layout_cells.addWidget(self.btn_cell_focus)
        layout_cells.addWidget(self.btn_cell_reset)
        layout_cells.addLayout(layout_highlight_mode)
        layout_cells.addWidget(self.btn_cell_reject)
        layout_cells.addWidget(btn_verified)
        w_cells = QWidget()
        w_cells.setLayout(layout_cells)


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


        
        layout_auto_ml = QVBoxLayout()
        layout_auto_ml.addWidget(self.frame_algo_events)
        # Auto and ML into one widget if torch exists
        if torch_imported:
            layout_auto_ml.addWidget(frame_ml_events)
        layout_auto_ml.addStretch()
        w_auto_ml = QWidget()
        w_auto_ml.setLayout(layout_auto_ml)


        # Manual Event Generation
        self.frame_manual_events = QFrame()
        self.frame_manual_events.setFrameShape(QFrame.Box)
        self.frame_manual_events.setFrameShadow(QFrame.Raised)
        self.frame_manual_events.setLineWidth(3)
        layout_manual_events = QVBoxLayout(self.frame_manual_events)
        layout_manual_events.addWidget(self.manual_label)
        layout_manual_events.addWidget(btn_create_event)
        layout_manual_events.addWidget(btn_clear_events) 
        

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
        layout_plot_options.addWidget(self.chkbox_plot_options_snr)
        layout_plot_options.addWidget(self.chkbox_plot_options_noise)
        layout_plot_options.addWidget(self.chkbox_plot_options_C)
        layout_plot_options.addWidget(self.chkbox_plot_options_S)
        layout_plot_options.addWidget(self.chkbox_plot_options_YrA)
        layout_plot_options.addWidget(self.chkbox_plot_options_dff)
        layout_plot_options.addWidget(self.chkbox_plot_options_savgol)


        layout_plot_utility.addWidget(tab_params)
        layout_plot_utility.addWidget(frame_plot_options)
        widget_plot_utility = QWidget()
        widget_plot_utility.setLayout(layout_plot_utility)
        widget_plot_utility.setMaximumWidth(320)

        tabs_signal.addTab(widget_plot_utility, "Params")
        tabs_signal.addTab(tab_transient_detection, "Event Detection")
        tabs_signal.addTab(frame_stats, "Local Stats")

        layout_video_cells.addLayout(layout_video)
        layout_video_cells.addWidget(self.tabs_video)
        widget_video_cells = QWidget()
        widget_video_cells.setLayout(layout_video_cells)

        layout_video_cells_visualize.addWidget(widget_video_cells)

        layout_plot_utility = QVBoxLayout()
        layout_plot_utility.addWidget(btn_clear_traces)
        layout_plot_utility.addWidget(btn_clear_events)
        layout_plot_utility.addWidget(tabs_signal)

        layout_plot = QHBoxLayout()
        layout_plot.addWidget(self.w_signals)
        layout_plot.addLayout(layout_plot_utility)
        widget_plot = QWidget()
        widget_plot.setLayout(layout_plot)
        layout_video_cells_visualize.addWidget(widget_plot)
        widget_video_cells_visualize = QWidget()
        widget_video_cells_visualize.setLayout(layout_video_cells_visualize)

        layout = QVBoxLayout()
        main_widget = QSplitter(Qt.Orientation.Vertical)
        main_widget.setFrameShape(QFrame.StyledPanel)
        main_widget.setStyleSheet(            
            "QSplitter::handle{background-color: gray; width: 5px; border: 1px dotted gray}"
        )
        main_widget.addWidget(widget_video_cells)
        main_widget.addWidget(widget_video_cells_visualize)
        layout.addWidget(main_widget)
        layout.setMenuBar(menu)

        self.setLayout(layout)
        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))

        self.video_timer = QTimer()
        self.video_timer.setInterval(50)
        self.video_timer.timeout.connect(self.next_frame)

        self.missed_cell_init()

        self.imv.scene.sigMouseMoved.connect(self.detect_cell_hover)

    def run_model(self):
        from ml_training import config
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
                            
                            Yra_sample = self.session.data["YrA"].sel(unit_id=unit_id).values
                            C_sample = self.session.data["C"].sel(unit_id=unit_id).values
                            DFF_sample = self.session.data["DFF"].sel(unit_id=unit_id).values

                            x = np.stack([Yra_sample, C_sample, DFF_sample]).T
                            x = torch.as_tensor(x).to(torch.float32)

                            pred = model(x)
                            pred = torch.sigmoid(pred).cpu().detach().numpy().flatten() # Lol
                            pred[pred >= confidence] = 1
                            pred[pred < confidence] = 0

                            # Prediction made now update the events
                            self.session.update_and_save_E(unit_id, pred)
                    i += 1

            self.visualize_signals(reset_view=False)

    

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
        pos = self.imv.getImageItem().mapFromScene(event)
        pos_rounded = (round(pos.y()-0.5), round(pos.x()-0.5))
        if pos_rounded in self.A_pos_to_cell:
            potential_ids = set(self.A_pos_to_cell[pos_rounded])
            new_ids = potential_ids.difference(self.hovered_cells.keys())
            delete = set(self.hovered_cells.keys()).difference(potential_ids)

            for id in new_ids:
                y, x = self.session.centroids[id]
                text = pg.TextItem(text=str(id), anchor=(0.4,0.4), color=(255, 0, 0, 255))
                self.imv.addItem(text)
                self.hovered_cells[id] = text
                text.setFont(QFont('Times', 7))
                text.setPos(round(x), round(y))
            
            for id in delete:
                self.imv.removeItem(self.hovered_cells[id])
                self.hovered_cells.pop(id)

        else:
            for id in list(self.hovered_cells.keys()):
                text = self.hovered_cells[id]
                self.imv.removeItem(text)
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
            self.imv.getView().setMenuEnabled(True)
            self.imv.getView().setMouseEnabled(x=True, y=True)
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


    def switch_missed_cell_mode(self):
        if self.select_missed_mode:
            self.select_missed_mode = False
            self.btn_missed_select.setText("Enable Select Cell Mode")
            self.imv.getView().setMenuEnabled(True)
            self.imv.getView().setMouseEnabled(x=True, y=True)
            self.missed_cell_indices = set()
            self.video_missed_mask_candidate = np.zeros(self.mask.shape)
            self.current_frame -= 1
            self.next_frame()
            self.missed_cell_signals_disabled()
            self.w_missed_utility.hide()
        else:
            self.select_missed_mode = True
            self.btn_missed_select.setText("Disable Missed Cell Mode")
            self.imv.getView().setMenuEnabled(False)
            self.imv.getView().setMouseEnabled(x=False, y=False)
            self.video_missed_mask_candidate = np.zeros(self.mask.shape)
            self.missed_cell_signals_enabled()
            self.w_missed_utility.show()

    def missed_cell_signals_enabled(self):
        self.missed_cell_signals_disabled()
        self.imv.scene.sigMousePressMove.connect(self.draw_trace)
        self.imv.scene.sigMousePressAltMove.connect(self.remove_trace)
        self.imv.scene.sigMouseRelease.connect(self.finished_trace)

    def missed_cell_signals_disabled(self):
        try:
            self.imv.scene.sigMousePressMove.disconnect(self.draw_trace)
            self.imv.scene.sigMousePressAltMove.disconnect(self.remove_trace)
            self.imv.scene.sigMouseRelease.disconnect(self.finished_trace)
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
        point = self.imv.getImageItem().mapFromScene(event)
        pos_rounded = (round(point.y()-0.5), round(point.x()-0.5)) # Switch x and y due to transpose

        self.missed_cell_indices.add(pos_rounded)
        x, y = pos_rounded

        self.video_missed_mask_candidate[x, y] = 1
        if not self.btn_play.isChecked():
            self.current_frame -= 1
            self.next_frame()
            

    def remove_trace(self, event):
        point = self.imv.getImageItem().mapFromScene(event)
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

    def video_click(self, event):       
        point = self.imv.getImageItem().mapFromScene(event.pos())
        converted_point = (round(point.y() - 0.5), round(point.x() - 0.5)) # Switch x and y due to transpose
        if converted_point in self.A_pos_to_cell:
            temp_ids = set()
            for cell_id in self.A_pos_to_cell[converted_point]:
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
        
        if converted_point in self.A_pos_to_missed_cell:
            temp_ids = set()
            for missed_id in self.A_pos_to_missed_cell[converted_point]:
                temp_ids.add(missed_id)

            self.missed_cells_selection = (self.missed_cells_selection | temp_ids) - (self.missed_cells_selection & temp_ids)
            self.video_missed_mask  = np.sum(self.session.data["M"].sel(missed_id=list(self.missed_cells_selection)).values, axis=0)
            if not self.btn_play.isChecked():
                self.current_frame -= 1
                self.next_frame()
            self.visualize_signals(reset_view=False)
            
            
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
        cell_ids = [int(item.text()) for item in self.list_cell.selectedItems()]
        self.session.update_verified(cell_ids)
        self.refresh_cell_list()   


    def focus_mask(self):
        
        if self.tabs_video.currentIndex() == 0:            
            cell_ids = [int(item.text()) for item in self.list_cell.selectedItems()]
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
        image = self.generate_image()
        self.imv.setImage(image, autoRange=False, autoLevels=False)

    def prev_frame(self):
        self.current_frame = (self.current_frame - 1) % self.video_length
        self.scroll_video.setValue(self.current_frame)
        image = self.generate_image()
        self.imv.setImage(image, autoRange=False, autoLevels=False)

    def generate_image(self):
        image = self.current_video.sel(frame=self.current_frame).values // self.mask
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
        return image
    
    def refresh_image(self):
        image = self.generate_image()
        self.imv.setImage(image, autoRange=False, autoLevels=False)

    def refresh_cell_list(self):
        self.list_cell.clear()
        self.list_rejected_cell.clear()
        good_bad_cells = self.session.data['E']['good_cells'].values
        reject_size = 0
        for i, cell_id in enumerate(self.session.data['E']['unit_id'].values):
            if good_bad_cells[i]:
                self.list_cell.addItem(str(cell_id))
                if self.session.data['E']['verified'].loc[{'unit_id': cell_id}].values.item():
                    self.list_cell.item(i-reject_size).setBackground(Qt.green)
            else:
                self.list_rejected_cell.addItem(str(cell_id))
                reject_size += 1

    def reject_cells(self):
        cell_ids = [int(item.text()) for item in self.list_cell.selectedItems()]
        # Update good_cell list in E values
        self.session.reject_cells(cell_ids)
        self.refresh_cell_list()

    def approve_cells(self):
        cell_ids = [int(item.text()) for item in self.list_rejected_cell.selectedItems()]
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

    def change_video(self, type):
        self.current_video = self.session.video_data[type]
        for action in self.submenu_videos.actions():
            if action.text() == f"&{self.video_to_title[type]}":
                action.setChecked(True)
            else:
                action.setChecked(False)
        self.imv.setImage(self.current_video.sel(frame=self.current_frame).values, autoRange=False)
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
        super(ExplorationWidget, self).closeEvent(event)
        self.main_window_ref.remove_window(self.name)
    
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
                # Remove events
                item.clear_event_curves()
                # Plot spikes
                item.draw_event_curves(spikes)
                # Save back to E
                self.session.update_and_save_E(item.id, spikes)
                
            idx += 1




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

    def add_main_curve(self, data, is_C=False, pen='w'):
        if is_C:
            self.C_signal = data
        curve = PlotCurveItemEnhanced(np.arange(len(data)), data, pen=pen, is_event=False)
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