"""
The following file will be used for doing a deeper dive into the selected session
"""
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QAction, QStyle, 
                            QSlider, QLabel, QListWidget, QAbstractItemView, QLineEdit, QSplitter,
                            QApplication, QStyleFactory, QFrame, QTabWidget, QMenuBar)
from PyQt5.QtCore import (Qt, QTimer)
from PyQt5.QtGui import (QIntValidator, QDoubleValidator)
from pyqtgraph import (PlotItem, PlotCurveItem, ScatterPlotItem)
import pyqtgraph as pg
import numpy as np
from pyqtgraph import InfiniteLine
from scipy.signal import find_peaks
from core.exploration_statistics import GeneralStatsWidget

class ExplorationWidget(QWidget):
    def __init__(self, session, name, main_window_ref, timestamps=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.session.check_E()
        self.name = name
        self.main_window_ref = main_window_ref
        self.timestamps = timestamps
        self.gen_stats_window = None

        # Set up main view
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.imv = pg.ImageView()
        self.videos = self.session.load_videos()
        self.current_video = self.videos["varr"]
        self.video_length = self.current_video.shape[0]
        self.mask = np.ones((self.current_video.shape[1], self.current_video.shape[2]))
        self.current_frame = 0
        self.imv.setImage(self.current_video.sel(frame=self.current_frame).values)

        # Add Context Menu Action
        self.submenu_videos = self.imv.getView().menu.addMenu('&Video Format')
        for type in self.videos.keys():
            button_video_type = QAction(f"&{type}", self.submenu_videos)
            button_video_type.triggered.connect(lambda state, x=type: self.change_video(x))
            button_video_type.setCheckable(True)
            if type == "varr":
                button_video_type.setChecked(True)
            else:
                button_video_type.setChecked(False)
            self.submenu_videos.addAction(button_video_type)   

        # Menu Bar for statistics
        menu = QMenuBar()
        pixmapi_save = QStyle.StandardPixmap.SP_FileDialogListView
        btn_general_stats = QAction(self.style().standardIcon(pixmapi_save), "&General Statistics", self)
        btn_general_stats.setStatusTip("Produce General Statistics")
        btn_general_stats.triggered.connect(self.generate_gen_stats)
        stats_menu = menu.addMenu("&Statistics")
        stats_menu.addAction(btn_general_stats)
        

        # We'll load in a copy of the visualization of the cells we are monitoring
        self.A = self.session.A.copy()
        for outlier in self.session.outliers_list:
            self.A.pop(outlier)

        self.A_posToCell = {}
        for cell_id, cell_ROI in self.A.items():
            indices = np.argwhere(cell_ROI.values > 0)
            for pair in indices:
                if tuple(pair) in self.A_posToCell:
                    # We need to switch x and y positions because of how the image is displayed
                    self.A_posToCell[tuple(pair)].append(cell_id)
                else:
                    self.A_posToCell[tuple(pair)] = [cell_id]
                    


        # Select Cells
        w_cell_label = QLabel("Pick which cells to focus (Hold ctrl):")
        self.list_cell = QListWidget()
        self.list_cell.setMaximumSize(250, 600)
        self.list_cell.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.btn_cell_focus = QPushButton("Focus Selection")
        self.btn_cell_focus.clicked.connect(self.focus_mask)
        self.btn_cell_reset = QPushButton("Reset Mask")
        self.btn_cell_reset.clicked.connect(self.reset_mask)
        self.btn_cell_clear_color = QPushButton("Clear Color")
        self.btn_cell_clear_color.setCheckable(True)
        self.btn_cell_clear_color.clicked.connect(self.refresh_image)
        self.btn_cell_reject = QPushButton("Reject Cell")
        self.btn_cell_reject.clicked.connect(self.reject_cells)

        tabs = QTabWidget()
        tabs.setFixedWidth(320)

        # Rejected Cells
        w_rejected_cell_label = QLabel("Rejected Cells:")
        self.list_rejected_cell = QListWidget()
        self.list_rejected_cell.setMaximumSize(250, 600)
        self.list_rejected_cell.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        self.btn_cell_return = QPushButton("Return Cell")
        self.btn_cell_return.clicked.connect(self.approve_cells)

        # Plot utility
        self.auto_label = QLabel("Automatic Transient Detection")
        self.manual_label = QLabel("Manual Transient Detection")
        self.min_height_label = QLabel("Height Threshold")
        self.min_height_input = QLineEdit()
        self.min_height_input.setValidator(QDoubleValidator(0, 1000, 3))
        self.min_height_input.setText("0")

        self.dist_label = QLabel("Min IEI")
        self.dist_input = QLineEdit()
        self.dist_input.setValidator(QIntValidator(0, 1000))
        self.dist_input.setText("10")

        self.auc_label = QLabel("AUC")
        self.auc_input = QLineEdit()
        self.auc_input.setValidator(QDoubleValidator(0, 1000, 3))
        self.auc_input.setText("0")

        btn_algo_event = QPushButton("Calculate Events")
        btn_algo_event.clicked.connect(self.update_peaks)
        btn_clear_events = QPushButton("Clear Selected Events")
        btn_clear_events.clicked.connect(self.clear_selected_events)
        btn_create_event = QPushButton("Create Event")
        btn_create_event.clicked.connect(self.create_event)
        self.btn_verified = QPushButton("Verify/Unverify")
        self.btn_verified.clicked.connect(self.verification_state_changed)
        btn_stats_amp = QPushButton("Amplitude Frequency Histogram")
        btn_stats_amp.clicked.connect(lambda: self.generate_stats(type="amplitude"))
        btn_stats_iei = QPushButton("IEI Frequency Histogram")
        btn_stats_iei.clicked.connect(lambda: self.generate_stats(type="iei"))


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
        self.imv.scene.sigMouseClicked.connect(self.highlight_cell)
        self.video_cell_selection = set()
        self.video_selection_mask = np.zeros((self.current_video.shape[1], self.current_video.shape[2]))

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

        layout_cells = QVBoxLayout()
        layout_cells.addWidget(w_cell_label)
        layout_cells.addWidget(self.list_cell)
        layout_cells.addWidget(self.btn_cell_focus)
        layout_cells.addWidget(self.btn_cell_reset)
        layout_cells.addWidget(self.btn_cell_clear_color)
        layout_cells.addWidget(self.btn_cell_reject)
        layout_cells.addWidget(self.btn_verified)
        w_cells = QWidget()
        w_cells.setLayout(layout_cells)

        layout_rejected_cells = QVBoxLayout()
        layout_rejected_cells.addWidget(w_rejected_cell_label)
        layout_rejected_cells.addWidget(self.list_rejected_cell)
        layout_rejected_cells.addWidget(self.btn_cell_return)
        w_rejected_cells = QWidget()
        w_rejected_cells.setLayout(layout_rejected_cells)

        tabs.addTab(w_cells, "Approved Cells")
        tabs.addTab(w_rejected_cells, "Rejected Cells")


        # General plot utility
        layout_plot_utility = QVBoxLayout()
        layout_plot_utility.addStretch()
        layout_plot_utility.setDirection(3)

        # Event Generation Algorithm
        layout_height = QHBoxLayout()
        layout_height.addWidget(self.min_height_label)
        layout_height.addWidget(self.min_height_input)
        layout_dist = QHBoxLayout()
        layout_dist.addWidget(self.dist_label)
        layout_dist.addWidget(self.dist_input)
        layout_auc = QHBoxLayout()
        layout_auc.addWidget(self.auc_label)
        layout_auc.addWidget(self.auc_input)

        frame_algo_events = QFrame()
        frame_algo_events.setFrameShape(QFrame.Box)
        frame_algo_events.setFrameShadow(QFrame.Raised)
        frame_algo_events.setLineWidth(3)
        layout_algo_events = QVBoxLayout(frame_algo_events)
        layout_algo_events.addWidget(self.auto_label)
        layout_algo_events.addLayout(layout_height)
        layout_algo_events.addLayout(layout_dist)
        layout_algo_events.addLayout(layout_auc)
        layout_algo_events.addWidget(btn_algo_event)

        frame_manual_events = QFrame()
        frame_manual_events.setFrameShape(QFrame.Box)
        frame_manual_events.setFrameShadow(QFrame.Raised)
        frame_manual_events.setLineWidth(3)
        layout_manual_events = QVBoxLayout(frame_manual_events)
        layout_manual_events.addWidget(self.manual_label)
        layout_manual_events.addWidget(btn_create_event)
        layout_manual_events.addWidget(btn_clear_events)

        layout_plot_utility.addWidget(frame_manual_events)
        layout_plot_utility.addWidget(frame_algo_events)
        widget_plot_utility = QWidget()
        widget_plot_utility.setLayout(layout_plot_utility)
        widget_plot_utility.setMaximumWidth(250)

        layout_video_cells.addLayout(layout_video)
        layout_video_cells.addWidget(tabs)
        widget_video_cells = QWidget()
        widget_video_cells.setLayout(layout_video_cells)

        layout_video_cells_visualize.addWidget(widget_video_cells)

        layout_plot = QHBoxLayout()
        layout_plot.addWidget(self.w_signals)
        layout_plot.addWidget(widget_plot_utility)
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
        

    def highlight_cell(self, event):
        point = self.imv.getImageItem().mapFromScene(event.pos())
        converted_point = (round(point.y()), round(point.x())) # Switch x and y due to transpose

        if converted_point in self.A_posToCell:
            temp_ids = set()
            for cell_id in self.A_posToCell[converted_point]:
                temp_ids.add(cell_id)

            # We add selected cells and deactivate already selected cells
            self.video_cell_selection = (self.video_cell_selection | temp_ids) - (self.video_cell_selection & temp_ids)
            self.video_selection_mask = np.zeros(self.mask.shape)
            for id in self.video_cell_selection:
                self.video_selection_mask  += self.A[id].values
            self.video_selection_mask[self.video_selection_mask  > 0] = 1
            self.visualizeSignals(self.video_cell_selection)
            if not self.btn_play.isChecked():
                self.current_frame -= 1
                self.next_frame()



    def visualizeSignals(self, cell_ids):
        self.w_signals.clear()
        try:
            self.w_signals.scene().sigMouseClicked.disconnect(self.find_subplot)
        except:
            pass
        if cell_ids:
            self.w_signals.scene().sigMouseClicked.connect(self.find_subplot)
            for i, id in enumerate(cell_ids):
                data = self.session.data['C'].sel(unit_id=id).values
                p = PlotItemEnhanced(data, id=id)
                self.w_signals.addItem(p, row=i, col=0)
                p.add_main_curve()
                p.plotLine.setPos(self.scroll_video.value())
                p.setTitle(f"Cell {id}")
                if 'E' in self.session.data:
                    events = self.session.data['E'].sel(unit_id=id).values
                    events = np.nan_to_num(events, nan=0) # Sometimes saving errors can cause NaNs
                    indices = events.nonzero()[0]
                    if indices.any():
                        # Split up the indices into groups
                        indices = np.split(indices, np.where(np.diff(indices) != 1)[0]+1)
                        # Now Split the indices into pairs of first and last indices
                        indices = [(indices_group[0], indices_group[-1]+1) for indices_group in indices]
                        p.draw_event_curves(indices)

    def verification_state_changed(self):
        cell_ids = [int(item.text()) for item in self.list_cell.selectedItems()]
        self.session.update_verified(cell_ids)
        self.refresh_cell_list()   


    def focus_mask(self):
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
        if self.video_cell_selection:
            image = np.stack((image,)*3, axis=-1)
            if not self.btn_cell_clear_color.isChecked():
                image[:,:,:2][self.video_selection_mask == 1] = 0
        return image
    
    def refresh_image(self):
        image = self.generate_image()
        self.imv.setImage(image, autoRange=False, autoLevels=False)

    def refresh_cell_list(self):
        self.list_cell.clear()
        self.list_rejected_cell.clear()
        good_bad_cells = self.session.data['E']['good_cells'].values
        for i, cell_id in enumerate(self.session.data['E']['unit_id'].values):
            if good_bad_cells[i]:
                self.list_cell.addItem(str(cell_id))
                if self.session.data['E']['verified'].loc[{'unit_id': cell_id}].values.item():
                    self.list_cell.item(i).setBackground(Qt.green)
            else:
                self.list_rejected_cell.addItem(str(cell_id))

    def reject_cells(self):
        cell_ids = [int(item.text()) for item in self.list_cell.selectedItems()]
        # Update good_cell list in E values
        self.session.reject_cells(cell_ids)
        self.refresh_cell_list()

    def approve_cells(self):
        cell_ids = [int(item.text()) for item in self.list_rejected_cell.selectedItems()]
        self.session.approve_cells(cell_ids)
        self.refresh_cell_list()

    
    def pause_video(self):
        self.video_timer.stop()
        self.btn_play.setIcon(self.style().standardIcon(self.pixmapi_play))
    
    def start_video(self):
        self.video_timer.start()
        self.btn_play.setIcon(self.style().standardIcon(self.pixmapi_pause))

    def change_video(self, type):
        self.current_video = self.videos[type]
        for action in self.submenu_videos.actions():
            if action.text() == f"&{type}":
                action.setChecked(True)
            else:
                action.setChecked(False)
        self.imv.setImage(self.current_video.sel(frame=self.current_frame).values, autoRange=False)

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

    def create_event(self):
        accumulated_created_events = {}
        i = 0
        while self.w_signals.getItem(i,0) is not None:
            item = self.w_signals.getItem(i,0)
            if isinstance(item, PlotItemEnhanced):
                accumulated_created_events[item.id] = item.create_event_local()
            i += 1

        self.session.add_to_E(accumulated_created_events)

    def update_peaks(self):
        min_height = int(self.min_height_input.text()) if self.min_height_input.text() else 0
        distance = float(self.dist_input.text()) if self.dist_input.text() else 10        
        auc = float(self.auc_input.text()) if self.auc_input.text() else 0
        
        idx = 0
        while self.w_signals.getItem(idx,0) is not None:
            item = self.w_signals.getItem(idx,0)
            if isinstance(item, PlotItemEnhanced):
                C_signal = self.session.data['C'].sel(unit_id=item.id).values
                S_signal = self.session.data['S'].sel(unit_id=item.id).values
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
                    peak_height = C_signal[current_peak]
                    # Allocate the overlapping S values to the next peak
                    if S_signal[current_peak] == 0:
                        continue # This indicates no corresponding S signal
                    culminated_s_indices.add(self.get_S_dimensions(S_signal, current_peak))
                    if i < len(peaks) - 1 and C_signal[peaks[i+1]] > peak_height:
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

                    if np.sum(S_signal[beg:end]) < auc:
                        continue

                    if C_signal[beg:current_peak+1].max() - C_signal[beg:current_peak+1].min() < min_height:
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
    def __init__(self, C_signal, **kwargs):
        super(PlotItemEnhanced, self).__init__(**kwargs)
        self.C_signal = C_signal
        self.id = kwargs["id"] if "id" in kwargs else None
        self.plotLine = InfiniteLine(pos=0, angle=90, pen='g')
        self.addItem(self.plotLine)
        self.selected_events = set()
        self.clicked_points = []
        

    def clear_event_curves(self):
        for item in self.listDataItems():
            if isinstance(item, PlotCurveItemEnhanced):
                if item.is_event:
                    self.removeItem(item)

    def draw_event_curves(self, spikes):
        for beg, end in spikes:
            event_curve = PlotCurveItemEnhanced(np.arange(beg, end), self.C_signal[beg:end], pen='r', is_event=True, main_plot=self)
            self.addItem(event_curve)

    def add_main_curve(self):
        main_curve = PlotCurveItemEnhanced(np.arange(len(self.C_signal)), self.C_signal, pen='w', is_event=False)
        self.addItem(main_curve)

    def clear_selected_events_local(self):
        accumulated_selected_events = np.array([], dtype=int)
        for item in self.selected_events:
            accumulated_selected_events = np.concatenate([accumulated_selected_events, item.xData])
            self.removeItem(item)
        self.selected_events.clear()

        return accumulated_selected_events
        
    
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
        
    def create_event_local(self):
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

        return event_curve.xData




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
                self.main_plot.selected_events.add(self)
            else:
                self.setPen('r')
                self.main_plot.selected_events.remove(self)


