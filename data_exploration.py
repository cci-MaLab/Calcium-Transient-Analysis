"""
The following file will be used for doing a deeper dive into the selected session
"""
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QAction, QStyle, 
                            QSlider, QLabel, QListWidget, QAbstractItemView)
from PyQt5.QtCore import (Qt, QTimer)
from pyqtgraph import PlotItem
import pyqtgraph as pg
import numpy as np
from pyqtgraph import InfiniteLine

class ExplorationWidget(QWidget):
    def __init__(self, session, main_window_ref, parent=None):
        super().__init__(parent)
        self.session = session
        self.main_window_ref = main_window_ref

        # Set up main view
        self.imv = pg.ImageView()
        self.videos = self.session.load_videos()
        self.current_video = self.videos["varr"]
        self.video_length = self.current_video.shape[0]
        self.mask = np.ones((self.current_video.shape[1], self.current_video.shape[2]))
        self.current_frame = 0
        self.imv.setImage(self.current_video.sel(frame=self.current_frame).values.T)

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


        # Populate cell list
        for id in self.A.keys():
            self.list_cell.addItem(str(id))

        
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
        layout_video_cells_visualize = QVBoxLayout()

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

        layout_video_cells.addLayout(layout_video)
        layout_video_cells.addLayout(layout_cells)

        layout_video_cells_visualize.addLayout(layout_video_cells)
        layout_video_cells_visualize.addWidget(self.w_signals)

        self.setLayout(layout_video_cells_visualize)

        self.video_timer = QTimer()
        self.video_timer.setInterval(50)
        self.video_timer.timeout.connect(self.next_frame)

    def highlight_cell(self, event):
        point = self.imv.getImageItem().mapFromScene(event.pos())
        converted_point = (round(point.x()), round(point.y()))

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
        if cell_ids:
            for i, id in enumerate(cell_ids):
                # Check if Event data exists
                p = PlotWidgetEnhanced(id=i)
                self.w_signals.addItem(p, row=i, col=0)
                data = self.session.data['C'].sel(unit_id=id)
                # Plot with a thicker line
                p.plot(data)
                p.setTitle(f"Cell {id}")
                if 'E' in self.session.data:
                    events = self.session.data['E'].sel(unit_id=id).values
                    indices = events.nonzero()[0]
                    # Split up the indices into groups
                    indices = np.split(indices, np.where(np.diff(indices) != 1)[0]+1)
                    for indices_group in indices:
                        p.plot(indices_group, data[indices_group], pen='r')



            


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
            if isinstance(item, PlotWidgetEnhanced):
                item.plotLine.setPos(self.scroll_video.value())
            i += 1

    def closeEvent(self, event):
        super(ExplorationWidget, self).closeEvent(event)
        self.main_window_ref.removeWindow(self.name)


class PlotWidgetEnhanced(PlotItem):
    def __init__(self, id=None, **kwargs):
        super(PlotWidgetEnhanced, self).__init__(**kwargs)
        self.id = id
        self.plotLine = InfiniteLine(pos=0, angle=90, pen='r')
        self.addItem(self.plotLine)


