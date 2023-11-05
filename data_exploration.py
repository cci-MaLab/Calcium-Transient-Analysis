"""
The following file will be used for doing a deeper dive into the selected session
"""
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QWidget,
                            QCheckBox, QGridLayout, QFrame, QGraphicsView, QGraphicsScene, QPushButton, 
                            QComboBox, QListWidget, QAbstractItemView, QSplitter, QApplication, QStyleFactory,
                            QAction, QFileDialog)
from PyQt5.QtGui import (QIntValidator, QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont)
from PyQt5.QtCore import (Qt, QTimer)
import pyqtgraph as pg
from pyqtgraph import PlotItem

class ExplorationWidget(QWidget):
    def __init__(self, session, main_win_ref, parent=None):
        super().__init__(parent)
        self.session = session
        self.main_win_ref = main_win_ref

        # Set up main view
        self.imv = pg.ImageView()
        self.videos = self.session.load_videos()
        self.current_video = self.videos["varr"]
        self.video_length = self.current_video.shape[0]
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

        
        button_pdf = QAction("&Play", self.imv.getView().menu)
        button_pdf.triggered.connect(lambda: self.video_timer.start())
        self.imv.getView().menu.addAction(button_pdf)

        
                
        

        # We'll load in a copy of the visualization of the cells we are monitoring
        self.A = self.session.A.copy()
        self.centroids = self.session.centroids
        for outlier in self.session.outliers_list:
            self.A.pop(outlier)

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.imv)
        self.setLayout(layout)

        self.video_timer = QTimer()
        self.video_timer.setInterval(50)
        self.video_timer.timeout.connect(self.next_frame)
        self.video_timer.start()

    def next_frame(self):
        self.current_frame = (1 + self.current_frame) % self.video_length
        self.imv.setImage(self.current_video.sel(frame=self.current_frame).values, autoRange=False) 
    
    def pause_video(self):
        self.video_timer.stop()

    def change_video(self, type):
        self.pause_video()
        self.current_video = self.videos[type]
        for action in self.submenu_videos.actions():
            if action.text() == f"&{type}":
                action.setChecked(True)
            else:
                action.setChecked(False)
        self.imv.setImage(self.current_video.sel(frame=self.current_frame).values, autoRange=False)

    def closeEvent(self, event):
        super(ExplorationWidget, self).closeEvent(event)
        self.main_window_ref.removeWindow(self.name)

