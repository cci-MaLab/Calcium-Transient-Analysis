from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QHBoxLayout, QWidget, QPushButton, QAction, QFileDialog,
                            QComboBox, QListWidget, QAbstractItemView, QSplitter, QApplication, QStyleFactory)
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QColor, QBrush, QFont)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph import PlotItem
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtGui import QPixmap
import os


class InspectionWidget(QWidget):
    def __init__(self, session, main_ref, parent=None):
        super().__init__(parent)
        matplotlib.use('Qt5Agg')
        self.session = session
        self.total_cells = len(self.session.clustering_result["all"]["ids"]) - len(self.session.outliers_list)
        self.selected_plot = 0
        self.cell_ids = None
        self.displaying = "None"
        self.current_labels = []
        self.name = f"{session.mouseID} {session.day} {session.session} Inspection"
        self.main_ref = main_ref

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

        self.w_cell_average_button = QPushButton("Visualize Average")
        self.w_cell_average_button.clicked.connect(self.createAverageSignals)

        self.total_cells_label = QLabel(f"Looking at {self.total_cells} out of {self.total_cells} cells")
        self.total_cells_label.setWordWrap(True)

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
        mid_layout.addWidget(self.w_cell_average_button)
        mid_layout.addWidget(self.w_cell_button)
        mid_layout.addWidget(self.w_cell_list)
        mid_layout.addWidget(w_cell_label)
        mid_layout.addWidget(self.total_cells_label)

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
        
        self.total_cells_label.setText(f"Looking at {total_in_group} out of {self.total_cells} cells")

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

    def createAverageSignals(self):
        self.cell_ids = [int(item.text()) for item in self.w_cell_list.selectedItems()]
        self.selected_plot = None

        if self.cell_ids:
            self.w_signals.clear()
            i = 0
            p = MyPlotWidget(id=i)
            self.w_signals.addItem(p, row=i, col=0)
            data = self.session.data['C'].sel(unit_id=self.cell_ids).mean("unit_id")
            p.plot(data)
            p.setTitle(f"Cell Average")

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
        if not self.w_signals.getItem(1,0):
            # Funky stuff to get the PlotItem clicked
            current_height = self.w_signals.range.height()
            click_height = event.scenePos().y()
            self.selected_plot = round(click_height / current_height * (len(self.cell_ids) - 1))
            if self.selected_plot > len(self.cell_ids) - 1:
                self.selected_plot = len(self.cell_ids) - 1


    def closeEvent(self, event):
        super(InspectionWidget, self).closeEvent(event)
        self.main_ref.remove_window(self.name)

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

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)