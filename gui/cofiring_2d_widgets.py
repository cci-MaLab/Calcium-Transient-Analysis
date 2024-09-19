from PyQt5.QtWidgets import QWidget, QHBoxLayout
import pyqtgraph as pg
from core.backend import DataInstance
from typing import List
import numpy as np

class Cofiring2DWidget(QWidget):
    def __init__(self, session: DataInstance, name: str, parent=None, nums_to_visualize: str = 'Verified',
                 cofiring_nums: list = None, shareA: bool = False, shareB: bool = False,
                 direction: str = 'bidirectional', cofiring_data: List[int] = [], **kwargs):
        super(Cofiring2DWidget, self).__init__(parent)
        self.session = session
        name += f" {nums_to_visualize}"
        name += " ShareA" if shareA else ""
        name += " ShareB" if shareB else ""
        name += f" {direction}"
        
        self.name = name
        self.setWindowTitle(self.name)
        self.widget_2d = pg.GraphicsLayoutWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.widget_2d)

        self.setLayout(layout)

        # Populate the plot
        centroids_max = self.session.centroids_max

        cell_ids = self.session.get_cell_ids(nums_to_visualize)

        centroids = np.array([[self.session.centroids[cell_id][0], self.session.centroids[cell_id][1]] for cell_id in cell_ids])
        data = {"pos": centroids}
        scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0, 255), symbol="s")
        scatter.addPoints(**data)

        plot = self.widget_2d.addPlot()
        plot.addItem(scatter)


