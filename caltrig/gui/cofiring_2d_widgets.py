from PyQt5.QtWidgets import QWidget, QHBoxLayout, QTabWidget
import pyqtgraph as pg
from ..core.backend import DataInstance
from typing import List
import numpy as np
from matplotlib import cm

class Cofiring2DWidget(QWidget):
    def __init__(self, session: DataInstance, name: str, parent=None, nums_to_visualize: str = 'Verified',
                 cofiring_nums: list = None, shareA: bool = False, shareB: bool = False,
                 direction: str = 'bidirectional', cofiring_data: dict = {}, **kwargs):
        super(Cofiring2DWidget, self).__init__()
        self.parent = parent
        self.session = session
        name += f" {nums_to_visualize}"
        name += " ShareA" if shareA else ""
        name += " ShareB" if shareB else ""
        name += f" {direction}"
        
        self.name = name
        self.setWindowTitle(self.name)
        layout_view_2d = QHBoxLayout()
        self.view_2d = pg.GraphicsLayoutWidget()
        self.overlap_2d = pg.GraphicsLayoutWidget()
        tabs = QTabWidget()
        tabs.addTab(self.view_2d, "Cofiring 2D View")
        tabs.addTab(self.overlap_2d, "Overlap Cofiring 2D View")
        layout = QHBoxLayout()
        layout.addWidget(tabs)


        self.dims = self.session.data["A"].shape
        self.max_cofire = max(cofiring_data["cells"].values())
        self.cofiring_data = cofiring_data

        self.setLayout(layout)

        # Populate the 2D View Plot
        self.centroids_max = self.session.centroids_max

        cell_ids = self.session.get_cell_ids(nums_to_visualize)

        centroids = np.array([[self.centroids_max[cell_id][0], self.centroids_max[cell_id][1]] for cell_id in cell_ids])
        data = {"pos": self._convert_coords(centroids)}
        scatter = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(255, 0, 0, 255), symbol="s")
        scatter.addPoints(**data)

        plot = self.view_2d.addPlot()
        plot.addItem(scatter)

        # Now generate the arrows based on the cofiring data
        for cell_id1, cell_id2 in cofiring_data["cells"]:
            self.add_arrow(plot, cell_id1, cell_id2)


        # Populate the Overlap 2D View Plot
        '''
        The idea here is to plot all cells in the center of the plot, and then plot
        all other cells relative to the center cell.
        '''
        plot_overlap = self.overlap_2d.addPlot()
        center = np.array([self.dims[0] // 2, self.dims[1] // 2])
        # Plot the center cell
        center_scatter = pg.ScatterPlotItem(size=15, brush=pg.mkBrush(255, 255, 255, 255), symbol="o")
        center_scatter.addPoints(pos=[center])
        plot_overlap.addItem(center_scatter)

        scatter_points = []
        scatter_brushes = []
        for cell_id1 in cell_ids:
            for cell_id2 in cell_ids:
                if cell_id1 == cell_id2:
                    continue
                # Check if the cells are in the cofiring data
                if (cell_id1, cell_id2) in cofiring_data["cells"]:
                    value = cofiring_data["cells"][(cell_id1, cell_id2)]
                    normalized_value = value / self.max_cofire

                    colormap = cm.get_cmap("rainbow")
                    color = colormap(normalized_value)
                    # Convert to 255
                    color = [int(c * 255) for c in color]
                    # Make it slightly transparent
                    color[-1] = 200

                    # Now we need to see what the translation is from cell_id1 to
                    # the center cell and apply it to the position of cell_id2
                    cell1_pos = self._convert_coords(np.array(self.centroids_max[cell_id1]))
                    translation = center - cell1_pos
                    cell2_pos = self._convert_coords(np.array(self.centroids_max[cell_id2])) + translation

                    scatter_points.append(cell2_pos)
                    scatter_brushes.append(pg.mkBrush(*color))
                    
        overlap_scatter = pg.ScatterPlotItem(size=10, symbol="o")
        overlap_scatter.addPoints(pos=scatter_points, brush=scatter_brushes)
        plot_overlap.addItem(overlap_scatter)


    def _convert_coords(self, data):
        """
        We need to convert the data to be compatible with the rest of the visualization
        """
        if len(data.shape) == 1:
            data = np.flip(data)
            data[1] = self.dims[1] - data[1]
        else:    
            # Flip x and y
            data = np.flip(data, axis=1)
            # Flip y
            data[:, 1] = self.dims[1] - data[:, 1]
        return data

    def add_arrow(self, plot, cell_id1, cell_id2):
        """
        This doesn't work too well, we'll instead plot lines between the centroids
        """
        if (cell_id1, cell_id2) not in self.cofiring_data["cells"]:
            return
        cofire_value = self.cofiring_data["cells"][(cell_id1, cell_id2)]
        if cofire_value == 0:
            return
        a = np.array([self.centroids_max[cell_id1][0], self.centroids_max[cell_id1][1]])
        b = np.array([self.centroids_max[cell_id2][0], self.centroids_max[cell_id2][1]])
        # Convert coordinates
        a = self._convert_coords(a)
        b = self._convert_coords(b)

        # Normalize the cofire_value to a range [0, 1] for color mapping
        normalized_value = cofire_value / self.max_cofire

        # Use a colormap to get the color based on the normalized value
        colormap = cm.get_cmap("rainbow")
        color = colormap(normalized_value)
        # Convert to 255
        color = [int(c * 255) for c in color]
        width = 10 * normalized_value
        # Add the line
        plot.plot([a[0], b[0]], [a[1], b[1]], pen=pg.mkPen(color, width=width))

        # Add just the arrow head 3/4 of the way pointing towards b
        direction = b - a
        arrow_pos = a + 0.75 * direction
        angle = 180 + np.degrees(np.arctan2(direction[1], direction[0]))
        headLen = 1 + (7 * normalized_value)
        arrow = pg.ArrowItem(pos=arrow_pos, angle=angle, headLen=headLen,
                              tipAngle=30, baseAngle=20, brush=color, pxMode=False)
        plot.addItem(arrow)
        

    def closeEvent(self, event):
        self.parent.remove_cofire_window(self.name)
        event.accept()
        