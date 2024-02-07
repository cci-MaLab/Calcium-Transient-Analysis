from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QHBoxLayout, QWidget, QComboBox, QTableWidget, 
                             QTableWidgetItem, QMainWindow)
import pyqtgraph as pg
from core.genetic_algorithm import Genetic_Algorithm  


class GAWindowWidget(QWidget):
    def __init__(self, main_ref: QMainWindow, ga: Genetic_Algorithm, parent=None):
        super().__init__(parent)
        self.main_ref = main_ref
        self.ga = ga
        self.name = "Genetic Algorithm"
        self.length = len(self.ga.examples)
        self.setWindowTitle("Genetic Algorithm Results")
        self.cocaine_view = pg.GraphicsLayoutWidget(title="Cocaine")
        self.saline_view = pg.GraphicsLayoutWidget(title="Saline")

        # Dropdown
        layout_dropdown = QHBoxLayout()
        self.dropdown = QComboBox()
        self.dropdown.addItems([f"Rank {i+1}" for i in range(self.length)])
        self.dropdown.setCurrentIndex(0)
        self.dropdown.currentIndexChanged.connect(self.update)
        layout_dropdown.addWidget(self.dropdown)

        # Trace Visualization
        layout_trace = QHBoxLayout()
        layout_trace.addWidget(self.cocaine_view)
        layout_trace.addWidget(self.saline_view)
        self.update()

        # Table of top 5 values
        layout_table = QVBoxLayout()
        label_table = QLabel(f"Top {self.length} Parameters for {ga.event}")
        table = QTableWidget()
        table.setRowCount(self.length)
        table.setColumnCount(5)
        table.verticalHeader().setVisible(False)
        data = {"Rank": [f"Rank {i+1}" for i in range(self.length)],
                "Fitness": ['{:.4f}'.format(ga._best_fitness[i].item()) for i in range(self.length)],
                "PreBinNum": [str(ga.preBinNum[i].item()) for i in range(self.length)],
                "PostBinNum": [str(ga.postBinNum[i].item()) for i in range(self.length)],
                "Bin Size": [str(ga.binSize[i].item()) for i in range(self.length)]}
        horHeaders = []
        for n, key in enumerate(data.keys()):
            horHeaders.append(key)
            for m, item in enumerate(data[key]):
                newitem = QTableWidgetItem(item)
                table.setItem(m, n, newitem)
        table.setHorizontalHeaderLabels(horHeaders)
        layout_table.addWidget(label_table)
        layout_table.addWidget(table)

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(layout_dropdown)
        layout.addLayout(layout_trace)
        layout.addLayout(layout_table)
        self.setLayout(layout)

    def update(self):
        index = self.dropdown.currentIndex()
        cocaine_traces = self.ga.examples[index]["Cocaine"]
        saline_traces = self.ga.examples[index]["Saline"]

        self.cocaine_view.clear()
        self.saline_view.clear()
        # We'll set a label that spans the middle of the grid
        cocaine_label = pg.LabelItem(justify='center')
        cocaine_label.setText("Cocaine Traces")
        saline_label = pg.LabelItem(justify='center')
        saline_label.setText("Saline Traces")
        self.cocaine_view.addItem(cocaine_label, row=0, col=0)
        self.saline_view.addItem(saline_label, row=0, col=0)
        # In both cases we'll create a 5 by 4 grid
        for i in range(5):
            for j in range(4):
                cocaine_item = self.cocaine_view.addPlot(row=i+1, col=j)
                cocaine_item.plot(self.ga.xvalues[index]["Cocaine"][i*4+j], cocaine_traces[i*4+j])
                saline_item = self.saline_view.addPlot(row=i+1, col=j)
                saline_item.plot(self.ga.xvalues[index]["Saline"][i*4+j], saline_traces[i*4+j])
    
    def closeEvent(self, event):
        super(GAWindowWidget, self).closeEvent(event)
        self.main_ref.remove_window(self.name)