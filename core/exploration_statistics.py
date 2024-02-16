from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QAction, QStyle, 
                            QSlider, QLabel, QListWidget, QAbstractItemView, QLineEdit, QSplitter,
                            QApplication, QStyleFactory, QTableWidgetItem, QTableWidget, QMenuBar)

import xarray as xr


class GeneralStatsWidget(QWidget):
    def __init__(self, session, parent=None):
        super(GeneralStatsWidget, self).__init__(parent)
        '''
        The window will display from the exploration window and will display general statistics
        related to results from a specific mouse.
        '''
        unit_ids = session.data["unit_ids"]

        # Rows will indicate the feature we are interested in and columns indicate the corresponding neuron
        table = QTableWidget(9, len(unit_ids))

        # Menu Bar for Tools
        menu = QMenuBar()
        pixmapi_save = QStyle.StandardPixmap.SP_FileDialogListView
        btn_general_stats = QAction(self.style().standardIcon(pixmapi_save), "&Tools", self)
        btn_general_stats.setStatusTip("Data related utlities")
        btn_general_stats.triggered.connect(self.copy_to_clipboard)
        stats_menu = menu.addMenu("&Copy Data to Clipboard")
        stats_menu.addAction(btn_general_stats)

        layout = QVBoxLayout()
        layout.addWidget(table)
        layout.setMenuBar(menu)
        self.setLayout(layout)

        # Fill out the table with headers
        table.setHorizontalHeaderLabels([f"Neuron ID #{id}"for id in unit_ids])
        table.setVerticalHeaderLabels(["Cell Size(pixel)", "Location (x,y)", "Total Ca2+ transient #", 
                                       "Average Frequency (Hz)", "Average Amplitude (ΔF/F)", "Average Rising (# of frames)",
                                       "Average Rising Time (seconds)", "Average Ca2+ transient-interval (# of frames)", "Average interval (seconds)"])

        # Fill out the table with data

        sizes = session.get_cell_sizes()
        total_transients = session.get_total_transients()
        
        for i, id in enumerate(unit_ids):
            # 1.) Cell Size
            table.setItem(0, i, QTableWidgetItem(str(sizes.sel(unit_id=id).item())))
            # 2.) Location (x,y)
            table.setItem(1, i, QTableWidgetItem(str((round(session.centroids[id][0]),
                                                      round(session.centroids[id][1])))))
            # 3.) Total Ca2+ transient #
            table.setItem(2, i, QTableWidgetItem(str(total_transients.sel(unit_id=id).item())))
        
    def copy_to_clipboard(self):
        '''
        Copy the data from the table to the clipboard
        '''
        pass