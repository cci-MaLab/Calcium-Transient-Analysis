from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QAction, QStyle, QLabel, QComboBox,
                            QApplication, QTableWidgetItem, QTableWidget, QMenuBar, QLineEdit)

from PyQt5.QtGui import QIntValidator

import numpy as np
from gui.clustering_inspection_widgets import MplCanvas


class GeneralStatsWidget(QWidget):
    def __init__(self, session, parent=None):
        super(GeneralStatsWidget, self).__init__(parent)
        '''
        The window will display from the exploration window and will display general statistics
        related to results from a specific mouse.
        '''
        self.clipboard = QApplication.clipboard()

        unit_ids = session.data["unit_ids"]

        # Rows will indicate the feature we are interested in and columns indicate the corresponding neuron
        self.table = QTableWidget(9, len(unit_ids))

        # Menu Bar for Tools
        menu = QMenuBar()
        pixmapi_tools = QStyle.StandardPixmap.SP_FileDialogListView
        btn_copy = QAction(self.style().standardIcon(pixmapi_tools), "&Copy Data to Clipboard", self)
        btn_copy.setStatusTip("Data related utilities")
        btn_copy.triggered.connect(lambda: copy_to_clipboard(self.table, self.clipboard))
        stats_menu = menu.addMenu("&Tools")
        stats_menu.addAction(btn_copy)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.setMenuBar(menu)
        self.setLayout(layout)

        # Fill out the self.table with headers
        self.table.setHorizontalHeaderLabels([f"Neuron ID #{id}"for id in unit_ids])
        self.table.setVerticalHeaderLabels(["Cell Size(pixel)", "Location (x,y)", "Total Ca2+ transient #", 
                                       "Average Frequency (Hz)", "Average Amplitude (ΔF/F)", "Average Rising (# of frames)",
                                       "Average Rising Time (seconds)", "Average Ca2+ transient-interval (# of frames)", "Average interval (seconds)"])

        # Fill out the self.table with data
        E = session.data['E']
        sizes = session.get_cell_sizes()
        total_transients = session.get_total_transients()
        timestamps = E.coords["timestamp(ms)"].values
        total_time = timestamps[-1] - timestamps[0]
        average_frequency = total_transients / total_time * 1000
        average_amplitude = session.get_amplitude_dff() / total_transients
        total_rising_frames = session.get_total_rising_frames()
        average_rising_frames = total_rising_frames / total_transients
        frames_per_second = len(timestamps) / total_time * 1000
        average_rising_time = average_rising_frames / frames_per_second
        transient_frames = session.get_transient_frames()

        
        for i, id in enumerate(unit_ids):
            # 1.) Cell Size
            self.table.setItem(0, i, QTableWidgetItem(str(sizes.sel(unit_id=id).item())))
            # 2.) Location (x,y)
            self.table.setItem(1, i, QTableWidgetItem(str((round(session.centroids[id][0]),
                                                      round(session.centroids[id][1])))))
            # 3.) Total Ca2+ transient #
            self.table.setItem(2, i, QTableWidgetItem(str(int(total_transients.sel(unit_id=id).item()))))

            # 4.) Average Frequency (Hz)
            self.table.setItem(3, i, QTableWidgetItem(str(round(average_frequency.sel(unit_id=id).item(), 5))))

            # 5.) Average Amplitude (ΔF/F)
            # DOUBLE CHECK THIS. For the time being fill with N/A
            if average_amplitude.sel(unit_id=id).isnull().item():
                self.table.setItem(4, i, QTableWidgetItem("N/A"))
            else:
                self.table.setItem(4, i, QTableWidgetItem(str(round(average_amplitude.sel(unit_id=id).item(), 3))))

            # 6.) Average Rising (# of frames)
            if average_rising_frames.sel(unit_id=id).isnull().item():
                self.table.setItem(5, i, QTableWidgetItem("N/A"))
            else:
                self.table.setItem(5, i, QTableWidgetItem(str(round(average_rising_frames.sel(unit_id=id).item()))))

            # 7.) Average Rising Time (seconds)
            if average_rising_time.sel(unit_id=id).isnull().item():
                self.table.setItem(6, i, QTableWidgetItem("N/A"))
            else:
                self.table.setItem(6, i, QTableWidgetItem(str(round(average_rising_time.sel(unit_id=id).item(), 3))))

            # 8.) Average Ca2+ transient-interval (# of frames)
            self.table.setItem(7, i, QTableWidgetItem(session.get_mean_iei_per_cell(transient_frames, id, total_transients)))
                
            
            # 9.) Average interval (seconds)
            self.table.setItem(8, i, QTableWidgetItem(session.get_mean_iei_per_cell(transient_frames, id, total_transients, frame_rate=frames_per_second)))

        self.table.resizeColumnsToContents()   

class LocalStatsWidget(QWidget):
    def __init__(self, session, unit_id, main_win_ref, parent=None):
        super(LocalStatsWidget, self).__init__(parent)
        '''
        The window will display from the exploration window and will display general statistics
        related to results from a specific mouse.
        '''
        self.main_window_ref = main_win_ref

        self.clipboard = QApplication.clipboard()

        self.unit_id = unit_id

        total_transients = int(session.get_total_transients(unit_id=unit_id).item())

        self.iei_win = None

        E = session.data['E'].sel(unit_id=unit_id).values
        C = session.data['C'].sel(unit_id=unit_id)
        timestamps = session.data['E'].coords["timestamp(ms)"].values
        total_time = timestamps[-1] - timestamps[0]
        
        transients = E.nonzero()[0]
        if transients.any():
            # Split up the indices into groups
            transients = np.split(transients, np.where(np.diff(transients) != 1)[0]+1)
            # Now Split the indices into pairs of first and last indices
            transients = [(indices_group[0], indices_group[-1]+1) for indices_group in transients]

        # Rows will indicate the feature we are interested in and columns indicate the corresponding neuron
        self.table = QTableWidget(total_transients, 8)

        # Menu Bar for Tools
        menu = QMenuBar()
        btn_copy = QAction("&Copy Data to Clipboard", self)
        btn_copy.setStatusTip("Data related utilities")
        btn_copy.triggered.connect(lambda: copy_to_clipboard(self.table, self.clipboard))

        btn_stats_amp = QAction("&Amplitude Frequency Histogram", self)

        btn_stats_iei = QAction("IEI Frequency Histogram", self)
        btn_stats_iei.triggered.connect(self.generate_iei_histogram)

        tools_menu = menu.addMenu("&Tools")
        tools_menu.addAction(btn_copy)
        visualization_menu = menu.addMenu("&Visualization")
        visualization_menu.addAction(btn_stats_amp)
        visualization_menu.addAction(btn_stats_iei)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.setMenuBar(menu)
        self.setLayout(layout)

        # Fill out the self.table with headers
        self.table.setVerticalHeaderLabels([f"Transient #{i}"for i in range(1, total_transients+1)])
        self.table.setHorizontalHeaderLabels(["Rising-Start(frames)", "Rising-Stop(frames)", "Total # of Rising Frames", 
                                       "Rising-Start(seconds)", "Rising-Stop(seconds)", "Total # of Rising Frames (seconds)",
                                       "Interval with Previous Transient (frames)", "Interval with Previous Transient (seconds)"])
        previous_transient = -1
        # Fill out the self.table with data
        self.iei_msec = []
        for i, transient in enumerate(transients):
            rising_start = transient[0]+1
            rising_stop = transient[1]+1
            rising_total_frames = rising_stop - rising_start
            rising_start_seconds = timestamps[rising_start-1] / 1000
            rising_stop_seconds = timestamps[rising_stop-1] / 1000
            rising_total_seconds = rising_stop_seconds - rising_start_seconds

            if previous_transient == -1:
                interval_frames = "N/A"
                interval_seconds = "N/A"
                previous_transient = transient[0]+1
            else:
                interval_frames = transient[0]+1 - previous_transient
                interval_seconds = (timestamps[transient[0]] - timestamps[previous_transient]) / 1000
                self.iei_msec.append(interval_seconds * 1000)
                interval_seconds = str(round(interval_seconds, 3))
                previous_transient = transient[0]+1
            
            self.table.setItem(i, 0, QTableWidgetItem(str(rising_start)))
            self.table.setItem(i, 1, QTableWidgetItem(str(rising_stop)))
            self.table.setItem(i, 2, QTableWidgetItem(str(rising_total_frames)))
            self.table.setItem(i, 3, QTableWidgetItem(str(round(rising_start_seconds, 3))))
            self.table.setItem(i, 4, QTableWidgetItem(str(round(rising_stop_seconds, 3))))
            self.table.setItem(i, 5, QTableWidgetItem(str(round(rising_total_seconds, 3))))
            self.table.setItem(i, 6, QTableWidgetItem(str(interval_frames)))
            self.table.setItem(i, 7, QTableWidgetItem(interval_seconds))


        self.table.resizeColumnsToContents()

    def generate_iei_histogram(self):
        pass
        
    def closeEvent(self, event):
        super(LocalStatsWidget, self).closeEvent(event)
        self.main_window_ref.delete_local_stats_win(self.unit_id)     
        
def copy_to_clipboard(table, clipboard):
    '''
    Copy the data from the self.table to the clipboard.
    '''
    text = ""
    col_headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    text += "\t".join(col_headers) + "\n"
    for i in range(table.rowCount()):
        row = [table.verticalHeaderItem(i).text()]
        for j in range(table.columnCount()):
            row.append(table.item(i, j).text())
        text += "\t".join(row) + "\n"
    
    clipboard.setText(text)



class LocalIEIWidget(QWidget):
    def __init__(self, iei_values, parent=None):
        super(LocalIEIWidget, self).__init__(parent)
        self.iei_values = iei_values

        layout_type = QHBoxLayout()
        label_type = QLabel("Select type of visualization:")
        dropdown_type = QComboBox()
        dropdown_type.addItems(["Histogram", "CDF"])
        layout_type.addWidget(label_type)
        layout_type.addWidget(dropdown_type)

        layout_bins = QHBoxLayout()
        label_bins = QLabel("Select size of bins (msec):")
        input_bins = QLineEdit()
        input_bins.setPlaceholderText("100")
        input_bins.setValidator(QIntValidator(1, 10000))
        layout_bins.addWidget(label_bins)
        layout_bins.addWidget(input_bins)

        self.visualization = MplCanvas()

        layout  = QVBoxLayout()
        layout.addLayout(layout_type)
        layout.addLayout(layout_bins)
        layout.addWidget(self.visualization)
