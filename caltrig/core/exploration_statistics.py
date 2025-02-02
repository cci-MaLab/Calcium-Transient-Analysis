from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QAction, QStyle, QLabel, QComboBox,
                            QApplication, QTableWidgetItem, QTableWidget, QMenuBar, QLineEdit, QPushButton)

from PyQt5.QtGui import (QIntValidator, QDoubleValidator, QColor)
from PyQt5.QtWidgets import QFileDialog

import matplotlib.pyplot as plt
import numpy as np
from ..gui.clustering_inspection_widgets import MplCanvas
import pandas as pd
from scipy.fftpack import fft

class StatsWidget(QWidget):
    def __init__(self, parent=None):
        super(StatsWidget, self).__init__(parent)
        '''
        This is the parent class for the statistics window. It shouldn't be directly called but rather
        the children classes should be called, i.e. GeneralStatsWidget and LocalStatsWidget.
        '''
        self.clipboard = QApplication.clipboard()

        self.menu = QMenuBar()
        pixmapi_tools = QStyle.StandardPixmap.SP_FileDialogListView
        btn_copy = QAction(self.style().standardIcon(pixmapi_tools), "&Copy Data to Clipboard", self)
        btn_copy.setStatusTip("Data related utilities")
        btn_copy.triggered.connect(self.copy_to_clipboard)
        stats_menu = self.menu.addMenu("&Tools")
        stats_menu.addAction(btn_copy)
        self.type = "Cell"

    def pandas_to_table(self):
        '''
        Convert a pandas dataframe to a QTableWidget.
        '''
        self.table.clear()
        self.table.setRowCount(self.pd_table.shape[0])
        self.table.setColumnCount(self.pd_table.shape[1])
        self.table.setHorizontalHeaderLabels(self.pd_table.columns)
        self.table.setVerticalHeaderLabels([f"{self.type} {idx}" for idx in self.pd_table.index])
        for i in range(self.pd_table.shape[0]):
            for j in range(self.pd_table.shape[1]):
                val = self.pd_table.iloc[i, j]
                val = val if not pd.isnull(val) else "N/A"
                self.table.setItem(i, j, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

    def copy_to_clipboard(self):
        '''
        Copy the data from the self.table to the clipboard.
        '''
        text = "\t"
        col_headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
        text += "\t".join(col_headers) + "\n"
        for i in range(self.table.rowCount()):
            row = [self.table.verticalHeaderItem(i).text()]
            for j in range(self.table.columnCount()):
                row.append(self.table.item(i, j).text())
            text += "\t".join(row) + "\n"
        
        self.clipboard.setText(text)

    def finalize_window(self):
        '''
        Last thing to call in init.
        '''
        # Set it up that double clicking on a header will sort the table for the double clicked column
        self.table.horizontalHeader().sectionDoubleClicked.connect(self.sort_column)
        # If double clicking a row, then sort by row index
        self.table.verticalHeader().sectionDoubleClicked.connect(self.sort_row)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.setMenuBar(self.menu)
        self.setLayout(layout)

    def sort_column(self, logical_index):
        '''
        Sort the table by the double clicked column.
        '''
        col_name = self.table.horizontalHeaderItem(logical_index).text()
        # If location then skip
        if col_name == "Location (x,y)":
            return
        # If the column is already sorted, reverse the order
        if self.pd_table[col_name].is_monotonic_increasing:
            self.pd_table = self.pd_table.sort_values(by=col_name, ascending=False)
        else:
            self.pd_table = self.pd_table.sort_values(by=col_name)

        self.pandas_to_table()

    def sort_row(self, logical_index):
        '''
        Sort the table by the double clicked row.
        If already sorted, reverse the order.
        '''
        if self.pd_table.index.is_monotonic_increasing:
            self.pd_table = self.pd_table.sort_index(ascending=False)
        else:
            self.pd_table = self.pd_table.sort_index()

        self.pandas_to_table()
        

class GeneralStatsWidget(StatsWidget):
    def __init__(self, session, parent=None):
        super(GeneralStatsWidget, self).__init__(parent)
        '''
        The window will display from the exploration window and will display general statistics
        related to results from a specific mouse.
        '''

        unit_ids = session.data["unit_ids"]
        self.E = session.data['E']

        # Rows will indicate the feature we are interested in and columns indicate the corresponding cell
        self.table = QTableWidget(len(unit_ids), 11)

        self.pd_table = pd.DataFrame(index=unit_ids, columns=["Cell Size(pixel)", "Location (x,y)", "Total Ca2+ transient #", 
                                       "Average Frequency (Hz)", "Average Peak Amplitude (ΔF/F)", "Average Rising (# of frames)",
                                       "Average Rising Time (seconds)", "Average Ca2+ transient-interval (# of frames)", "Average interval (seconds)",
                                       "Std(ΔF/F)", "MAD(ΔF/F)"])
        
        btn_stats_amp = QAction("&Amplitude Frequency Boxplot", self)
        btn_stats_amp.triggered.connect(self.generate_amp_boxplot)

        btn_stats_iti = QAction("IEI Frequency Boxplot", self)
        btn_stats_iti.triggered.connect(self.generate_iti_boxplot)      
        
        visualization_menu = self.menu.addMenu("&Visualization")
        visualization_menu.addAction(btn_stats_amp)
        visualization_menu.addAction(btn_stats_iti)

        # Fill out the self.table with data
        E = session.data['E']
        sizes = session.get_cell_sizes()
        total_transients = session.get_total_transients()
        timestamps = E.coords["timestamp(ms)"].values
        total_time = timestamps[-1] - timestamps[0]
        average_frequency = total_transients / total_time * 1000
        average_amplitude = session.get_average_peak_dff()
        total_rising_frames = session.get_total_rising_frames()
        average_rising_frames = total_rising_frames / total_transients
        frames_per_second = len(timestamps) / total_time * 1000
        average_rising_time = average_rising_frames / frames_per_second
        transient_frames = session.get_transient_frames()
        std_dff = session.get_std()
        mad_dff = session.get_mad()


        
        for i, id in enumerate(unit_ids):
            # 1.) Cell Size
            self.pd_table.at[id, "Cell Size(pixel)"] = sizes.sel(unit_id=id).item()
            # 2.) Location (x,y)
            self.pd_table.at[id, "Location (x,y)"] = (round(session.centroids[id][0]), round(session.centroids[id][1]))
            # 3.) Total Ca2+ transient #
            self.pd_table.at[id, "Total Ca2+ transient #"] = int(total_transients.sel(unit_id=id).item())

            # 4.) Average Frequency (Hz)list_rejected_cell
            self.pd_table.at[id, "Average Frequency (Hz)"] = round(average_frequency.sel(unit_id=id).item(), 5)

            # 5.) Average Peak Amplitude (ΔF/F)
            if average_amplitude.get(id, None) is None:
                self.pd_table.at[id, "Average Peak Amplitude (ΔF/F)"] = np.NaN
            else:
                self.pd_table.at[id, "Average Peak Amplitude (ΔF/F)"] = str(round(average_amplitude[id], 3))

            # 6.) Average Rising (# of frames)
            if average_rising_frames.sel(unit_id=id).isnull().item():
                self.pd_table.at[id, "Average Rising (# of frames)"] = np.NaN
            else:
                self.pd_table.at[id, "Average Rising (# of frames)"] = str(round(average_rising_frames.sel(unit_id=id).item()))

            # 7.) Average Rising Time (seconds)
            if average_rising_time.sel(unit_id=id).isnull().item():
                self.pd_table.at[id, "Average Rising Time (seconds)"] = np.NaN
            else:
                self.pd_table.at[id, "Average Rising Time (seconds)"] = str(round(average_rising_time.sel(unit_id=id).item(), 3))

            # 8.) Average Ca2+ transient-interval (# of frames)
            self.pd_table.at[id, "Average Ca2+ transient-interval (# of frames)"] = session.get_mean_iei_per_cell(transient_frames, id, total_transients)
                
            
            # 9.) Average interval (seconds)
            self.pd_table.at[id, "Average interval (seconds)"] = session.get_mean_iei_per_cell(transient_frames, id, total_transients, frame_rate=frames_per_second)

            # 10.) Std(ΔF/F)
            self.pd_table.at[id, "Std(ΔF/F)"] = round(std_dff.sel(unit_id=id).item(), 3)

            # 11.) MAD(ΔF/F)
            self.pd_table.at[id, "MAD(ΔF/F)"] = round(mad_dff.sel(unit_id=id).item(), 3)

            # 12.) Verified
            verified_val = ""
            if self.E.sel(unit_id=id).coords["good_cells"].values.item():
                if self.E.sel(unit_id=id).coords["verified"].values.item():
                    verified_val = "Yes"
                else:
                    verified_val = "No"
            else:
                verified_val = "Rejected"
            self.pd_table.at[id, "Verified"] = verified_val

        self.pandas_to_table()
        self.table.resizeColumnsToContents()  

        self.finalize_window()

    def generate_amp_boxplot(self):
        data = self.pd_table["Average Peak Amplitude (ΔF/F)"].dropna()
        if data.empty:
            return
        self.amp_win = GeneralVizWidget(data, "Amplitude")
        self.amp_win.setWindowTitle("Amplitude Box Plot")
        self.amp_win.show()

    def generate_iti_boxplot(self):
        data = self.pd_table["Average interval (seconds)"].dropna()
        if data.empty:
            return
        data = data.drop(data[data == 'N/A'].index).astype(float)
        self.iti_win = GeneralVizWidget(data, "ITI")
        self.iti_win.setWindowTitle("ITI Box Plot")
        self.iti_win.show()

    def pandas_to_table(self):
        super().pandas_to_table()
        # Iterate through the cell header rows and change the background color based on E data
        for i, cell_id in enumerate(self.pd_table.index):
            # Red if the cell is not a good_cell
            if not self.E.sel(unit_id=cell_id).coords["good_cells"].values.item():
                self.table.verticalHeaderItem(i).setBackground(QColor(255, 0, 0, 100))
            # Green if the cell is verified
            elif self.E.sel(unit_id=cell_id).coords["verified"].values.item():
                self.table.verticalHeaderItem(i).setBackground(QColor(0, 255, 0, 100))
            

class GeneralVizWidget(QWidget):
    def __init__(self, data: pd.Series, viz_type: str, parent=None):
        super(GeneralVizWidget, self).__init__(parent)
        '''
        The window will display either ITI or amplitude box plots for all cells.
        '''
        self.visualization = MplCanvas()
        btn_save_fig = QPushButton("Save Figure")
        btn_save_fig.clicked.connect(lambda: save_figure(self.visualization.figure))

        layout = QVBoxLayout()
        layout.addWidget(self.visualization)
        layout.addWidget(btn_save_fig)
        self.setLayout(layout)

        data = data.astype(float)
        bp = data.plot.box(ax=self.visualization.axes)
        bp.set_title(f"{viz_type} Box Plot")
        bp.set_ylabel(f"Seconds" if viz_type == "ITI" else f"(ΔF/F)")
        # Add jitter to the box plot
        for point in data:
            self.visualization.axes.plot([np.random.normal(1, 0.04)], point, 'r.', alpha=0.2)

class MetricsWidget(StatsWidget):
    def __init__(self, ids, ground_truths, predictions):
        super(MetricsWidget, self).__init__()
        # Output TP, FP, TN, FN, Precision, Recall, F1, Accuracy
        self.table = QTableWidget(len(ids), 8)
        self.pd_table = pd.DataFrame(index=ids, columns=["TP", "FP", "TN", "FN", "Precision", "Recall", "F1", "Accuracy"])

        for i, id in enumerate(ids):
            ground_truth = ground_truths[i]
            prediction = predictions[i]

            tp = np.sum(np.logical_and(ground_truth == 1, prediction == 1))
            fp = np.sum(np.logical_and(ground_truth == 0, prediction == 1))
            tn = np.sum(np.logical_and(ground_truth == 0, prediction == 0))
            fn = np.sum(np.logical_and(ground_truth == 1, prediction == 0))
            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0

            self.pd_table.at[id, "TP"] = tp
            self.pd_table.at[id, "FP"] = fp
            self.pd_table.at[id, "TN"] = tn
            self.pd_table.at[id, "FN"] = fn
            self.pd_table.at[id, "Precision"] = round(precision, 3)
            self.pd_table.at[id, "Recall"] = round(recall, 3)
            self.pd_table.at[id, "F1"] = round(f1, 3)
            self.pd_table.at[id, "Accuracy"] = round(accuracy, 3)

        self.pandas_to_table()
        self.table.resizeColumnsToContents()  

        self.finalize_window()

        

class LocalStatsWidget(StatsWidget):
    def __init__(self, session, unit_id, main_win_ref, parent=None):
        super(LocalStatsWidget, self).__init__(parent)
        '''
        The window will display from the exploration window and will display general statistics
        related to results from a specific mouse.
        '''
        self.type = "Transient"
        self.main_window_ref = main_win_ref

        self.unit_id = unit_id

        total_transients = int(session.get_total_transients(unit_id=unit_id).item())

        self.iti_win = None
        self.amp_win = None

        E = session.data['E'].sel(unit_id=unit_id).values
        self.DFF = session.data['DFF'].sel(unit_id=unit_id)
        timestamps = session.data['E'].coords["timestamp(ms)"].values
        self.frames_per_msec = len(timestamps) / (timestamps[-1] - timestamps[0])
        
        transients = E.nonzero()[0]
        if transients.any():
            # Split up the indices into groups
            transients = np.split(transients, np.where(np.diff(transients) != 1)[0]+1)
            # Now Split the indices into pairs of first and last indices
            transients = [(indices_group[0], indices_group[-1]+1) for indices_group in transients]

        # Rows will indicate the feature we are interested in and columns indicate the corresponding cell
        self.table = QTableWidget(total_transients, 10)
        self.pd_table = pd.DataFrame(index=range(1, total_transients+1), columns=["Rising-Start(frames)", "Rising-Stop(frames)", "Total # of Rising Frames",
                                        "Rising-Start(seconds)", "Rising-Stop(seconds)", "Duration (seconds)",
                                        "Interval with Previous Transient (frames)", "Interval with Previous Transient (seconds)",
                                        "Peak Amplitude (ΔF/F)", "Total Amplitude (ΔF/F)"])

        btn_stats_amp = QAction("&Amplitude Frequency Histogram", self)
        btn_stats_amp.triggered.connect(self.generate_amp_histogram)

        btn_stats_iti = QAction("ITI Frequency Histogram", self)
        btn_stats_iti.triggered.connect(self.generate_iti_histogram)

        btn_fft = QAction("FFT Frequency", self)
        btn_fft.triggered.connect(self.generate_fft_frequency)

        visualization_menu = self.menu.addMenu("&Visualization")
        visualization_menu.addAction(btn_stats_amp)
        visualization_menu.addAction(btn_stats_iti)
        visualization_menu.addAction(btn_fft)

        previous_transient = -1
        # Fill out the self.table with data
        self.iti_msec = []
        self.total_amplitude_list = []
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
                self.iti_msec.append(interval_seconds * 1000)
                interval_seconds = str(round(interval_seconds, 3))
                previous_transient = transient[0]+1

            peak_amplitude = self.DFF.sel(frame=slice(rising_start, rising_stop)).max().values.item()
            total_amplitude = self.DFF.sel(frame=slice(rising_start, rising_stop)).sum().values.item()
            self.total_amplitude_list.append(total_amplitude)
            
            self.pd_table.at[i+1, "Rising-Start(frames)"] = rising_start
            self.pd_table.at[i+1, "Rising-Stop(frames)"] = rising_stop
            self.pd_table.at[i+1, "Total # of Rising Frames"] = rising_total_frames
            self.pd_table.at[i+1, "Rising-Start(seconds)"] = round(rising_start_seconds, 3)
            self.pd_table.at[i+1, "Rising-Stop(seconds)"] = round(rising_stop_seconds, 3)
            self.pd_table.at[i+1, "Duration (seconds)"] = round(rising_total_seconds, 3)
            self.pd_table.at[i+1, "Interval with Previous Transient (frames)"] = interval_frames
            self.pd_table.at[i+1, "Interval with Previous Transient (seconds)"] = interval_seconds
            self.pd_table.at[i+1, "Peak Amplitude (ΔF/F)"] = round(peak_amplitude, 3)
            self.pd_table.at[i+1, "Total Amplitude (ΔF/F)"] = round(total_amplitude, 3)

        self.pandas_to_table()
        self.table.resizeColumnsToContents()

        self.finalize_window()

    def generate_iti_histogram(self):
        self.iti_win = LocalITIWidget(self.iti_msec)
        self.iti_win.setWindowTitle(f"ITI Histogram for cell {self.unit_id}")
        self.iti_win.show()

    def generate_amp_histogram(self):
        self.amp_win = LocalAmpWidget(self.total_amplitude_list)
        self.amp_win.setWindowTitle(f"Amplitude Histogram for cell {self.unit_id}")
        self.amp_win.show()

    def generate_fft_frequency(self):
        self.fft_win = localFrequencyWidget(self.DFF.values)
        self.fft_win.show()
        
    def closeEvent(self, event):
        super(LocalStatsWidget, self).closeEvent(event)
        self.main_window_ref.delete_local_stats_win(self.unit_id)     



class LocalITIWidget(QWidget):
    def __init__(self, iti_msec, parent=None):
        super(LocalITIWidget, self).__init__(parent)
        self.iti_msec = iti_msec

        layout_type = QHBoxLayout()
        label_type = QLabel("Select type of visualization:")
        self.dropdown_type = QComboBox()
        self.dropdown_type.addItems(["Histogram", "CDF"])
        self.dropdown_type.setCurrentIndex(0)
        self.dropdown_type.currentIndexChanged.connect(self.enable_disable_input)
        self.dropdown_type.currentIndexChanged.connect(self.update_visualization)
        layout_type.addWidget(label_type)
        layout_type.addWidget(self.dropdown_type)

        layout_bins = QHBoxLayout()
        label_bins = QLabel("Select size of bins (msec):")
        self.input_bins = QLineEdit()
        self.input_bins.setPlaceholderText("100")
        self.input_bins.setValidator(QIntValidator(10, 100000))
        layout_bins.addWidget(label_bins)
        layout_bins.addWidget(self.input_bins)

        btn_visualize = QPushButton("Visualize")
        btn_visualize.clicked.connect(self.update_visualization)
        btn_visualize.setFixedWidth(300)

        self.visualization = MplCanvas()
        btn_save_fig = QPushButton("Save Figure")
        btn_save_fig.clicked.connect(lambda: save_figure(self.visualization.figure))

        layout  = QVBoxLayout()
        layout.addLayout(layout_type)
        layout.addLayout(layout_bins)
        layout.addWidget(btn_visualize)
        layout.addWidget(self.visualization)
        layout.addWidget(btn_save_fig)
        self.setLayout(layout)

        self.update_visualization()

    def update_visualization(self):
        self.visualization.axes.clear()
        if self.dropdown_type.currentText() == "Histogram":
            bin_size = int(self.input_bins.text()) if self.input_bins.text() else 1000

            no_of_bins = int((max(self.iti_msec) - min(self.iti_msec)) / bin_size)

            self.visualization.axes.hist(self.iti_msec, no_of_bins, rwidth=0.8)

            self.visualization.axes.set_xlabel("IEI (msec)")
            self.visualization.axes.set_ylabel("No. of Events")
        
        elif self.dropdown_type.currentText() == "CDF":
            self.visualization.axes.ecdf(self.iti_msec)

            self.visualization.axes.set_xlabel("IEI (msec)")
            self.visualization.axes.set_ylabel("Cumulative Fraction")

        self.visualization.draw()

    def enable_disable_input(self):
        if self.dropdown_type.currentText() == "Histogram":
            self.input_bins.setEnabled(True)
        else:
            self.input_bins.setEnabled(False)


class LocalAmpWidget(QWidget):
    def __init__(self, total_amplitude_list, parent=None):
        super(LocalAmpWidget, self).__init__(parent)
        self.total_amplitude_list = total_amplitude_list

        layout_type = QHBoxLayout()
        label_type = QLabel("Select type of visualization:")
        self.dropdown_type = QComboBox()
        self.dropdown_type.addItems(["Histogram", "CDF"])
        self.dropdown_type.setCurrentIndex(0)
        self.dropdown_type.currentIndexChanged.connect(self.enable_disable_input)
        self.dropdown_type.currentIndexChanged.connect(self.update_visualization)
        layout_type.addWidget(label_type)
        layout_type.addWidget(self.dropdown_type)

        layout_bins = QHBoxLayout()
        label_bins = QLabel("Select size of bins (ΔF/F):")
        self.input_bins = QLineEdit()
        self.input_bins.setPlaceholderText("10.0")
        self.input_bins.setValidator(QDoubleValidator(1.0, 100.0, 3))
        layout_bins.addWidget(label_bins)
        layout_bins.addWidget(self.input_bins)

        btn_visualize = QPushButton("Visualize")
        btn_visualize.clicked.connect(self.update_visualization)
        btn_visualize.setFixedWidth(300)

        self.visualization = MplCanvas()
        btn_save_fig = QPushButton("Save Figure")
        btn_save_fig.clicked.connect(lambda: save_figure(self.visualization.figure))

        layout  = QVBoxLayout()
        layout.addLayout(layout_type)
        layout.addLayout(layout_bins)
        layout.addWidget(btn_visualize)
        layout.addWidget(self.visualization)
        layout.addWidget(btn_save_fig)
        self.setLayout(layout)

        self.update_visualization()

    def update_visualization(self):
        self.visualization.axes.clear()
        if self.dropdown_type.currentText() == "Histogram":
            bin_size = float(self.input_bins.text()) if self.input_bins.text() else 10.0

            no_of_bins = int((max(self.total_amplitude_list) - min(self.total_amplitude_list)) / bin_size)

            self.visualization.axes.hist(self.total_amplitude_list, no_of_bins, rwidth=0.8)

            self.visualization.axes.set_xlabel("Amplitude (ΔF/F)")
            self.visualization.axes.set_ylabel("No. of Events")
        
        elif self.dropdown_type.currentText() == "CDF":
            self.visualization.axes.ecdf(self.total_amplitude_list)

            self.visualization.axes.set_xlabel("Amplitude (ΔF/F)")
            self.visualization.axes.set_ylabel("Cumulative Fraction")
        
        self.visualization.draw()

    def enable_disable_input(self):
        if self.dropdown_type.currentText() == "Histogram":
            self.input_bins.setEnabled(True)
        else:
            self.input_bins.setEnabled(False)


class localFrequencyWidget(QWidget):
    def __init__(self, DFF, parent=None):
        super(localFrequencyWidget, self).__init__(parent)
        self.setWindowTitle("FFT Frequency")
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.visualization = MplCanvas()
        btn_save_fig = QPushButton("Save Figure")
        btn_save_fig.clicked.connect(lambda: save_figure(self.visualization.figure))
        self.visualization.axes.set_xlabel("Frequency (Hz)")
        self.visualization.axes.set_ylabel("Amplitude")
        layout.addWidget(self.visualization)
        layout.addWidget(btn_save_fig)

        fft_values = fft(DFF)
        fft_values = np.abs(fft_values)
        self.visualization.axes.plot(fft_values)

        
def save_figure(fig):
    # Open file explorer to save the figure
    file_path, ext = QFileDialog.getSaveFileName(None, "Save Figure", "", "SVG Files (*.svg);;PNG Files (*.png);;JPEG Files (*.jpeg);;PDF Files (*.pdf)")
    if not file_path:
        return
    # Append the extension if it is not already there
    ext = "." + ext.split()[0].lower()
    if not file_path.endswith(ext):
        file_path += ext

    fig.savefig(file_path)