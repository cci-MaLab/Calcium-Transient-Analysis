import numpy as np
from ..gui.sda_widgets import check_cofiring, _precalculate, calculate_fpr, calculate_single_value_windowed_data, add_distance_to_fpr
from ..gui.pop_up_messages import ProgressWindow
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import pandas as pd
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QWidget, QMenuBar, QAction, QStyle, QCheckBox, QVBoxLayout, QApplication)
from PyQt5.QtCore import pyqtSignal

def shuffle_cofiring(session, target_cells, comparison_cells, n=500, seed=None, **kwargs):
    """Shuffle the data, keeping the co-firing structure.

    Parameters
    ----------
    session : Session
        The DataInstance object containing all data for the session.
    target_cells : list of int
        The cell IDs that will be used for calculating the co-firing metric.
    comparison_cells: list of int
        The cell IDs that will be compared to the target cells.
    n : int
        The number of shuffles to perform.
    seed : int
        The random seed to use.

    Returns
    -------
    VisualizeShuffledCofiring
        The PyQt5 window to visualize the shuffled data.
    """

    if seed is not None:
        np.random.seed(seed)
    

    all_cells = np.unique(target_cells + comparison_cells)

    frame_start, itis = session.get_transient_frames_iti_dict(all_cells)
    positions = session.centroids

    # If we are temporally shuffling, we need to permute the ITIs hence
    # we need to omit the first transients from the calculation for 1-to-1 comparison
    omit_first = kwargs['temporal']
    cofiring_distances_original = {}
    # First get the cofiring metric of the original data
    cofiring_original, spatial_original, connections_used_original = calculate_cofiring_for_group(frame_start, positions, target_cells,
                                                                        comparison_cells, cofiring_distances_original,
                                                                        omit_first=omit_first, **kwargs)
    # Set up the PyQt application and progress window
    progress_window = ProgressWindow(total_steps=n)
    progress_window.show()

    shuffled_temporal_cofiring = []
    shuffled_spatial_distances = {}
    for i in range(n):
        progress_window.update_progress(i + 1)
        if kwargs['temporal']:
            shuffled_frame_start = permute_itis_to_start_indices(itis)
        else:
            shuffled_frame_start = frame_start
        if kwargs['spatial']:
            shuffled_spatial = permute_spatial(positions)
        else:
            shuffled_spatial = positions

        # Calculate the cofiring metric for the shuffled data
        cofiring, shuffled_spatial_distances, _ = calculate_cofiring_for_group(shuffled_frame_start, shuffled_spatial,
                                                                    target_cells, comparison_cells, shuffled_spatial_distances,
                                                                    omit_first=False, **kwargs)

        shuffled_temporal_cofiring.append(cofiring)

    progress_window.close()

    visualize_shuffled = VisualizeShuffledCofiring(cofiring_original, shuffled_temporal_cofiring, shuffled_spatial_distances, spatial_original, connections_used_original, temporal=kwargs['temporal'])
    
    return visualize_shuffled


def shuffle_advanced(session, target_cells, comparison_cells, n=100, seed=None, current_window=1, **kwargs):
    """
    Shuffle the data with advanced options.

    Parameters
    ----------
    session : Session
        The DataInstance object containing all data for the session.
    target_cells : list of int
        The cell IDs that will be used for calculating the co-firing metric.
    comparison_cells: list of int
        The cell IDs that will be compared to the target cells.
    n : int
        The number of shuffles to perform.
    seed : int
        The random seed to use.
    
    Returns
    -------
    VisualizeShuffledCofiring
        The PyQt5 window to visualize the shuffled data.
    """
    if seed is not None:
        np.random.seed(seed)
    

    all_cells = np.unique(target_cells + comparison_cells)
    window_size = kwargs['shuffling']['window_size']
    readout = kwargs['shuffling']['readout']
    fpr = kwargs['shuffling']['fpr']
    anchor = kwargs['anchor']

    name = f"Shuffle Advanced, Window Size: {window_size}, Readout: {readout}, FPR: {fpr}, Anchor: {anchor}, Shuffles: {n}"

    # Preliminary Calculations
    precalculated_values = _precalculate(session)
    sv_win_data_base = calculate_single_value_windowed_data(session, precalculated_values, readout, window_size)
    fpr_values_base = calculate_fpr(target_cells, all_cells, sv_win_data_base, fpr)
    fpr_values_dist_base = add_distance_to_fpr(fpr_values_base, session, target_cells, all_cells)

    # Set up the PyQt application and progress window
    progress_window = ProgressWindow(total_steps=n)
    progress_window.show()

    shuffled_fprs_dist = []
    for i in range(n):
        progress_window.update_progress(i + 1)
        if kwargs['temporal']:
            sv_win_data_permuted = permute_sv_win(sv_win_data_base)
        else:
            sv_win_data_permuted = sv_win_data_base
        
        # Calculate the cofiring metric for the shuffled data
        shuffled_fpr = calculate_fpr(target_cells, all_cells, sv_win_data_permuted, fpr, sv_win_data_base=sv_win_data_base, anchor=anchor)
        shuffled_fprs_dist.append(add_distance_to_fpr(shuffled_fpr, session, target_cells, all_cells, shuffle=kwargs['spatial']))
        # If at any point the window is closed, break the loop
        if progress_window.isHidden():
            return None

    progress_window.close()

    visualized_shuffled = VisualizeShuffledAdvanced(name, fpr_values_dist_base, shuffled_fprs_dist, target_cells,
                                                     all_cells, temporal=kwargs['temporal'], spatial=kwargs['spatial'],
                                                     current_window=current_window)

    return visualized_shuffled if visualized_shuffled.global_z_score is not None else None




def calculate_cofiring_for_group(frame_start, cell_positions, target_cells, comparison_cells, cofiring_distances, omit_first=True, **kwargs):
    """This method will call

    Parameters
    ----------
    frame_start : dict
        A dictionary where keys are unit IDs and values are lists of start indices.
    target_cells : list of int
        The cell IDs that will be used for calculating the co-firing metric.
    comparison_cells: list of int
        The cell IDs that will be compared to the target cells.
    Returns
    -------
    float
        The co-firing metric.
    """
    cofiring_params = kwargs['cofiring']
    total_cofiring = 0
    kwargs = {"connections_used": {}}
    for unit_id in target_cells:
        for unit_id2 in comparison_cells:
            if unit_id == unit_id2:
                continue
            
            # Update which cells are being compared
            kwargs["A_id"] = unit_id
            kwargs["B_id"] = unit_id2
            # Get the time bins where the two neurons fire together
            cofiring = check_cofiring(frame_start[unit_id], frame_start[unit_id2],
                                       window_size=cofiring_params["window_size"], omit_first=omit_first,
                                       shareA=cofiring_params["share_a"], shareB=cofiring_params["share_b"], 
                                       direction=cofiring_params["direction"], **kwargs)
            total_cofiring += cofiring

            if cofiring > 0:
                # Get the spatial distance between the two neurons
                spatial_distance = np.linalg.norm(np.array(cell_positions[unit_id]) - np.array(cell_positions[unit_id2]))
                if cofiring in cofiring_distances:
                    cofiring_distances[cofiring].append(spatial_distance)
                else:
                    cofiring_distances[cofiring] = [spatial_distance]


    return total_cofiring, cofiring_distances, kwargs["connections_used"]

def permute_sv_win(sv_win_data_base):
    """
    Permute the single value windowed data.

    Parameters:
    - sv_win_data_base (xr.DataArray): The single value windowed data.

    Returns:
    - sv_win_data_permuted (xr.DataArray): The permuted single value windowed data.
    """
    sv_win_data_permuted = sv_win_data_base.copy()
    unit_ids = sv_win_data_base.unit_id.values
    for unit_id in unit_ids:
        sv_win_data_permuted.loc[{"unit_id": unit_id}] = np.random.permutation(sv_win_data_base.sel(unit_id=unit_id).values)
    return sv_win_data_permuted


def permute_itis_to_start_indices(ieis_dict):
    """
    Generate a new dictionary of start indices by randomly permuting ITIs.

    Parameters:
    - ieis_dict (dict): A dictionary where keys are unit IDs and values are lists of ITIs.

    Returns:
    - dict: A new dictionary where the start indices are calculated based on random permutation of ITIs.
    """
    start_indices_dict = {}

    for unit_id, ieis in ieis_dict.items():
        # Randomly permute the ITIs
        permuted_ieis = np.random.permutation(ieis)

        start_indices = []
        accum_iei = 0
        for iei in permuted_ieis:
            start_indices.append(iei + accum_iei)
            accum_iei += iei

        start_indices_dict[unit_id] = start_indices

    return start_indices_dict

def permute_spatial(positions):
    """
    Permute the spatial positions of the cells.

    Parameters:
    - positions (dict): A dictionary where keys are unit IDs and values are lists of 2D positions.
    """
    positions_values = list(positions.values())
    random.shuffle(positions_values)
    return dict(zip(positions.keys(), positions_values))


class VisualizeShuffledCofiring(QWidget):
    """
    PyQt5 window to visualize shuffled co-firing data with a Matplotlib plot,
    labels for Z-Score and other details, and Matplotlib toolbar.
    """
    def __init__(self, cofiring_original, shuffled_temporal_cofiring, 
                 shuffled_spatial_distances, spatial_original, connections_used, temporal=True):
        super().__init__()

        # Calculate statistics
        mean_shuffled = np.mean(shuffled_temporal_cofiring)
        std_shuffled = np.std(shuffled_temporal_cofiring)
        z_score = (cofiring_original - mean_shuffled) / std_shuffled

        self.connections_used = connections_used
        self.spatial_original = spatial_original
        self.prep_copy_data()

        self.menu = QMenuBar()
        pixmapi_tools = QStyle.StandardPixmap.SP_FileDialogListView
        btn_copy_1 = QAction(self.style().standardIcon(pixmapi_tools), "&Copy Original Data to Clipboard", self)
        btn_copy_1.setStatusTip("Data related utilities")
        btn_copy_1.triggered.connect(lambda: self.copy_to_clipboard(self.df_standard))
        btn_copy_2 = QAction(self.style().standardIcon(pixmapi_tools), "&Copy Original Data to Clipboard (Cell to Cell Matrix)", self)
        btn_copy_2.setStatusTip("Data related utilities")
        btn_copy_2.triggered.connect(lambda: self.copy_to_clipboard(self.df_alt, index=True))
        stats_menu = self.menu.addMenu("&Tools")
        stats_menu.addAction(btn_copy_1)
        stats_menu.addAction(btn_copy_2)

        # Initialize the window
        self.setWindowTitle("Shuffled Cofiring Visualization")
        self.setGeometry(100, 100, 900, 700)

        # Create layout
        layout = QVBoxLayout()

        # Create Matplotlib figure and axes
        figure, ax = plt.subplots(1, 2 if temporal else 1, figsize=(12, 6))

        # Plot histogram
        if temporal:
            ax[0].hist(shuffled_temporal_cofiring, bins=30, color='blue', alpha=0.7, edgecolor='black')
            ax[0].axvline(cofiring_original, color='red', linestyle='--', label='Original')
            ax[0].set_xlabel("Total Co-firing")
            ax[0].set_ylabel("Frequency")
            ax[0].set_title("Co-firing Histogram")
            ax[0].legend()

        # Prepare scatterplot data
        x, y = [], []
        for key, values in shuffled_spatial_distances.items():
            x.extend([key] * len(values))
            y.extend(values)

        # Original data
        x_orig, y_orig = [], []
        for key, values in spatial_original.items():
            x_orig.extend([key] * len(values))
            y_orig.extend(values)

        # Add jitter
        x = np.array(x) + np.random.normal(0, 0.1, len(x))
        x_orig = np.array(x_orig) + np.random.normal(0, 0.1, len(x_orig))

        # Plot scatterplot
        scatter_ax = ax[1] if temporal else ax[0]
        scatter_ax.scatter(y, x, color='lightskyblue', alpha=0.6, label='Shuffled', s=3)
        scatter_ax.scatter(y_orig, x_orig, color='red', alpha=0.8, label='Original', s=4)
        scatter_ax.set_ylabel("Co-firing")
        scatter_ax.set_xlabel("Spatial Distance")
        scatter_ax.set_title("Spatial Distance vs Co-firing")
        scatter_ax.legend()

        # Adjust layout to prevent cut-off
        figure.tight_layout()

        # Embed Matplotlib figure in PyQt5
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        # Add Matplotlib toolbar for interactive options
        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(toolbar)

        # Add labels
        layout.addWidget(QLabel(f"Mean of Shuffled Data: {mean_shuffled:.2f}"))
        layout.addWidget(QLabel(f"Standard Deviation of Shuffled Data: {std_shuffled:.2f}"))
        layout.addWidget(QLabel(f"Original Data: {cofiring_original:.2f}"))
        layout.addWidget(QLabel(f"Z-Score: {z_score:.2f}"))

        # Set layout
        self.setLayout(layout)
        self.layout().setMenuBar(self.menu)

        # Ensure the updated layout is drawn
        canvas.draw()
    
    def prep_copy_data(self):
        """
        Prepare data for copying to clipboard.
        """
        
        # First we do it for the standard format, Where we have three columns, co-firing #, pairs detected, cell pair ids
        df_standard = pd.DataFrame(columns=["Co-firing #", "Pairs Detected", "Cell Pair IDs"])
        # For cofiring # we used the keys of self.spatial_original in ascending order
        cofiring_numbers = sorted(self.spatial_original.keys())
        # For pairs detected we check the length of the values in self.spatial_original
        pairs_detected = [len(self.spatial_original[cofiring]) for cofiring in cofiring_numbers]
        # For cell pair ids we iterate through self.connections. The key are cell pairs and the length of the values are the number of times they co-fired
        cofiring_to_pairs = {}
        for key, values in self.connections_used.items():
            num_of_cofiring = len(values)
            if num_of_cofiring not in cofiring_to_pairs:
                cofiring_to_pairs[num_of_cofiring] = ""
            cell_pair = f"Cell {key[0]} - Cell {key[1]}"
            cofiring_to_pairs[num_of_cofiring] += f"{cell_pair}, "
        
        cell_pair_ids = [cofiring_to_pairs[cofiring] for cofiring in cofiring_numbers]
        df_standard["Co-firing #"] = cofiring_numbers
        df_standard["Pairs Detected"] = pairs_detected
        df_standard["Cell Pair IDs"] = cell_pair_ids

        self.df_standard = df_standard

        # Now we need to something similar for the alternative format, where we have a matrix of cell pairs and their co-firing numbers
        cell_ids = []
        for key in self.connections_used.keys():
            cell_ids.extend(key)
        # Unique and in ascending order
        cell_ids = sorted(list(set(cell_ids)))
        # Now cell ids are the columns and the rows, default to 0, and the name should be "Cell #"
        df_alt = pd.DataFrame(0, index=[f"Cell {i}" for i in cell_ids], columns=[f"Cell {i}" for i in cell_ids])
        # Now we iterate through self.connections_used and set the values in the matrix in two places at once
        for key, values in self.connections_used.items():
            for cell_pair in values:
                df_alt.at[f"Cell {key[0]}", f"Cell {key[1]}"] = len(values)
                df_alt.at[f"Cell {key[1]}", f"Cell {key[0]}"] = len(values)

        self.df_alt = df_alt


    def copy_to_clipboard(self, table, index=False):
        '''
        Copy the data from the table to the clipboard.
        '''
        table.to_clipboard(index=index)
        
        

class VisualizeShuffledAdvanced(QWidget):
    closed = pyqtSignal()
    """
    PyQt5 window to visualize shuffled advanced data with a Matplotlib plot,
    labels for Z-Score and other details, and Matplotlib toolbar.
    """
    def __init__(self, name, fpr_values_base, shuffled_fprs, target_cells, all_cells, temporal=True, spatial=True, current_window=1):
        super().__init__()

        self.fpr_values_base = fpr_values_base
        self.shuffled_fprs = shuffled_fprs
        self.target_cells = target_cells
        self.all_cells = all_cells
        self.temporal = temporal
        self.spatial = spatial
        self.precompute_metrics()
        self.parent = None
        self.name = name
        self.win_num = current_window

        # Initialize the window
        self.setWindowTitle(name)
        self.setGeometry(100, 100, 900, 700)

        # Add menu bar to copy data to clipboard
        self.menu = QMenuBar()
        pixmapi_tools = QStyle.StandardPixmap.SP_FileDialogListView
        btn_copy = QAction(self.style().standardIcon(pixmapi_tools), "&Copy Data to Clipboard", self)
        btn_copy.setStatusTip("Data related utilities")
        btn_copy.triggered.connect(self.copy_to_clipboard)
        stats_menu = self.menu.addMenu("&Tools")
        stats_menu.addAction(btn_copy)


        # Create Matplotlib figure and axes
        self.figure, _ = plt.subplots(1, 2 if (temporal and spatial) else 1, figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        chkbox_layout = QHBoxLayout()
        self.chkbox_colors = QCheckBox("Separate cells by color")
        chkbox_layout.addWidget(self.chkbox_colors)
        self.chkbox_average_spatial = QCheckBox("Average spatial distance")
        chkbox_layout.addWidget(self.chkbox_average_spatial)
        self.chkbox_show_z_score = QCheckBox("Show Global Z-Score")
        chkbox_layout.addWidget(self.chkbox_show_z_score)
        self.chkbox_show_z_score.stateChanged.connect(lambda: self.update_plot(-1))
        self.chkbox_average_spatial.stateChanged.connect(lambda: self.update_plot(-1))
        self.chkbox_colors.stateChanged.connect(lambda: self.update_plot(-1))
        self.layout.addLayout(chkbox_layout)
        self.layout.setMenuBar(self.menu)
        self.setLayout(self.layout)
        self.update_plot(self.win_num)


    def update_plot(self, win_num):
        """
        Update the plot with the data dependent on the window number chosen.
        """
        separate = self.chkbox_colors.isChecked()
        average_spatial = self.chkbox_average_spatial.isChecked()
        if win_num == -1:
            win_num = self.win_num
        
        self.win_num = win_num
        
        # Get the data for the window number
        shuffled_hist_data = self.shuffled_hist_data[self.win_num]
        fpr_temporal = self.fpr_temporal[self.win_num]
        fpr_spatial = self.fpr_spatial[self.win_num]
        shuffled_fpr_temporal = self.shuffled_fpr_temporal[self.win_num]
        shuffled_fpr_spatial = self.shuffled_fpr_spatial[self.win_num]
        hist_data = self.hist_data[self.win_num]        

        self.figure.clf()
        num_plots = [self.temporal, self.spatial, self.chkbox_show_z_score.isChecked()].count(True)
        ax = self.figure.subplots(1, num_plots)
        cmap = plt.get_cmap('gist_ncar')
        num_colors = len(shuffled_hist_data) * 2
        colors = [cmap(i / num_colors) for i in range(num_colors)]
        # Negative values will signify the shuffled data
        mirrored_keys = list(shuffled_hist_data.keys())
        mirrored_keys.extend([-key for key in shuffled_hist_data.keys()])
        cell_to_color = { key: colors[idx] for idx, key in enumerate(mirrored_keys) }
        i = 0
        if self.temporal:
            if num_plots > 1:
                target_axes = ax[0]
            else:
                target_axes = ax
            # First plot histogram of the shuffled data
            if not separate:
                target_axes.hist(self.values_to_list(shuffled_hist_data), bins=30, color='blue', alpha=0.7, edgecolor='black')
            else:
                # Separate the cells by color
                for key, value in shuffled_hist_data.items():
                    target_axes.hist(value, bins=30, color=cell_to_color[key], alpha=0.7, edgecolor='black', label=f"Cell {key}")
            # Keep track of the values so in case of overlap we shift the text
            used_values = {}
            for unit_id, point in zip(fpr_temporal.keys(), self.values_to_list(hist_data)):
                if separate:
                    target_axes.axvline(point, color=cell_to_color[unit_id], linestyle='--')
                else:
                    target_axes.axvline(point, color='red', linestyle='--')
                if point not in used_values:
                    used_values[point] = 0
                else:
                    used_values[point] += 1
                coef = (0.95 - used_values[point] * 0.05) % 1                
                y_value = target_axes.get_ylim()[1] * coef
                target_axes.text(point, y_value, "Cell " + str(unit_id), 
                        rotation=70, verticalalignment='top', fontsize=12, color='black')
            target_axes.set_xlabel("FPR")
            target_axes.set_ylabel("Frequency")
            target_axes.set_title(f"FPR Histogram for Window {self.win_num}")
            if separate:
                target_axes.legend()
            i += 1
        if self.spatial:
            if num_plots > 1:
                target_axes = ax[i]
            else:
                target_axes = ax
            # Now plot the scatterplot of the shuffled data
            if not separate:
                x, y = [], []
                for key in fpr_temporal.keys():
                    x.extend(fpr_spatial[key])
                    y.extend(fpr_temporal[key])
                x_shuffled, y_shuffled = [], []
                for j in range(len(shuffled_fpr_temporal)):
                    for key in shuffled_fpr_temporal[j].keys():
                        x_shuffled.extend(shuffled_fpr_spatial[j][key])
                        y_shuffled.extend(shuffled_fpr_temporal[j][key])
                s = 3
                if average_spatial:
                    x = np.mean(x)
                    x_shuffled = np.mean(x_shuffled)
                    y = np.mean(y)
                    y_shuffled = np.mean(y_shuffled)
                    s = 6
                target_axes.scatter(x_shuffled, y_shuffled, color='lightskyblue', alpha=0.6, label='Shuffled', s=6)
                s = 4 if not average_spatial else 8
                target_axes.scatter(x, y, color='red', alpha=0.8, label='Original', s=s)
            else:
                for key in fpr_temporal.keys():
                    x = fpr_spatial[key]
                    y = fpr_temporal[key]
                    s = 4
                    if average_spatial:
                        x = np.mean(x)
                        y = np.mean(y)
                        s = 8
                    target_axes.scatter(x, y, color=cell_to_color[key], alpha=0.8, label=f"Cell {key}", s=8)
                
                shuffled_fpr_temp_adjusted = {}
                shuffled_fpr_spatial_adjusted = {}
                for key in shuffled_fpr_temporal[0].keys():
                    for j in range(len(shuffled_fpr_temporal)):
                        if key not in shuffled_fpr_temp_adjusted:
                            shuffled_fpr_temp_adjusted[key] = []
                        shuffled_fpr_temp_adjusted[key].extend(shuffled_fpr_temporal[j][key])
                        if key not in shuffled_fpr_spatial_adjusted:
                            shuffled_fpr_spatial_adjusted[key] = []
                        shuffled_fpr_spatial_adjusted[key].extend(shuffled_fpr_spatial[j][key])

                for key in shuffled_fpr_temp_adjusted.keys():
                    x = shuffled_fpr_spatial_adjusted[key]
                    y = shuffled_fpr_temp_adjusted[key]
                    s = 3
                    if average_spatial:
                        x = np.mean(x)
                        y = np.mean(y)
                        s = 6
                    target_axes.scatter(x, y, color=cell_to_color[-key], alpha=0.6, label=f"Shuffled Cell {key}", s=s)

            target_axes.set_ylabel("FPR")
            target_axes.set_xlabel("Spatial Distance")
            target_axes.set_title(f"Spatial Distance vs FPR for Window {self.win_num}")
            target_axes.legend()
            if separate:
                # Deal with duplicate labels
                handles, labels = target_axes.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                target_axes.legend(by_label.values(), by_label.keys())
        if self.chkbox_show_z_score.isChecked():
            if num_plots > 1:
                target_axes = ax[-1]
            else:
                target_axes = ax
            
            # Track overlapping points
            point_registry = {}  # Format: {(x, y): count}
            jitter_amount = 0.1  # Amount to shift points horizontally
            
            # Each Cell ID will have it's own color and the x axis should say the Window Number
            for key, value in self.global_z_score.items():
                # Convert to arrays for easier processing
                x_coords = list(range(1, len(value) + 1))
                y_coords = value
                
                # Check for overlapping points and apply jitter
                jittered_x = []
                for x, y in zip(x_coords, y_coords):
                    # Round y for practical overlap detection
                    point_key = (x, round(float(y), 3))
                    if point_key in point_registry:
                        # Point exists, apply jitter based on count
                        count = point_registry[point_key]
                        # Alternate left/right shift based on count parity
                        shift = jitter_amount * (1 if count % 2 else -1) * ((count + 1) // 2)
                        jittered_x.append(x + shift)
                        point_registry[point_key] += 1
                    else:
                        # New point, no jitter needed
                        jittered_x.append(x)
                        point_registry[point_key] = 1
                
                # Plot with semi-transparency and jittered x-coordinates
                target_axes.scatter(jittered_x, y_coords, label=f"Cell {key}", 
                                  color=cell_to_color[key], alpha=0.7)  # Added alpha for semi-transparency
            
            xs = list(range(1, len(value) + 1))
            for x in xs:
                target_axes.axvline(x=x, color='red', linestyle=':', linewidth=1)
            target_axes.axhline(y=0, color='black', linestyle='-', linewidth=1)
            # Two horizontal line at -1.96 and 1.96 with text
            target_axes.axhline(y=1.96, color='black', linestyle='--', linewidth=1)
            target_axes.text(0, 1.96, "1.96", color='black', fontsize=12)
            target_axes.axhline(y=-1.96, color='black', linestyle='--', linewidth=1)
            target_axes.text(0, -1.96, "-1.96", color='black', fontsize=12)
            target_axes.set_xlabel("Window Number")
            target_axes.set_ylabel("Z-Score")
            target_axes.set_title("Z-Score for all Windows")
            target_axes.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()

    def precompute_metrics(self):
        """
        Precompute the metrics for both z-score and histogram data.
        """
        num_windows = next(iter(self.fpr_values_base.values()))[0].shape[0]

        # Get the data for the window number
        fpr_temporal = {}
        fpr_spatial = {}
        shuffled_fpr_temporal = {}
        shuffled_fpr_spatial = {}
        shuffled_hist_data = {}
        hist_data = {}
        z_scores = {}

        progress_window = ProgressWindow(total_steps=num_windows, text="Precomputing Metrics")
        progress_window.show()

        for i in range(num_windows):
            fpr_temporal[i+1] = {}
            fpr_spatial[i+1] = {}
            shuffled_fpr_temporal[i+1] = []
            shuffled_fpr_spatial[i+1] = []
            shuffled_hist_data[i+1] = {}
            hist_data[i+1] = {}
            for key, value in self.fpr_values_base.items():
                target_cell = key[0]
                if target_cell not in fpr_temporal[i+1]:
                    fpr_temporal[i+1][target_cell] = []
                if target_cell not in fpr_spatial[i+1]:
                    fpr_spatial[i+1][target_cell] = []
                fpr_temporal[i+1][target_cell].append(value[0][i].item())
                fpr_spatial[i+1][target_cell].append(value[1])
        
            
            for key, value in fpr_temporal[i+1].items():
                hist_data[i+1][key] = np.mean(value)
            

            for shuffled_fpr in self.shuffled_fprs:
                local_fpr_temporal = {}
                local_fpr_spatial = {}
                for key, value in shuffled_fpr.items():
                    target_cell = key[0]
                    if target_cell not in local_fpr_temporal:
                        local_fpr_temporal[target_cell] = []
                    if target_cell not in local_fpr_spatial:
                        local_fpr_spatial[target_cell] = []
                    local_fpr_temporal[target_cell].append(value[0][i].item())
                    local_fpr_spatial[target_cell].append(value[1])
                shuffled_fpr_temporal[i+1].append(local_fpr_temporal)
                shuffled_fpr_spatial[i+1].append(local_fpr_spatial)
                
            shuffled_hist_data[i+1] = {}
            for shuffled_fpr_temporal_local in shuffled_fpr_temporal[i+1]:
                for key, value in shuffled_fpr_temporal_local.items():
                    if key not in shuffled_hist_data[i+1]:
                        shuffled_hist_data[i+1][key] = []
                    shuffled_hist_data[i+1][key].append(np.mean(value))

            mean_shuffled = {cell_id: np.mean(self.values_to_list(shuffled_hist_data[i+1])) for cell_id in hist_data[i+1].keys()}
            std_shuffled = {cell_id: np.std(self.values_to_list(shuffled_hist_data[i+1])) for cell_id in hist_data[i+1].keys()}
            for key, value in hist_data[i+1].items():
                if key not in z_scores:
                    z_scores[key] = []
                z_scores[key].append((value - mean_shuffled[key]) / std_shuffled[key])

            progress_window.update_progress(i + 1)

        self.global_z_score = z_scores
        self.fpr_temporal = fpr_temporal
        self.fpr_spatial = fpr_spatial
        self.shuffled_fpr_temporal = shuffled_fpr_temporal
        self.shuffled_fpr_spatial = shuffled_fpr_spatial
        self.hist_data = hist_data
        self.shuffled_hist_data = shuffled_hist_data
        progress_window.close()


    def values_to_list(self, data):
        """
        Convert a dictionary of values to a list.
        """
        return np.array([value for value in data.values()]).flatten()
    

    def get_id(self):
        """
        Get the ID of the window. This will be a combination of the name target and all cells.
        """
        return f"{self.name} - Target Cells: {self.target_cells} - All Cells: {self.all_cells}"

    def copy_to_clipboard(self):
        '''
        Copy the data from the table to the clipboard.
        The copy format will look something like this:
        For each Cell A
        Cell A | 2
        Cell Bs | 3 4 5 ...
        # of Cell Bs | 3
        Shuffling Type | Temporal, Spatial, Temporal + Spatial
        For each time window:
        Window Number | 1
        Then there will be two columns, one for the binned FPR value and the other for the # of cells in that bin
        Distance | FPR
        '''
        copy_lines = []
    
        # Process each target cell separately
        for cell_id in self.target_cells:
            cell_block = []
            # Use tab separation for cells (not a literal comma)
            cell_block.append(f"Cell A:\t{cell_id}")
            
            # Build list of comparison cells (Cell Bs)
            comparison_cells = [str(cell) for cell in self.all_cells if cell != cell_id]
            cell_block.append(f"Cell Bs:\t{' '.join(comparison_cells)}")
            cell_block.append(f"# of Cell Bs:\t{len(comparison_cells)}")
            
            # Determine shuffling type
            shuffling_type = ""
            if self.temporal:
                shuffling_type += "Temporal"
            if self.spatial:
                shuffling_type += (" + " if shuffling_type else "") + "Spatial"
            cell_block.append(f"Shuffling Type:\t{shuffling_type}")
            
            # Blank line to separate cell info from window data
            cell_block.append("")
            
            # ---- Arrange window data side-by-side ----
            # Assume self.fpr_spatial and self.fpr_temporal are dicts keyed by window number.
            windows = sorted(self.fpr_spatial.keys())
            
            # Gather per-window (distance, FPR) pairs for the current cell and track maximum rows
            window_data = {}
            max_rows = 0
            for win in windows:
                dists = self.fpr_spatial[win][cell_id]
                fprs  = self.fpr_temporal[win][cell_id]
                pairs = list(zip(dists, fprs))
                window_data[win] = pairs
                if len(pairs) > max_rows:
                    max_rows = len(pairs)
            
            # Build header row for windows: two columns per window plus an extra gap column.
            header_parts = []
            for win in windows:
                header_parts.append(f"Window {win} - Distance")
                header_parts.append(f"Window {win} - FPR")
                header_parts.append("")  # extra gap column
            cell_block.append("\t".join(header_parts))
            
            # Build data rows: one row per index up to max_rows.
            for i in range(max_rows):
                row_parts = []
                for win in windows:
                    pairs = window_data[win]
                    if i < len(pairs):
                        dist, fpr = pairs[i]
                        row_parts.append(str(dist))
                        row_parts.append(str(fpr))
                    else:
                        row_parts.append("")
                        row_parts.append("")
                    row_parts.append("")  # gap column between windows
                cell_block.append("\t".join(row_parts))
            
            # Add a couple of blank lines to separate blocks for different cells.
            cell_block.append("")
            cell_block.append("")
            
            copy_lines.extend(cell_block)

        # Shuffled data
        # It will keep the same format as above except we will have a new line for each shuffled data
        copy_lines.append("Shuffled Data:")
        copy_lines.append("")
        for i in range(len(self.shuffled_fprs)):
            copy_lines.append(f"Shuffled Data {i+1}:")
            copy_lines.append("")
            # Process each target cell separately
            for cell_id in self.target_cells:
                cell_block = []
                # Use tab separation for cells (not a literal comma)
                cell_block.append(f"Cell A:\t{cell_id}")
                
                # Build list of comparison cells (Cell Bs)
                comparison_cells = [str(cell) for cell in self.all_cells if cell != cell_id]
                cell_block.append(f"Cell Bs:\t{' '.join(comparison_cells)}")
                cell_block.append(f"# of Cell Bs:\t{len(comparison_cells)}")
                
                # Determine shuffling type
                shuffling_type = ""
                if self.temporal:
                    shuffling_type += "Temporal"
                if self.spatial:
                    shuffling_type += (" + " if shuffling_type else "") + "Spatial"
                cell_block.append(f"Shuffling Type:\t{shuffling_type}")
                
                # Blank line to separate cell info from window data
                cell_block.append("")
                
                # ---- Arrange window data side-by-side ----
                # Assume self.fpr_spatial and self.fpr_temporal are dicts keyed by window number.
                windows = sorted(self.fpr_spatial.keys())
                
                # Gather per-window (distance, FPR) pairs for the current cell and track maximum rows
                window_data = {}
                max_rows = 0
                for win in windows:
                    dists = self.shuffled_fpr_spatial[win][i][cell_id]
                    fprs  = self.shuffled_fpr_temporal[win][i][cell_id]
                    pairs = list(zip(dists, fprs))
                    window_data[win] = pairs
                    if len(pairs) > max_rows:
                        max_rows = len(pairs)
                
                
                # Build header row for windows: two columns per window plus an extra gap column.
                header_parts = []
                for win in windows:
                    header_parts.append(f"Window {win} - Distance")
                    header_parts.append(f"Window {win} - FPR")
                    header_parts.append("")
                
                cell_block.append("\t".join(header_parts))

                for j in range(max_rows):
                    row_parts = []
                    for win in windows:
                        pairs = window_data[win]
                        if j < len(pairs):
                            dist, fpr = pairs[j]
                            row_parts.append(str(dist))
                            row_parts.append(str(fpr))
                        else:
                            row_parts.append("")
                            row_parts.append("")
                        row_parts.append("")
                
                    cell_block.append("\t".join(row_parts))
                
                cell_block.append("")
                cell_block.append("")

                copy_lines.extend(cell_block)




        # Join all lines into a single string and copy it to the clipboard.
        copy_string = "\n".join(copy_lines)
        clipboard = QApplication.clipboard()
        clipboard.setText(copy_string)


            
            

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()