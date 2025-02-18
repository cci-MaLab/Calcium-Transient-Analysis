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
from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QWidget, QMenuBar, QAction, QStyle)

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


def shuffle_advanced(session, target_cells, comparison_cells, n=100, seed=None, **kwargs):
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

    # Preliminary Calculations
    precalculated_values = _precalculate(session)
    sv_win_data_base = calculate_single_value_windowed_data(session, precalculated_values, readout, window_size)
    fpr_values_base = calculate_fpr(target_cells, all_cells, sv_win_data_base, fpr)
    fpr_values_dist_base = add_distance_to_fpr(fpr_values_base, session)

    # Set up the PyQt application and progress window
    progress_window = ProgressWindow(total_steps=n)
    progress_window.show()

    shuffled_fprs_dist = []
    for i in range(n):
        progress_window.update_progress(i + 1)
        if kwargs['temporal']:
            sv_win_data_permuted = permute_sv_win(sv_win_data_base)
        
        # Calculate the cofiring metric for the shuffled data
        shuffled_fpr = calculate_fpr(target_cells, all_cells, sv_win_data_permuted, fpr)
        shuffled_fprs_dist.append(add_distance_to_fpr(shuffled_fpr, session, shuffle=kwargs['spatial']))

    progress_window.close()

    visualized_shuffled = VisualizeShuffledAdvanced(fpr_values_dist_base, shuffled_fprs_dist, target_cells, all_cells, temporal=kwargs['temporal'], spatial=kwargs['spatial'])

    return visualized_shuffled




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
    """
    PyQt5 window to visualize shuffled advanced data with a Matplotlib plot,
    labels for Z-Score and other details, and Matplotlib toolbar.
    """
    def __init__(self, fpr_values_base, shuffled_fprs, target_cells, all_cells, temporal=True, spatial=True):
        super().__init__()

        self.fpr_values_base = fpr_values_base
        self.shuffled_fprs = shuffled_fprs
        self.target_cells = target_cells
        self.all_cells = all_cells
        self.temporal = temporal
        self.spatial = spatial

        # Initialize the window
        self.setWindowTitle("Shuffled Advanced Visualization")
        self.setGeometry(100, 100, 900, 700)

        # Create Matplotlib figure and axes
        self.figure, _ = plt.subplots(1, 2 if (temporal and spatial) else 1, figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
        self.setLayout(self.layout)
        self.update_plot(1)


    def update_plot(self, win_num):
        """
        Update the plot with the data dependent on the window number chosen.
        """
        i = win_num - 1
        # Get the data for the window number
        fpr_temporal = {}
        fpr_spatial = {}
        for key, value in self.fpr_values_base.items():
            target_cell = key[0]
            if target_cell not in fpr_temporal:
                fpr_temporal[target_cell] = []
            if target_cell not in fpr_spatial:
                fpr_spatial[target_cell] = []
            fpr_temporal[target_cell].append(value[0][i].item())
            fpr_spatial[target_cell].append(value[1])
    
        hist_data = []
        for key, value in fpr_temporal.items():
            hist_data.append(np.mean(value))

        shuffled_fpr_temporal = []
        shuffled_fpr_spatial = []
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
            shuffled_fpr_temporal.append(local_fpr_temporal)
            shuffled_fpr_spatial.append(local_fpr_spatial)
            
        shuffled_hist_data = []
        for shuffled_fpr_temporal_local in shuffled_fpr_temporal:
            for key, value in shuffled_fpr_temporal_local.items():
                shuffled_hist_data.append(np.mean(value))
        

        self.figure.clear()
        ax = self.figure.subplots(1, 2 if (self.temporal and self.spatial) else 1)
        i = 0
        if self.temporal:
            # First plot histogram of the shuffled data
            ax[i].hist(shuffled_hist_data, bins=30, color='blue', alpha=0.7, edgecolor='black')
            for point in hist_data:
                ax[i].axvline(point, color='red', linestyle='--')
            ax[i].set_xlabel("FPR")
            ax[i].set_ylabel("Frequency")
            ax[i].set_title("FPR Histogram")
            i += 1
        if self.spatial:
            # Now plot the scatterplot of the shuffled data
            x, y = [], []
            for key in fpr_temporal.keys():
                x.extend(fpr_spatial[key])
                y.extend(fpr_temporal[key])
            x_shuffled, y_shuffled = [], []
            for j in range(len(shuffled_fpr_temporal)):
                for key in shuffled_fpr_temporal[j].keys():
                    x_shuffled.extend(shuffled_fpr_spatial[j][key])
                    y_shuffled.extend(shuffled_fpr_temporal[j][key])

            ax[i].scatter(x, y, color='lightskyblue', alpha=0.6, label='Shuffled', s=3)
            ax[i].scatter(x_shuffled, y_shuffled, color='red', alpha=0.8, label='Original', s=4)
            ax[i].set_ylabel("FPR")
            ax[i].set_xlabel("Spatial Distance")
            ax[i].set_title("Spatial Distance vs FPR")
            ax[i].legend()
        self.figure.tight_layout()
        self.canvas.draw()
