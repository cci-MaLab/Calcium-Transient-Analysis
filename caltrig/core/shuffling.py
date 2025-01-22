import numpy as np
from ..gui.sda_widgets import check_cofiring
from ..gui.pop_up_messages import ProgressWindow
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QWidget

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
    list of list of int
        The shuffled data.
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
    cofiring_original, spatial_original = calculate_cofiring_for_group(frame_start, positions, target_cells,
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
        cofiring, shuffled_spatial_distances = calculate_cofiring_for_group(shuffled_frame_start, shuffled_spatial,
                                                                    target_cells, comparison_cells, shuffled_spatial_distances,
                                                                    omit_first=False, **kwargs)

        shuffled_temporal_cofiring.append(cofiring)

    progress_window.close()

    visualize_shuffled = VisualizeShuffledCofiring(cofiring_original, shuffled_temporal_cofiring, shuffled_spatial_distances, spatial_original, temporal=kwargs['temporal'])
    
    return visualize_shuffled



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


    return total_cofiring, cofiring_distances


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
                 shuffled_spatial_distances, spatial_original, temporal=True):
        super().__init__()

        # Calculate statistics
        mean_shuffled = np.mean(shuffled_temporal_cofiring)
        std_shuffled = np.std(shuffled_temporal_cofiring)
        z_score = (cofiring_original - mean_shuffled) / std_shuffled

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
            ax[0].set_xlabel("Co-firing")
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

        # Ensure the updated layout is drawn
        canvas.draw()