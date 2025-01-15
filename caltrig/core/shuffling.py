import numpy as np
from ..gui.sda_widgets import check_cofiring
from ..gui.pop_up_messages import ProgressWindow
import matplotlib.pyplot as plt
import random

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


    # First get the cofiring metric of the original data
    cofiring_original, spatial_original = calculate_cofiring_for_group(frame_start, positions, target_cells,
                                                                        comparison_cells, omit_first=True, **kwargs)
    # Set up the PyQt application and progress window
    progress_window = ProgressWindow(total_steps=n)
    progress_window.show()

    shuffled_temporal_cofiring = []
    shuffled_spatial_distances = {}
    for i in range(n):
        progress_window.update_progress(i + 1)
        shuffled_frame_start = permute_itis_to_start_indices(itis)
        shuffled_spatial = permute_spatial(positions)

        # Calculate the cofiring metric for the shuffled data
        cofiring, shuffled_spatial_distances = calculate_cofiring_for_group(shuffled_frame_start, shuffled_spatial,
                                                                    target_cells, comparison_cells, 
                                                                    omit_first=False, cofiring_distances=shuffled_spatial_distances, **kwargs)

        shuffled_temporal_cofiring.append(cofiring)

    progress_window.close()

    # Now we want to see the mean and standard deviation of the shuffled data
    print("Mean of shuffled data:", np.mean(shuffled_temporal_cofiring))
    print("Standard deviation of shuffled data:", np.std(shuffled_temporal_cofiring))
    print("Original data:", cofiring_original)
    # Z score to see if the p-value is significant
    z_score = (cofiring_original - np.mean(shuffled_temporal_cofiring)) / np.std(shuffled_temporal_cofiring)
    print("Z score:", z_score)

    # First plot: Histogram with vertical line
    plt.figure(figsize=(12, 6))

    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    plt.hist(shuffled_temporal_cofiring, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(cofiring_original, color='red', linestyle='--', label='Original')
    plt.xlabel("Co-firing")
    plt.ylabel("Frequency")
    plt.title("Co-firing Histogram")
    plt.legend()

    # Prepare data for the scatterplot
    x = []
    y = []

    for key, values in shuffled_spatial_distances.items():
        x.extend([key] * len(values))  # Repeat the key for the number of values
        y.extend(values)

    # Map the original data as well in a different color
    x_orig = []
    y_orig = []
    for key, values in spatial_original.items():
        x_orig.extend([key] * len(values))
        y_orig.extend(values)
    
    # For the x values we want to add some jitter to the data
    x = np.array(x) + np.random.normal(0, 0.1, len(x))
    x_orig = np.array(x_orig) + np.random.normal(0, 0.1, len(x_orig))

    # Subplot 2: Scatterplot
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color='blue', alpha=0.6, label='Shuffled', s=3)
    plt.scatter(x_orig, y_orig, color='red', alpha=0.8, label='Original', s=3)
    plt.xlabel("Co-firing")
    plt.ylabel("Spatial Distance")
    plt.title("Co-firing vs Spatial Distance")
    plt.legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()



def calculate_cofiring_for_group(frame_start, cell_positions, target_cells, comparison_cells, cofiring_distances={}, omit_first=True, **kwargs):
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
    temporal = kwargs.get("temporal", True)
    spatial = True if "spatial" in kwargs else False
    total_cofiring = 0
    for unit_id in target_cells:
        for unit_id2 in comparison_cells:
            if unit_id == unit_id2:
                continue

            # Get the time bins where the two neurons fire together
            cofiring = check_cofiring(frame_start[unit_id], frame_start[unit_id2],
                                       window_size=temporal["window_size"], omit_first=omit_first,
                                       shareA=temporal["share_a"], shareB=temporal["share_b"], direction=temporal["direction"])
            total_cofiring += cofiring

            if spatial:
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

    