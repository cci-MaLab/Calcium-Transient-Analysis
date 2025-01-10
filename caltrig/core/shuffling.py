import numpy as np
from ..gui.sda_widgets import check_cofiring
from ..gui.pop_up_messages import ProgressWindow
import matplotlib.pyplot as plt

def shuffle_cofiring(session, unit_ids, n=1000, seed=None):
    """Shuffle the data, keeping the co-firing structure.

    Parameters
    ----------
    data : list of list of int
        The data to shuffle. Each sublist represents a neuron and contains the
        indices of the time bins where the neuron fires.
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

    frame_start, itis = session.get_transient_frames_iti_dict(unit_ids)

    # First get the cofiring metric of the original data
    cofiring_original = calculate_cofiring_for_group(frame_start, unit_ids, omit_first=True)
    # Set up the PyQt application and progress window
    progress_window = ProgressWindow(total_steps=n)
    progress_window.show()

    shuffled_data = []
    for i in range(n):
        progress_window.update_progress(i + 1)
        shuffled_frame_start = permute_ieis_to_start_indices(itis)

        # Calculate the cofiring metric for the shuffled data
        cofiring_shuffled = calculate_cofiring_for_group(shuffled_frame_start, unit_ids)

        shuffled_data.append(cofiring_shuffled)        

    progress_window.close()

    # Now we want to see the mean and standard deviation of the shuffled data
    print("Mean of shuffled data:", np.mean(shuffled_data))
    print("Standard deviation of shuffled data:", np.std(shuffled_data))
    print("Original data:", cofiring_original)
    # Z score to see if the p-value is significant
    z_score = (cofiring_original - np.mean(shuffled_data)) / np.std(shuffled_data)
    print("Z score:", z_score)

    # Make matplotlib plot of the distribution of the shuffled data
    # and show a line for the original data
    plt.hist(shuffled_data, bins=30)
    plt.axvline(cofiring_original, color='r')
    plt.show()



def calculate_cofiring_for_group(frame_start, unit_ids, omit_first=True):
    """This method will call

    Parameters
    ----------


    Returns
    -------
    float
        The co-firing metric.
    """
    cofiring = 0

    for unit_id in unit_ids:
        for unit_id2 in unit_ids:
            if unit_id == unit_id2:
                continue

            # Get the time bins where the two neurons fire together
            cofiring += check_cofiring(frame_start[unit_id], frame_start[unit_id2], window_size=10, omit_first=omit_first)
    return cofiring


def permute_ieis_to_start_indices(ieis_dict):
    """
    Generate a new dictionary of start indices by randomly permuting IEIs.

    Parameters:
    - ieis_dict (dict): A dictionary where keys are unit IDs and values are lists of IEIs.

    Returns:
    - dict: A new dictionary where the start indices are calculated based on random permutation of IEIs.
    """
    start_indices_dict = {}

    for unit_id, ieis in ieis_dict.items():
        # Randomly permute the IEIs
        permuted_ieis = np.random.permutation(ieis)

        start_indices = []
        accum_iei = 0
        for iei in permuted_ieis:
            start_indices.append(iei + accum_iei)
            accum_iei += iei

        start_indices_dict[unit_id] = start_indices

    return start_indices_dict