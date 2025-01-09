import numpy as np
from ..gui.sda_widgets import check_cofiring

def shuffle_cofiring(session, unit_ids, n=1, seed=None):
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

    itis = session.get_iei_per_cell()

    shuffled_data = []
    for _ in range(n):
        shuffled_data.append([np.random.permutation(neuron) for neuron in data])

    if n == 1:
        return shuffled_data[0]
    return shuffled_data