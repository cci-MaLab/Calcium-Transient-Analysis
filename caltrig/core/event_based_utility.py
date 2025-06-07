import numpy as np
import pandas as pd
from gui.sda_widgets import _precalculate


def extract_event_based_data(session, cells, event_indices, event_type="events"):
    """
    Returns a dict mapping each cell_id to its own DataFrame.
    Each DataFrame has rows ["Amplitude","Frequency","Total Amplitude"]
    and columns ["Event {i}" for i in event_indices].
    """
    # 1) prepare sorted event columns
    event_indices = sorted(event_indices)
    event_cols = [f"Event {i+1}" for i in range(len(event_indices))]

    precalculated_values = _precalculate(session)


    result = {}
    # 2) one DataFrame per cell
    for cell in cells:
        # create empty df
        df = pd.DataFrame(
            index=["Average Amplitude", "Frequency", "Total Amplitude"],
            columns=event_cols,
            dtype=float
        )
        # fill in metrics
        for idx in event_indices:
            # Average Amplitude
            amp = 0.0
            freq = 0.0       # your frequency calc
            total_amp = 0.0  # your total-amplitude calc



            df.at["Amplitude",        f"Event {idx}"] = amp
            df.at["Frequency",        f"Event {idx}"] = freq
            df.at["Total Amplitude",  f"Event {idx}"] = total_amp

        result[cell] = df

    return result