import numpy as np
import pandas as pd
from ..gui.sda_widgets import _precalculate
from PyQt5.QtWidgets import QApplication


def extract_event_based_data(session, cells, event_indices, window_size, event_type="events"):
    """
    Returns a dict mapping each cell_id to its own DataFrame.
    Each DataFrame has rows ["Amplitude","Frequency","Total Amplitude"]
    and columns ["Event {i}" for i in event_indices].
    """
    # 1) prepare sorted event columns
    event_indices = np.array(sorted(event_indices)).flatten()
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
        dff_values = precalculated_values['transient_info'][cell]['DFF_values']
        start_values = precalculated_values['transient_info'][cell]['frame_start']
        # fill in metrics
        for i, idx in enumerate(event_indices):
            transient_values = []
            for j, start in enumerate(start_values):
                if start <= idx < start + window_size:
                    transient_values.append(dff_values[j])
                elif idx <= start + window_size:
                    break

            ave_amp = np.mean(transient_values) if transient_values else 0.0
            freq = len(transient_values)
            total_amp = np.sum(transient_values) if transient_values else 0.0



            df.at["Average Amplitude",        f"Event {i+1}"] = ave_amp
            df.at["Frequency",        f"Event {i+1}"] = freq
            df.at["Total Amplitude",  f"Event {i+1}"] = total_amp

        result[cell] = df    # Now we will convert the result into string format so it can be copied to clipboard
    # The format will be:
    # Cell ID, <cell_id>
    # Cell Position, <cell_position>
    # DataFrame, separated by commas    
    result_str = []
    for cell_id, df in result.items():
        result_str.append(f"Cell ID\t{cell_id}")
        result_str.append(f"Cell Position\t{session.centroids[cell_id]}")
        result_str.append(df.to_csv(sep='\t', index=True, header=True))
        result_str.append("\n")  # Add a newline for separation between cells
    result_str = '\n'.join(result_str)
    
    # Copy to clipboard
    clipboard = QApplication.clipboard()
    clipboard.setText(result_str)