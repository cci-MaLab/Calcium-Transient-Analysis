import numpy as np
import pandas as pd
from ..gui.sda_widgets import _precalculate
from PyQt5.QtWidgets import QApplication


def extract_event_based_data(session, cells, event_indices, window_size, event_type="events", num_subwindows=1):
    """
    Returns a dict mapping each cell_id to its own DataFrame.
    Each DataFrame has rows ["Average Amplitude", "Frequency", "Total Amplitude"]
    and columns ["Event {i} - Subwindow {j}" for i in event_indices and j in subwindows].
    """
    # 1) prepare sorted event columns
    event_indices = np.array(sorted(event_indices)).flatten()
    
    # Create columns for each event-subwindow combination
    event_cols = []
    if num_subwindows <= 1:
        # No subwindowing, keep original format
        event_cols = [f"Event {i+1}" for i in range(len(event_indices))]
    else:
        # With subwindowing, create columns for each subwindow
        subwindow_size = window_size // num_subwindows
        for i, idx in enumerate(event_indices):
            for subwindow in range(num_subwindows):
                event_cols.append(f"Event {i+1} - Subwindow {subwindow+1}")

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
            if num_subwindows <= 1:
                # Original logic - no subwindowing
                transient_values = []
                for j, start in enumerate(start_values):
                    if start <= idx < start + window_size:
                        transient_values.append(dff_values[j])
                    elif idx <= start + window_size:
                        break

                ave_amp = np.mean(transient_values) if transient_values else 0.0
                freq = len(transient_values)
                total_amp = np.sum(transient_values) if transient_values else 0.0

                df.at["Average Amplitude", f"Event {i+1}"] = ave_amp
                df.at["Frequency", f"Event {i+1}"] = freq
                df.at["Total Amplitude", f"Event {i+1}"] = total_amp
            else:
                # Subwindowing logic
                subwindow_size = window_size // num_subwindows
                for subwindow in range(num_subwindows):
                    # Calculate the start and end of this subwindow
                    subwindow_start = idx + (subwindow * subwindow_size)
                    subwindow_end = subwindow_start + subwindow_size
                    
                    transient_values = []
                    for j, start in enumerate(start_values):
                        if subwindow_start <= start < subwindow_end:
                            transient_values.append(dff_values[j])
                        elif start >= subwindow_end:
                            break

                    ave_amp = np.mean(transient_values) if transient_values else 0.0
                    freq = len(transient_values)
                    total_amp = np.sum(transient_values) if transient_values else 0.0

                    col_name = f"Event {i+1} - Subwindow {subwindow+1}"
                    df.at["Average Amplitude", col_name] = ave_amp
                    df.at["Frequency", col_name] = freq
                    df.at["Total Amplitude", col_name] = total_amp

        result[cell] = df
    
    # Convert to clipboard format
    result_str = []
    for cell_id, df in result.items():
        result_str.append(f"Cell ID\t{cell_id}")
        result_str.append(f"Cell Position\t{session.centroids[cell_id]}")
        result_str.append(df.to_csv(sep='\t', index=True, header=True))
        result_str.append("\n")
    result_str = '\n'.join(result_str)
    
    # Copy to clipboard
    clipboard = QApplication.clipboard()
    clipboard.setText(result_str)
    
    return result