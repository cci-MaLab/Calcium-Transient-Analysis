import numpy as np
import pandas as pd

def extract_event_based_data(session, cells, event_indices, event_type="events"):
    """
    Returns a dict mapping each cell_id to its own DataFrame.
    Each DataFrame has rows ["Amplitude","Frequency","Total Amplitude"]
    and columns ["Event {i}" for i in event_indices].
    """
    # 1) prepare sorted event columns
    event_indices = sorted(event_indices)
    event_cols = [f"Event {i+1}" for i in range(len(event_indices))]

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


def _precalculate(session, cells):
    precalculated_values = {}
    E = session.data['E']
    C = session.data['C']
    DFF = session.data['DFF']

    # Unit ids
    unit_ids = cells

    # Output values
    C_based_events = np.zeros(C.shape)
    C_cumulative_events = np.zeros(C.shape)
    DFF_based_events = np.zeros(DFF.shape)
    DFF_cumulative_events = np.zeros(DFF.shape)
    frequency = np.zeros(C.shape)
    # We are also going to maintain a dictionary for transient information
    # what frame the transient starts, its duration and the value in terms of C and DFF
    # Top level are unit_ids which leads to dictionaries of the aforementioned values
    transient_info = {}

    for i, unit_id in enumerate(unit_ids):
        row = E.sel(unit_id=unit_id).values
        C_row = C.sel(unit_id=unit_id).values
        DFF_row = DFF.sel(unit_id=unit_id).values
        events = np.nan_to_num(row, nan=0) # Sometimes saving errors can cause NaNs
        indices = events.nonzero()
        if indices[0].any():
            transient_info[unit_id] = {}
            frame_start = []
            frame_end = []
            c_values = []
            dff_values = []
            # Split up the indices into groups
            split_indices = np.split(indices[0], np.where(np.diff(indices[0]) != 1)[0]+1)
            # Now Split the indices into pairs of first and last indices
            split_indices = [(indices_group[0], indices_group[-1]+1) for indices_group in split_indices]

            C_based_total = 0
            DFF_based_total = 0
            frequency_total = 0
            indices_to_remove = []
            for j, (start, end) in enumerate(split_indices):
                C_based_val = abs(C_row[start:end].max() - C_row[start:end].min())
                DFF_based_val = abs(DFF_row[start:end].max() - DFF_row[start:end].min())

                if C_based_val == 0 or DFF_based_val == 0:
                    indices_to_remove.append(j)
                    continue # Can occur due to erroneous data, skip these

                frame_start.append(start)
                frame_end.append(end)
                c_values.append(C_based_val)
                dff_values.append(DFF_based_val)

                C_based_events[i, start:end] = C_based_val
                DFF_based_events[i, start:end] = DFF_based_val
                frequency_total += 1
                frequency[i, start:end] = frequency_total

                C_based_total += C_based_val
                DFF_based_total += DFF_based_val

                C_cumulative_events[i, start:end] = C_based_total
                DFF_cumulative_events[i, start:end] = DFF_based_total
            

            # Add to info dictionary
            transient_info[unit_id]['frame_start'] = frame_start
            transient_info[unit_id]['frame_end'] = frame_end
            transient_info[unit_id]['C_values'] = c_values
            transient_info[unit_id]['DFF_values'] = dff_values
            transient_info[unit_id]['C_total'] = C_based_total
            transient_info[unit_id]['DFF_total'] = DFF_based_total

            # Remove the erroneous indices
            split_indices = [split_indices[j] for j in range(len(split_indices)) if j not in indices_to_remove]
            
            # Normalize the values by the total in both cases
            C_based_events[i] /= C_based_total
            DFF_based_events[i] /= DFF_based_total
            C_cumulative_events[i] /= C_based_total
            DFF_cumulative_events[i] /= DFF_based_total
            frequency[i] /= frequency_total

            # Interpolate the values to fill in the gaps for cumulative events
            C_cumulative_events[i] = _forward_fill(C_cumulative_events[i])
            DFF_cumulative_events[i] = _forward_fill(DFF_cumulative_events[i])
            frequency[i] = _forward_fill(frequency[i])

            # We'll simulate decay for the base events by taking the last value and multiplying it by 0.95
            for j, (_, end) in enumerate(split_indices):
                last_val_C = C_based_events[i, end-1]
                last_val_DFF = DFF_based_events[i, end-1]

                # Be wary of when the next event starts to not overwrite the values
                next_start = split_indices[j+1][0] if j+1 < len(split_indices) else C_based_events.shape[1]

                # We need to calculate how many frames we need for the decay to be less than 1. 
                C_no_of_frames = int(min(max(np.ceil(np.emath.logn(0.95, 0.01/last_val_C)), 0), next_start-end))
                DFF_no_of_frames = int(min(max(np.ceil(np.emath.logn(0.95, 0.01/last_val_DFF)), 0), next_start-end))


                # Now we need to calculate the decay values, by exponentiation
                if C_no_of_frames > 0:
                    C_decay_powers = np.arange(C_no_of_frames) + 1
                    C_decay_values = last_val_C * 0.95**C_decay_powers
                    C_based_events[i, end:end+C_no_of_frames] = C_decay_values

                if DFF_no_of_frames > 0:
                    DFF_decay_powers = np.arange(DFF_no_of_frames) + 1
                    DFF_decay_values = last_val_DFF * 0.95**DFF_decay_powers
                    DFF_based_events[i, end:end+DFF_no_of_frames] = DFF_decay_values


    precalculated_values['C_transient'] = xr.DataArray(C_based_events, coords=[unit_ids, np.arange(C.shape[1])], dims=['unit_id', 'frame'], name='C_base')
    precalculated_values['C_cumulative'] = xr.DataArray(C_cumulative_events, coords=[unit_ids, np.arange(C.shape[1])], dims=['unit_id', 'frame'], name='C_cumulative')
    precalculated_values['DFF_transient'] = xr.DataArray(DFF_based_events, coords=[unit_ids, np.arange(C.shape[1])], dims=['unit_id', 'frame'], name='DFF_base')
    precalculated_values['DFF_cumulative'] = xr.DataArray(DFF_cumulative_events, coords=[unit_ids, np.arange(C.shape[1])], dims=['unit_id', 'frame'], name='DFF_cumulative')
    precalculated_values['Transient Count'] = xr.DataArray(frequency, coords=[unit_ids, np.arange(C.shape[1])], dims=['unit_id', 'frame'], name='transient_count')
    precalculated_values['transient_info'] = transient_info
    
    return precalculated_values