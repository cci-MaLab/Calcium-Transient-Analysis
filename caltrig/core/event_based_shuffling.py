"""
Event-based shuffling utilities for calcium transient analysis.

This module provides both temporal and spatial shuffling capabilities:

TEMPORAL SHUFFLING:
- Shuffles the timing of calcium transients while preserving spatial relationships
- Uses ITI (inter-transient interval) permutation to maintain realistic timing patterns
- Supports amplitude anchoring control (anchored vs independent DFF shuffling)

SPATIAL SHUFFLING:
- Shuffles which cells show activity while preserving temporal patterns
- Maintains event-locked timing structure but randomizes spatial organization
- Calculates clustering metrics: clustering_index, mean_neighbor_distance, spatial_dispersion

VISUALIZATION:
- For temporal: Shows histograms comparing original vs shuffled distributions per cell/event
- For spatial: Shows spatial clustering metrics with significance testing

USAGE:
Call event_based_shuffle_analysis() with shuffle_type="temporal" or "spatial"
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .event_based_utility import extract_event_based_data
from .shuffling import permute_itis_to_start_indices, permute_spatial
from ..gui.pop_up_messages import ProgressWindow
from ..gui.sda_widgets import _precalculate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QMenuBar, QAction, QStyle, QApplication, QTextEdit, QPushButton, QDialog, QLineEdit, QDoubleSpinBox
from PyQt5.QtCore import pyqtSignal, Qt
from scipy.spatial import cKDTree
from collections import defaultdict


def select_high_indices(values, mode="zscore", z_thresh=1.5, q=80.0, robust=True, min_high=3):
    """
    values: 1D array of per-cell values for a single event+metric (aligned to kept cells).
    Returns indices of 'high' cells.
    """
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return np.array([], dtype=int)

    if mode == "zscore":
        if robust:
            med = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals - med))
            sigma = 1.4826 * mad  # MAD->Gaussian sigma
            z = (vals - med) / sigma if sigma > 0 else np.zeros_like(vals)
        else:
            mu = np.nanmean(vals)
            sd = np.nanstd(vals, ddof=1)
            z = (vals - mu) / sd if sd > 0 else np.zeros_like(vals)
        high_idx = np.where(z >= z_thresh)[0]
    else:
        # percentile mode
        thr = np.nanpercentile(vals, q)
        high_idx = np.where(vals >= thr)[0]

    # ensure minimum count (avoids degenerate NN)
    if high_idx.size < min_high:
        # pick the top-k by value (ties handled by argsort)
        k = min_high if np.isfinite(vals).sum() >= min_high else np.isfinite(vals).sum()
        if k > 0:
            order = np.argsort(vals)
            high_idx = order[-k:]
        else:
            high_idx = np.array([], dtype=int)

    return high_idx


def _mean_nn_distance(coords: np.ndarray) -> float:
    """Calculate mean nearest neighbor distance."""
    if coords.shape[0] < 2:
        return np.nan
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)  # k=1 is self, k=2 is nearest neighbor
    return float(np.nanmean(dists[:, 1]))


def _event_metric_vectors(original_data, shuffled_list, selected_cells, event_col, metric):
    """Extract aligned vectors for original and shuffled data."""
    # aligned vectors (len = len(selected_cells))
    orig_vals = []
    for cid in selected_cells:
        v = 0.0
        if cid in original_data and event_col in original_data[cid].columns and metric in original_data[cid].index:
            vv = original_data[cid].loc[metric, event_col]
            if not np.isnan(vv):
                v = float(vv)
        orig_vals.append(v)
    
    shuf_vals = []
    for sh in shuffled_list:
        arr = []
        for cid in selected_cells:
            v = 0.0
            if cid in sh and event_col in sh[cid].columns and metric in sh[cid].index:
                vv = sh[cid].loc[metric, event_col]
                if not np.isnan(vv):
                    v = float(vv)
            arr.append(v)
        shuf_vals.append(np.array(arr))
    
    return np.array(orig_vals), shuf_vals


def _zstats(original_value, shuffled_values_array):
    """Calculate z-score statistics."""
    mu = np.nanmean(shuffled_values_array) if len(shuffled_values_array) else np.nan
    sd = np.nanstd(shuffled_values_array) if len(shuffled_values_array) else np.nan
    if (sd is None) or np.isnan(sd) or sd == 0:
        return 0.0, mu, sd
    return (original_value - mu) / sd, mu, sd


def event_based_shuffle_analysis(
    session,
    selected_cells: List[int],
    event_type: str,
    window_size: int,
    lag: int,
    num_subwindows: int,
    num_shuffles: int,
    amplitude_anchored: bool = True,
    shuffle_type: str = "temporal"
) -> Optional['VisualizeEventBasedShuffling']:
    """
    Perform event-based shuffling analysis on calcium transient data.
    
    This function reuses the window size preview logic for extracting event-based data
    and applies temporal or spatial shuffling similar to the 3D visualization approach.
    
    Parameters:
    -----------
    session : Session object
        The session containing the calcium imaging data
    selected_cells : List[int]
        List of cell IDs to include in the analysis
    event_type : str
        Type of events to analyze ('RNF', 'ALP', 'ILP', 'ALP_Timeout')
    window_size : int
        Size of the analysis window in frames
    lag : int
        Lag offset for the analysis window
    num_subwindows : int
        Number of subwindows to divide the analysis window into
    num_shuffles : int
        Number of shuffle iterations to perform
    amplitude_anchored : bool, optional
        If True, DFF amplitudes stay paired with their timing (default: True)
        If False, DFF amplitudes are shuffled independently from timing
        Only applies to temporal shuffling
    shuffle_type : str, optional
        Type of shuffling to perform: "temporal" or "spatial" (default: "temporal")
        
    Returns:
    --------
    VisualizeEventBasedShuffling or VisualizeSpatialEventShuffling or None
        Visualization window object if successful, None if failed
    """
    
    # Check if the event type exists in the session data (reusing preview logic)
    if event_type not in session.data:
        print(f"Error: Event type '{event_type}' not found in session data")
        return None
    
    if session.data[event_type] is None:
        print(f"Error: Event type '{event_type}' data is None")
        return None
    
    # Extract event indices using the same logic as copy_preview_data_to_clipboard
    events = np.argwhere(session.data[event_type].values == 1)
    if events.size == 0:
        print("Error: No events found for the specified event type")
        return None
    
    # Drop subsequent events that are within 50 frames of each other (same as preview)
    events = np.unique(events[events[:, 0] > 50], axis=0) - lag
    
    if not events.any():
        print("Error: No valid events found after filtering")
        return None
    
    # Extract original event-based data using existing utility function
    original_data = extract_event_based_data(
        session, selected_cells, events, window_size, 
        num_subwindows=num_subwindows, event_type=event_type, 
        name="Original Data"
    )
    
    if not original_data:
        print("Error: Failed to extract original event-based data")
        return None
    
    # Get ITI data for temporal shuffling or cell positions for spatial shuffling
    try:
        if shuffle_type == "temporal":
            frame_start_dict, itis_dict = session.get_transient_frames_iti_dict(selected_cells)
            
            # Validate that we have ITI data for our selected cells
            if not itis_dict:
                print("Error: No ITI data available for selected cells")
                return None
                
            # Check if any selected cells have empty ITI data
            empty_cells = [cell_id for cell_id in selected_cells if cell_id not in itis_dict or len(itis_dict[cell_id]) == 0]
            if empty_cells:
                print(f"Warning: Cells {empty_cells} have no ITI data and will be excluded from shuffling")
        
        elif shuffle_type == "spatial":
            # Get cell positions for spatial shuffling
            cell_positions = {}
            for cell_id in selected_cells:
                try:
                    # Use the same centroid method as main widgets for consistency
                    if hasattr(session, 'centroids') and cell_id in session.centroids:
                        y, x = session.centroids[cell_id]
                        cell_positions[cell_id] = (x, y)  # Store as (x, y) for plotting
                    else:
                        # Fallback to A array extraction if centroids not available
                        A_cell = session.data['A'].sel(unit_id=cell_id)
                        coords = np.unravel_index(np.argmax(A_cell.values), A_cell.shape)
                        cell_positions[cell_id] = (coords[1], coords[0])  # Convert (row, col) to (x, y)
                except Exception as e:
                    print(f"Warning: Could not get position for cell {cell_id}: {e}")
                    # Use random position as fallback
                    cell_positions[cell_id] = (np.random.randint(0, 100), np.random.randint(0, 100))
        
        else:
            print(f"Error: Invalid shuffle_type '{shuffle_type}'. Must be 'temporal' or 'spatial'")
            return None
            
        # Filter out cells with no data (only for temporal shuffling)
        if shuffle_type == "temporal":
            selected_cells = [cell_id for cell_id in selected_cells if cell_id in itis_dict and len(itis_dict[cell_id]) > 0]
            if not selected_cells:
                print("Error: No cells with valid transient data found")
                return None
            
    except Exception as e:
        print(f"Error getting shuffle data: {e}")
        return None
    
    # Set up progress window (reusing from shuffling.py)
    progress_window = ProgressWindow(total_steps=num_shuffles)
    progress_window.show()
    
    # Perform shuffling iterations
    shuffled_results = []
    
    try:
        for shuffle_idx in range(num_shuffles):
            progress_window.update_progress(shuffle_idx + 1)
            
            try:
                # Get precalculated values for this shuffle
                precalculated_values = _precalculate(session)
                
                if shuffle_type == "temporal":
                    # Temporally shuffle the transients using existing function
                    shuffled_frame_start = permute_itis_to_start_indices(itis_dict)
                    
                    # Store original values and replace with shuffled ones
                    original_transient_starts = {}
                    original_dff_values = {}
                    
                    for cell_id in selected_cells:
                        if cell_id in precalculated_values['transient_info']:
                            original_transient_starts[cell_id] = precalculated_values['transient_info'][cell_id]['frame_start'].copy()
                            original_dff_values[cell_id] = precalculated_values['transient_info'][cell_id]['DFF_values'].copy()
                            
                            if cell_id in shuffled_frame_start:
                                # Make sure we don't have any index issues
                                shuffled_starts = shuffled_frame_start[cell_id]
                                if len(shuffled_starts) > 0:
                                    precalculated_values['transient_info'][cell_id]['frame_start'] = np.array(shuffled_starts)
                                
                                # Handle amplitude anchoring
                                if not amplitude_anchored:
                                    # Shuffle DFF amplitudes independently from timing
                                    original_dff = original_dff_values[cell_id]
                                    if len(original_dff) > 0:
                                        # Randomly permute the DFF values
                                        shuffled_dff_indices = np.random.permutation(len(original_dff))
                                        precalculated_values['transient_info'][cell_id]['DFF_values'] = original_dff[shuffled_dff_indices]
                
                elif shuffle_type == "spatial":
                    # Spatially shuffle the cell assignments while keeping timing intact
                    # Create mapping from original to shuffled positions
                    cell_ids = list(selected_cells)
                    shuffled_cell_ids = cell_ids.copy()
                    np.random.shuffle(shuffled_cell_ids)
                    
                    # Store original transient data
                    original_transient_data = {}
                    for cell_id in selected_cells:
                        if cell_id in precalculated_values['transient_info']:
                            original_transient_data[cell_id] = {
                                'frame_start': precalculated_values['transient_info'][cell_id]['frame_start'].copy(),
                                'DFF_values': precalculated_values['transient_info'][cell_id]['DFF_values'].copy()
                            }
                    
                    # Apply spatial shuffle by reassigning transient data to different cells
                    for orig_idx, cell_id in enumerate(cell_ids):
                        shuffled_cell_id = shuffled_cell_ids[orig_idx]
                        if cell_id in original_transient_data and shuffled_cell_id in original_transient_data:
                            # Assign the shuffled cell's data to the current position
                            precalculated_values['transient_info'][cell_id]['frame_start'] = original_transient_data[shuffled_cell_id]['frame_start']
                            precalculated_values['transient_info'][cell_id]['DFF_values'] = original_transient_data[shuffled_cell_id]['DFF_values']
                            
            except Exception as e:
                print(f"Error in shuffling setup for iteration {shuffle_idx + 1}: {e}")
                continue
            
            # Extract shuffled event-based data using the same parameters
            try:
                shuffled_data = extract_event_based_data_with_precalculated(
                    session, selected_cells, events, window_size,
                    num_subwindows=num_subwindows, event_type=event_type,
                    name=f"Shuffle {shuffle_idx + 1}",
                    precalculated_values=precalculated_values
                )
                
                if not shuffled_data:
                    continue
                    
            except Exception as e:
                print(f"Error in shuffle {shuffle_idx + 1}: {e}")
                continue
            
            shuffled_results.append(shuffled_data)
            
    except Exception as e:
        print(f"Error during shuffling: {e}")
        progress_window.close()
        return None
    
    finally:
        progress_window.close()
    
    # Calculate statistics
    statistics = calculate_event_shuffling_statistics(
        original_data, shuffled_results, 
        shuffle_type=shuffle_type, 
        cell_positions=cell_positions if shuffle_type == "spatial" else None,
        selected_cells=selected_cells
    )
    
    # Create appropriate visualization window based on shuffle type
    if shuffle_type == "spatial":
        visualization = VisualizeSpatialEventShuffling(
            original_data=original_data,
            shuffled_data=shuffled_results,
            parameters={
                'event_type': event_type,
                'window_size': window_size,
                'lag': lag,
                'num_subwindows': num_subwindows,
                'num_shuffles': num_shuffles,
                'selected_cells': selected_cells,
                'event_indices': events.tolist(),
                'shuffle_type': shuffle_type,
                'cell_positions': cell_positions,
                'session': session,
                'statistics': statistics
            }
        )
    else:
        visualization = VisualizeEventBasedShuffling(
            original_data=original_data,
            shuffled_data=shuffled_results,
            statistics=statistics,
            parameters={
                'event_type': event_type,
                'window_size': window_size,
                'lag': lag,
                'num_subwindows': num_subwindows,
                'num_shuffles': num_shuffles,
                'selected_cells': selected_cells,
                'event_indices': events.tolist(),
                'shuffle_type': shuffle_type,
                'amplitude_anchored': amplitude_anchored
            }
        )
    
    return visualization


def extract_event_based_data_with_precalculated(session, cells, event_indices, window_size, 
                                               event_type="events", num_subwindows=1, name="",
                                               precalculated_values=None):
    """
    Extract event-based data using provided precalculated values.
    This is a modified version of extract_event_based_data that allows using shuffled transient data.
    """
    if precalculated_values is None:
        # Fall back to regular extraction if no precalculated values provided
        return extract_event_based_data(session, cells, event_indices, window_size, 
                                      event_type, num_subwindows, name)
    
    # Reuse the same logic as extract_event_based_data but with custom precalculated values
    event_indices = np.array(sorted(event_indices)).flatten()
    
    event_cols = []
    if num_subwindows <= 1:
        event_cols = [f"Event {i+1}" for i in range(len(event_indices))]
    else:
        subwindow_size = window_size // num_subwindows
        for i, idx in enumerate(event_indices):
            for subwindow in range(num_subwindows):
                event_cols.append(f"Event {i+1} - Subwindow {subwindow+1}")

    result = {}
    for cell in cells:
        df = pd.DataFrame(
            index=["Average Amplitude", "Frequency", "Total Amplitude"],
            columns=event_cols,
            dtype=float
        )
        
        if cell in precalculated_values['transient_info']:
            dff_values = precalculated_values['transient_info'][cell]['DFF_values']
            start_values = precalculated_values['transient_info'][cell]['frame_start']
            
            # Ensure arrays have matching lengths to prevent index errors
            min_length = min(len(dff_values), len(start_values))
            if min_length == 0:
                # Fill with zeros
                df.iloc[:, :] = 0
                continue
                
            # Use the same extraction logic as the original function
            for i, idx in enumerate(event_indices):
                if num_subwindows <= 1:
                    transient_values = []
                    for j, start in enumerate(start_values):
                        # Add bounds checking to prevent index errors
                        if j >= min_length:
                            break
                        if start <= idx < start + window_size:
                            transient_values.append(dff_values[j])
                        elif idx <= start + window_size:
                            break
                    
                    if transient_values:
                        df.iloc[0, i] = np.mean(transient_values)  # Average Amplitude
                        df.iloc[1, i] = len(transient_values)      # Frequency
                        df.iloc[2, i] = np.sum(transient_values)   # Total Amplitude
                    else:
                        df.iloc[:, i] = 0
                else:
                    # Handle subwindowing logic (similar to original)
                    subwindow_size = window_size // num_subwindows
                    for subwindow in range(num_subwindows):
                        col_idx = i * num_subwindows + subwindow
                        sub_start = idx + subwindow * subwindow_size
                        sub_end = sub_start + subwindow_size
                        
                        transient_values = []
                        for j, start in enumerate(start_values):
                            # Add bounds checking to prevent index errors
                            if j >= min_length:
                                break
                            if sub_start <= start < sub_end:
                                transient_values.append(dff_values[j])
                        
                        if transient_values:
                            df.iloc[0, col_idx] = np.mean(transient_values)
                            df.iloc[1, col_idx] = len(transient_values)
                            df.iloc[2, col_idx] = np.sum(transient_values)
                        else:
                            df.iloc[:, col_idx] = 0
        
        result[cell] = df
    
    return result


def calculate_event_shuffling_statistics(original_data: Dict, shuffled_data_list: List[Dict], 
                                       shuffle_type: str = "temporal", cell_positions: Dict = None, 
                                       selected_cells: List[int] = None) -> Dict[str, Any]:
    """
    Calculate statistical measures comparing original and shuffled event-based data.
    Uses the same approach as existing shuffling functions: z-scores based on mean and std of shuffled distribution.
    
    Parameters:
    -----------
    original_data : Dict
        Original event-based data (cell_id -> DataFrame)
    shuffled_data_list : List[Dict]
        List of shuffled event-based data results
    shuffle_type : str
        Type of shuffling performed ("temporal" or "spatial")
    cell_positions : Dict, optional
        Cell positions for spatial analysis (only needed for spatial shuffling)
    selected_cells : List[int], optional
        List of selected cell IDs to analyze (if None, uses all cells in original_data)
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing statistical measures for each cell and metric
    """
    
    if not shuffled_data_list:
        return {}
    
    statistics = {'shuffle_type': shuffle_type}
    
    # Calculate per-cell statistics (for temporal shuffling)
    if shuffle_type == "temporal":
        # Calculate statistics for each cell
        for cell_id in original_data.keys():
            cell_stats = {}
            original_df = original_data[cell_id]
            
            # For each metric (Average Amplitude, Frequency, Total Amplitude)
            for metric_name in ["Average Amplitude", "Frequency", "Total Amplitude"]:
                if metric_name not in original_df.index:
                    continue
                    
                # Get original values for this metric across all events/subwindows
                original_values = original_df.loc[metric_name].values
                original_mean = np.mean(original_values)
                
                # Collect shuffled values for this metric
                shuffled_means = []
                for shuffled_data in shuffled_data_list:
                    if cell_id in shuffled_data and metric_name in shuffled_data[cell_id].index:
                        shuffled_values = shuffled_data[cell_id].loc[metric_name].values
                        shuffled_means.append(np.mean(shuffled_values))
                
                if shuffled_means:
                    # Calculate statistics using the same approach as existing shuffling functions
                    mean_shuffled = np.mean(shuffled_means)
                    std_shuffled = np.std(shuffled_means)
                    
                    # Calculate z-score: (original - mean_shuffled) / std_shuffled
                    z_score = (original_mean - mean_shuffled) / std_shuffled if std_shuffled > 0 else 0.0
                    
                    cell_stats[metric_name] = {
                        'original_mean': original_mean,
                        'mean_shuffled': mean_shuffled,
                        'std_shuffled': std_shuffled,
                        'z_score': z_score
                    }
            
            statistics[cell_id] = cell_stats
    
    # Calculate spatial clustering metrics (NNR) for spatial shuffling
    elif shuffle_type == "spatial":
        if cell_positions is None:
            statistics["spatial_metrics"] = {}
            return statistics
        
        # Only cells with known positions
        if selected_cells is not None:
            selected_cells_all = selected_cells
        else:
            selected_cells_all = list(original_data.keys())
        kept_cells = [c for c in selected_cells_all if c in cell_positions]
        if len(kept_cells) < 3:
            statistics["spatial_metrics"] = {}
            return statistics

        XY = np.array([cell_positions[c] for c in kept_cells], dtype=float)

        first_cell_df = next(iter(original_data.values()))
        event_cols = list(first_cell_df.columns)
        value_metrics = ["Average Amplitude", "Frequency", "Total Amplitude"]

        spatial_metrics = defaultdict(dict)

        # Precompute NN of all cells once (denominator of ratio)
        nn_all = _mean_nn_distance(XY)

        for event_col in event_cols:
            for metric in value_metrics:
                # Values for this event/metric
                orig_vals_all, shuf_vals_list = _event_metric_vectors(
                    original_data, shuffled_data_list, selected_cells_all, event_col, metric
                )

                # Align to kept_cells mask
                mask = np.array([c in cell_positions for c in selected_cells_all], dtype=bool)
                orig_vals = orig_vals_all[mask]
                shuf_vals_list = [sv[mask] for sv in shuf_vals_list]

                if np.all(np.isnan(orig_vals)) or np.all(orig_vals == 0):
                    # Nothing active; skip
                    continue

                # Pick "high" cells using z-score method
                high_idx = select_high_indices(orig_vals, mode="zscore", z_thresh=1.5, robust=True, min_high=3)
                if len(high_idx) == 0:
                    continue

                # Observed NNR
                nn_high_obs = _mean_nn_distance(XY[high_idx])
                nnr_obs = nn_high_obs / nn_all if (nn_all and not np.isnan(nn_all)) else np.nan

                # Shuffle distribution of NNR
                nnr_shuf = []
                for sv in shuf_vals_list:
                    if np.all(np.isnan(sv)) or np.all(sv == 0):
                        continue
                    hi = select_high_indices(sv, mode="zscore", z_thresh=1.5, robust=True, min_high=3)
                    if len(hi) == 0:
                        continue
                    nn_high = _mean_nn_distance(XY[hi])
                    if nn_all and not np.isnan(nn_all):
                        nnr_shuf.append(nn_high / nn_all)

                nnr_shuf = np.array([x for x in nnr_shuf if not np.isnan(x)], dtype=float)
                z, mu, sd = _zstats(nnr_obs, nnr_shuf)

                key = f"{event_col}::{metric}"
                spatial_metrics["nnr_clustering"][key] = {
                    "original": float(nnr_obs) if not np.isnan(nnr_obs) else np.nan,
                    "shuffled_values": nnr_shuf.tolist(),
                    "shuffled_mean": float(mu) if mu is not None and not np.isnan(mu) else np.nan,
                    "shuffled_std": float(sd) if sd is not None and not np.isnan(sd) else np.nan,
                    "z_score": float(z) if not np.isnan(z) else 0.0,
                    "n_high_cells": len(high_idx),
                    "nn_all_cells": float(nn_all) if not np.isnan(nn_all) else np.nan
                }

        statistics["spatial_metrics"] = spatial_metrics
    
    return statistics


class VisualizeEventBasedShuffling(QWidget):
    """
    Interactive visualization window for event-based shuffling results.
    
    This class creates a window with dropdowns to select cells and events,
    displaying histograms comparing original vs shuffled distributions.
    """
    
    window_closed = pyqtSignal()
    
    def __init__(self, original_data, shuffled_data, statistics, parameters):
        super().__init__()
        
        self.original_data = original_data
        self.shuffled_data = shuffled_data
        self.statistics = statistics
        self.parameters = parameters
        
        self.setup_ui()
        self.plot_current_selection()
        
    def setup_ui(self):
        """Set up the user interface with dropdowns and matplotlib plots."""
        self.setWindowTitle(f"Event-Based Shuffling Results - {self.parameters['event_type']}")
        self.setGeometry(100, 100, 1000, 600)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Control panel at the top
        control_panel = QHBoxLayout()
        
        # Cell selection dropdown
        control_panel.addWidget(QLabel("Select Cell:"))
        self.cell_dropdown = QComboBox()
        self.populate_cell_dropdown()
        self.cell_dropdown.currentTextChanged.connect(self.on_selection_changed)
        control_panel.addWidget(self.cell_dropdown)
        
        # Event selection dropdown
        control_panel.addWidget(QLabel("Select Event:"))
        self.event_dropdown = QComboBox()
        self.populate_event_dropdown()
        self.event_dropdown.currentTextChanged.connect(self.on_selection_changed)
        control_panel.addWidget(self.event_dropdown)
        
        # Metric selection dropdown
        control_panel.addWidget(QLabel("Select Metric:"))
        self.metric_dropdown = QComboBox()
        self.populate_metric_dropdown()
        self.metric_dropdown.currentTextChanged.connect(self.on_selection_changed)
        control_panel.addWidget(self.metric_dropdown)
        
        # Statistics display
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("font-weight: bold; color: blue;")
        control_panel.addWidget(self.stats_label)
        
        control_panel.addStretch()
        main_layout.addLayout(control_panel)
        
        # Matplotlib figure - single plot instead of two side-by-side
        self.figure, self.ax = plt.subplots(1, 1, figsize=(10, 6))
        self.figure.suptitle("Event-Based Shuffling Analysis")
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
        
        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        
        self.setLayout(main_layout)
        
    def populate_cell_dropdown(self):
        """Populate the cell selection dropdown."""
        self.cell_dropdown.clear()
        # Add "All Cells" option first
        self.cell_dropdown.addItem("All Cells")
        # Add individual cells
        for cell_id in self.parameters['selected_cells']:
            self.cell_dropdown.addItem(f"Cell {cell_id}")
            
    def populate_metric_dropdown(self):
        """Populate the metric selection dropdown based on shuffle type."""
        self.metric_dropdown.clear()
        
        if self.parameters.get('shuffle_type') == 'spatial':
            # Add spatial metrics
            if 'spatial_metrics' in self.statistics:
                for metric_name in self.statistics['spatial_metrics'].keys():
                    display_name = metric_name.replace('_', ' ').title()
                    self.metric_dropdown.addItem(display_name)
            else:
                self.metric_dropdown.addItem("No spatial metrics available")
        else:
            # Add temporal metrics (default)
            self.metric_dropdown.addItems(["Average Amplitude", "Frequency", "Total Amplitude"])
            
    def populate_event_dropdown(self):
        """Populate the event selection dropdown."""
        self.event_dropdown.clear()
        num_events = len(self.parameters['event_indices'])
        num_subwindows = self.parameters['num_subwindows']
        
        if num_subwindows <= 1:
            # No subwindows - show events as before
            for i in range(min(num_events, 10)):  # Show first 10 events
                event_idx = self.parameters['event_indices'][i]
                self.event_dropdown.addItem(f"Event {i+1} (Frame {event_idx})")
        else:
            # With subwindows - show each subwindow separately
            for i in range(min(num_events, 10)):  # Show first 10 events
                event_idx = self.parameters['event_indices'][i]
                for subwindow in range(num_subwindows):
                    self.event_dropdown.addItem(f"Event {i+1} - Subwindow {subwindow+1} (Frame {event_idx})")
            
    def get_current_cell_id(self):
        """Get the currently selected cell ID."""
        current_text = self.cell_dropdown.currentText()
        if current_text == "All Cells":
            return "all"
        elif current_text:
            return int(current_text.split()[1])  # Extract cell ID from "Cell X"
        return None
        
    def get_current_event_index(self):
        """Get the currently selected event index."""
        current_index = self.event_dropdown.currentIndex()
        num_subwindows = self.parameters['num_subwindows']
        num_events = len(self.parameters['event_indices'])
        
        if current_index < 0:
            return None
            
        if num_subwindows <= 1:
            # No subwindows - direct event index
            if current_index < num_events:
                return current_index
        else:
            # With subwindows - need to calculate actual event index
            max_items = min(num_events, 10) * num_subwindows
            if current_index < max_items:
                return current_index // num_subwindows
                
        return None
        
    def get_current_subwindow_index(self):
        """Get the currently selected subwindow index (0-based)."""
        current_index = self.event_dropdown.currentIndex()
        num_subwindows = self.parameters['num_subwindows']
        
        if current_index < 0 or num_subwindows <= 1:
            return None
            
        return current_index % num_subwindows
        
    def get_current_metric(self):
        """Get the currently selected metric."""
        current_text = self.metric_dropdown.currentText()
        
        # Convert display names back to internal names for spatial metrics
        if self.parameters.get('shuffle_type') == 'spatial':
            return current_text.lower().replace(' ', '_')
        else:
            return current_text
        
    def on_selection_changed(self):
        """Handle dropdown selection changes."""
        self.plot_current_selection()
        
    def plot_current_selection(self):
        """Plot histogram for shuffled data with original value as a dashed vertical line."""
        
        # Check if this is spatial shuffling
        if self.parameters.get('shuffle_type') == 'spatial':
            self.plot_spatial_results()
        else:
            self.plot_temporal_results()
        
        # Refresh the plot
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_spatial_results(self):
        """Plot spatial clustering analysis results."""
        # Clear previous plot
        self.ax.clear()
        
        # Check if we have spatial metrics
        if 'spatial_metrics' not in self.statistics:
            self.ax.text(0.5, 0.5, 'No spatial metrics available', 
                        ha='center', va='center', transform=self.ax.transAxes)
            return
        
        spatial_metrics = self.statistics['spatial_metrics']
        
        # Create subplot layout for spatial analysis
        self.ax.clear()
        
        # Plot clustering metrics
        metrics_names = list(spatial_metrics.keys())
        if not metrics_names:
            self.ax.text(0.5, 0.5, 'No spatial metrics calculated', 
                        ha='center', va='center', transform=self.ax.transAxes)
            return
        
        # Select current metric from dropdown
        selected_metric = self.get_current_metric()
        if selected_metric not in spatial_metrics:
            selected_metric = metrics_names[0]  # Default to first metric
        
        metric_data = spatial_metrics[selected_metric]
        
        # Plot histogram of shuffled values
        shuffled_values = metric_data['shuffled_values']
        if shuffled_values:
            self.ax.hist(shuffled_values, bins=20, alpha=0.7, color='red', edgecolor='black',
                        label=f'Shuffled (n={len(shuffled_values)})')
        
        # Plot original value as vertical line
        original_value = metric_data['original']
        self.ax.axvline(original_value, color='blue', linestyle='--', linewidth=2, 
                       label=f'Original = {original_value:.3f}')
        
        # Add statistics text
        z_score = metric_data['z_score']
        shuffled_mean = metric_data['shuffled_mean']
        significance = "***" if abs(z_score) > 2.58 else "**" if abs(z_score) > 1.96 else "*" if abs(z_score) > 1.64 else ""
        
        # Format title and labels
        self.ax.set_xlabel(f'{selected_metric.replace("_", " ").title()}')
        self.ax.set_ylabel('Frequency')
        self.ax.set_title(f'Spatial Analysis: {selected_metric.replace("_", " ").title()}\n'
                         f'Z-score = {z_score:.2f} {significance}')
        self.ax.legend()
        
        # Add text box with statistics
        textstr = f'Original: {original_value:.3f}\nShuffled Mean: {shuffled_mean:.3f}\nZ-score: {z_score:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.ax.text(0.02, 0.98, textstr, transform=self.ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
    def plot_temporal_results(self):
        """Plot temporal shuffling analysis results (original behavior)."""
        cell_id = self.get_current_cell_id()
        event_idx = self.get_current_event_index()
        selected_metric = self.get_current_metric()
        
        if cell_id is None or event_idx is None or not selected_metric:
            return
            
        # Clear previous plot
        self.ax.clear()
        
        if cell_id == "all":
            # Handle "All Cells" case - collate data across all selected cells
            self.plot_all_cells(event_idx, selected_metric)
        else:
            # Handle individual cell case
            self.plot_individual_cell(cell_id, event_idx, selected_metric)
        
    def plot_individual_cell(self, cell_id, event_idx, selected_metric):
        """Plot data for an individual cell."""
        # Get the column name based on subwindows
        num_subwindows = self.parameters['num_subwindows']
        subwindow_idx = self.get_current_subwindow_index()
        
        if num_subwindows <= 1:
            event_col_name = f"Event {event_idx + 1}"
        else:
            if subwindow_idx is not None:
                event_col_name = f"Event {event_idx + 1} - Subwindow {subwindow_idx + 1}"
            else:
                return  # Invalid subwindow selection
        
        # Get original value for this cell, event, and metric
        original_value = None
        if cell_id in self.original_data:
            cell_data = self.original_data[cell_id]  # This is a pandas DataFrame
            if event_col_name in cell_data.columns and selected_metric in cell_data.index:
                original_value = cell_data.loc[selected_metric, event_col_name]
                
        # Get shuffled values for this cell, event, and metric
        shuffled_values = []
        for shuffle_result in self.shuffled_data:
            if cell_id in shuffle_result:
                cell_data = shuffle_result[cell_id]  # This is also a pandas DataFrame
                if event_col_name in cell_data.columns and selected_metric in cell_data.index:
                    shuffle_val = cell_data.loc[selected_metric, event_col_name]
                    if not np.isnan(shuffle_val):
                        shuffled_values.append(shuffle_val)
                    
        # Plot shuffled data histogram
        if shuffled_values:
            self.ax.hist(shuffled_values, bins=20, alpha=0.7, color='red', edgecolor='black', 
                        label=f'Shuffled (n={len(shuffled_values)})')
            
        # Plot original value as a dashed vertical line
        if original_value is not None and not np.isnan(original_value):
            self.ax.axvline(x=original_value, color='blue', linestyle='--', linewidth=2, 
                           label=f'Original ({original_value:.3f})')
            
        # Set labels and title
        title_text = f"{selected_metric}\nCell {cell_id}, {event_col_name}"
        self.ax.set_title(title_text)
        self.ax.set_xlabel(f"{selected_metric} Value")
        self.ax.set_ylabel("Frequency")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
            
        # Update statistics display
        self.update_statistics_display(cell_id, event_idx, selected_metric, subwindow_idx)
        
    def plot_all_cells(self, event_idx, selected_metric):
        """Plot collated data across all selected cells."""
        # Get the column name based on subwindows
        num_subwindows = self.parameters['num_subwindows']
        subwindow_idx = self.get_current_subwindow_index()
        
        if num_subwindows <= 1:
            event_col_name = f"Event {event_idx + 1}"
            title_suffix = f"Event {event_idx + 1}"
        else:
            if subwindow_idx is not None:
                event_col_name = f"Event {event_idx + 1} - Subwindow {subwindow_idx + 1}"
                title_suffix = f"Event {event_idx + 1} - Subwindow {subwindow_idx + 1}"
            else:
                return  # Invalid subwindow selection
        
        # Collect original values across all cells
        original_values = []
        for cell_id in self.parameters['selected_cells']:
            if cell_id in self.original_data:
                cell_data = self.original_data[cell_id]
                if event_col_name in cell_data.columns and selected_metric in cell_data.index:
                    orig_val = cell_data.loc[selected_metric, event_col_name]
                    if not np.isnan(orig_val):
                        original_values.append(orig_val)
        
        # Collect shuffled values across all cells
        all_shuffled_values = []
        for shuffle_result in self.shuffled_data:
            shuffle_values_this_iteration = []
            for cell_id in self.parameters['selected_cells']:
                if cell_id in shuffle_result:
                    cell_data = shuffle_result[cell_id]
                    if event_col_name in cell_data.columns and selected_metric in cell_data.index:
                        shuffle_val = cell_data.loc[selected_metric, event_col_name]
                        if not np.isnan(shuffle_val):
                            shuffle_values_this_iteration.append(shuffle_val)
            
            # Use sum for Total Amplitude, mean for others
            if shuffle_values_this_iteration:
                if selected_metric == "Total Amplitude":
                    all_shuffled_values.append(np.sum(shuffle_values_this_iteration))
                else:
                    all_shuffled_values.append(np.mean(shuffle_values_this_iteration))
                    
        # Plot shuffled data histogram
        if all_shuffled_values:
            self.ax.hist(all_shuffled_values, bins=20, alpha=0.7, color='red', edgecolor='black', 
                        label=f'Shuffled (n={len(all_shuffled_values)})')
            
        # Plot original mean/sum as a dashed vertical line
        if original_values:
            if selected_metric == "Total Amplitude":
                original_aggregate = np.sum(original_values)
                line_label = f'Original Sum ({original_aggregate:.3f})'
            else:
                original_aggregate = np.mean(original_values)
                line_label = f'Original Mean ({original_aggregate:.3f})'
            
            self.ax.axvline(x=original_aggregate, color='blue', linestyle='--', linewidth=2, 
                           label=line_label)
            
        # Set labels and title
        aggregation_type = "Sum" if selected_metric == "Total Amplitude" else "Mean"
        self.ax.set_title(f"{selected_metric}\nAll Cells (n={len(self.parameters['selected_cells'])}), {title_suffix}")
        self.ax.set_xlabel(f"{selected_metric} {aggregation_type}")
        self.ax.set_ylabel("Frequency")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
            
        # Update statistics display for all cells
        self.update_statistics_display_all_cells(event_idx, selected_metric, original_values, all_shuffled_values, subwindow_idx)
        
    def update_statistics_display(self, cell_id, event_idx, selected_metric, subwindow_idx=None):
        """Update the statistics label with current selection stats."""
        if subwindow_idx is not None:
            stats_text = f"Cell {cell_id}, Event {event_idx + 1} - Subwindow {subwindow_idx + 1}, {selected_metric}: "
        else:
            stats_text = f"Cell {cell_id}, Event {event_idx + 1}, {selected_metric}: "
        
        # Look for statistics for this cell and metric
        if cell_id in self.statistics and selected_metric in self.statistics[cell_id]:
            metric_stats = self.statistics[cell_id][selected_metric]
            z_score = metric_stats.get('z_score', 'N/A')
            original_mean = metric_stats.get('original_mean', 'N/A')
            mean_shuffled = metric_stats.get('mean_shuffled', 'N/A')
            
            if isinstance(z_score, (int, float)):
                stats_text += f"Z-score: {z_score:.3f}, "
            else:
                stats_text += f"Z-score: {z_score}, "
                
            if isinstance(original_mean, (int, float)):
                stats_text += f"Original: {original_mean:.3f}, "
            else:
                stats_text += f"Original: {original_mean}, "
                
            if isinstance(mean_shuffled, (int, float)):
                stats_text += f"Shuffled Mean: {mean_shuffled:.3f}"
            else:
                stats_text += f"Shuffled Mean: {mean_shuffled}"
        else:
            stats_text += "Statistics not available"
            
        self.stats_label.setText(stats_text)
        
    def update_statistics_display_all_cells(self, event_idx, selected_metric, original_values, shuffled_values, subwindow_idx=None):
        """Update the statistics label for the 'All Cells' case."""
        if subwindow_idx is not None:
            stats_text = f"All Cells, Event {event_idx + 1} - Subwindow {subwindow_idx + 1}, {selected_metric}: "
        else:
            stats_text = f"All Cells, Event {event_idx + 1}, {selected_metric}: "
        
        if original_values and shuffled_values:
            # Use sum for Total Amplitude, mean for others
            if selected_metric == "Total Amplitude":
                original_aggregate = np.sum(original_values)
                aggregation_type = "Sum"
            else:
                original_aggregate = np.mean(original_values)
                aggregation_type = "Mean"
                
            shuffled_mean = np.mean(shuffled_values)
            shuffled_std = np.std(shuffled_values)
            
            # Calculate z-score
            if shuffled_std > 0:
                z_score = (original_aggregate - shuffled_mean) / shuffled_std
                stats_text += f"Z-score: {z_score:.3f}, "
            else:
                stats_text += "Z-score: N/A, "
                
            stats_text += f"Original {aggregation_type}: {original_aggregate:.3f}, "
            stats_text += f"Shuffled Mean: {shuffled_mean:.3f}"
        else:
            stats_text += "Statistics not available"
            
        self.stats_label.setText(stats_text)
        
    def closeEvent(self, event):
        """Handle window close event."""
        self.window_closed.emit()
        event.accept()
        
    def show_window(self):
        """Show the visualization window."""
        self.show()
        self.raise_()
        self.activateWindow()


class VisualizeSpatialEventShuffling(QWidget):
    """
    Visualization window for spatial event-based shuffling results.
    
    Shows spatial maps with NNR clustering statistics.
    """
    
    window_closed = pyqtSignal()
    
    def __init__(self, original_data, shuffled_data, parameters):
        super().__init__()
        
        self.original_data = original_data
        self.shuffled_data = shuffled_data
        self.parameters = parameters
        
        # Extract cell positions
        self.cell_positions = parameters['cell_positions']
        
        # Get image dimensions for coordinate flipping
        if 'session' in parameters:
            try:
                A_data = parameters['session'].data['A']
                self.image_height = A_data.coords["height"].values.max()
            except:
                self.image_height = None
        else:
            self.image_height = None
        
        # Track colorbars to remove them when updating plots
        self.colorbars = []
        
        # Initialize parameter values BEFORE UI setup
        self.current_pvalue = 0.05  # Default p < 0.05
        self.current_method = "Z-score"  # Default method
        self.z_threshold = 1.5
        self.quantile_low = 0.1
        self.quantile_high = 0.9
        self.range_low = 0.0
        self.range_high = 1.0
        
        # Get statistics from parameters or calculate them
        if 'statistics' in parameters:
            self.statistics = parameters['statistics']
        else:
            # Calculate spatial statistics if not provided
            self.statistics = calculate_event_shuffling_statistics(
                original_data, shuffled_data, 
                shuffle_type="spatial", 
                cell_positions=parameters['cell_positions'],
                selected_cells=parameters['selected_cells']
            )
        
        # Prepare spatial data for visualization
        self.prepare_spatial_data()
        
        self.setup_ui()
        self.plot_current_selection()
        self.update_statistics_display()
        
    def prepare_spatial_data(self):
        """Prepare spatial data organized by event and metric."""
        self.spatial_data = {}
        
        # Get event columns
        first_cell_data = next(iter(self.original_data.values()))
        event_columns = list(first_cell_data.columns)
        
        for event_col in event_columns:
            self.spatial_data[event_col] = {
                'original': {},
                'shuffled_avg': {}
            }
            
            # For each metric
            for metric in ["Average Amplitude", "Frequency", "Total Amplitude"]:
                self.spatial_data[event_col]['original'][metric] = {}
                self.spatial_data[event_col]['shuffled_avg'][metric] = {}
                
                # Get original data for this event and metric
                for cell_id in self.parameters['selected_cells']:
                    if cell_id in self.original_data and cell_id in self.cell_positions:
                        if event_col in self.original_data[cell_id].columns and metric in self.original_data[cell_id].index:
                            value = self.original_data[cell_id].loc[metric, event_col]
                            self.spatial_data[event_col]['original'][metric][cell_id] = value
                
                # Calculate average across all shuffled data for this event and metric
                shuffled_averages = {}
                for cell_id in self.parameters['selected_cells']:
                    if cell_id in self.cell_positions:
                        cell_shuffled_values = []
                        for shuffle_result in self.shuffled_data:
                            if cell_id in shuffle_result and event_col in shuffle_result[cell_id].columns and metric in shuffle_result[cell_id].index:
                                value = shuffle_result[cell_id].loc[metric, event_col]
                                if not np.isnan(value):
                                    cell_shuffled_values.append(value)
                        
                        if cell_shuffled_values:
                            shuffled_averages[cell_id] = np.mean(cell_shuffled_values)
                        else:
                            shuffled_averages[cell_id] = 0.0
                
                self.spatial_data[event_col]['shuffled_avg'][metric] = shuffled_averages
    
    def setup_ui(self):
        """Set up the user interface with spatial plots and statistics."""
        self.setWindowTitle(f"Spatial Event-Based Shuffling Results - {self.parameters['event_type']}")
        self.setGeometry(100, 100, 1400, 800)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Control panel at the top
        control_panel = QVBoxLayout()
        
        # First row: Event and Metric selection
        selection_row = QHBoxLayout()
        
        # Event selection dropdown
        selection_row.addWidget(QLabel("Select Event:"))
        self.event_dropdown = QComboBox()
        self.populate_event_dropdown()
        self.event_dropdown.currentTextChanged.connect(self.on_selection_changed)
        selection_row.addWidget(self.event_dropdown)
        
        # Metric selection dropdown
        selection_row.addWidget(QLabel("Select Metric:"))
        self.metric_dropdown = QComboBox()
        self.metric_dropdown.addItems(["Average Amplitude", "Frequency", "Total Amplitude"])
        self.metric_dropdown.currentTextChanged.connect(self.on_selection_changed)
        selection_row.addWidget(self.metric_dropdown)
        
        selection_row.addStretch()
        control_panel.addLayout(selection_row)
        
        # Second row: Analysis parameters
        params_row = QHBoxLayout()
        
        # P-value selection for NNR significance
        params_row.addWidget(QLabel("NNR Significance:"))
        self.pvalue_dropdown = QComboBox()
        self.pvalue_dropdown.addItems(["p < 0.05", "p < 0.01", "p < 0.10"])
        self.pvalue_dropdown.currentTextChanged.connect(self.on_parameters_changed)
        params_row.addWidget(self.pvalue_dropdown)
        
        # Cell selection method
        params_row.addWidget(QLabel("Cell Selection:"))
        self.selection_method_dropdown = QComboBox()
        self.selection_method_dropdown.addItems(["Z-score", "Quantile Range", "Raw Range"])
        self.selection_method_dropdown.currentTextChanged.connect(self.on_method_changed)
        params_row.addWidget(self.selection_method_dropdown)
        
        params_row.addStretch()
        control_panel.addLayout(params_row)
        
        # Third row: Method-specific parameters (dynamic)
        self.method_params_layout = QHBoxLayout()
        self.setup_method_parameters()
        control_panel.addLayout(self.method_params_layout)
        
        # Fourth row: Action buttons
        button_row = QHBoxLayout()
        
        # Show full statistics button
        self.full_stats_button = QPushButton("Show Full Statistics")
        self.full_stats_button.clicked.connect(self.show_full_statistics)
        button_row.addWidget(self.full_stats_button)
        
        button_row.addStretch()
        control_panel.addLayout(button_row)
        main_layout.addLayout(control_panel)
        
        # Create horizontal layout for plots and statistics
        content_layout = QHBoxLayout()
        
        # Left side: Matplotlib figure - single original plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        self.figure, self.ax_original = plt.subplots(1, 1, figsize=(8, 6))
        self.figure.suptitle("Original Spatial Pattern")
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        
        content_layout.addWidget(plot_widget, stretch=1)
        
        # Right side: Statistics panel (full height)
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        stats_title = QLabel("NNR Clustering Statistics")
        stats_title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        stats_layout.addWidget(stats_title)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("font-family: 'Courier New', monospace; font-size: 12px;")
        # Remove height restriction to use full vertical space
        stats_layout.addWidget(self.stats_text)
        
        content_layout.addWidget(stats_widget, stretch=1)
        
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)
        
    def setup_method_parameters(self):
        """Set up the method-specific parameter controls."""
        # Clear existing widgets and layout items properly
        while self.method_params_layout.count():
            child = self.method_params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.spacerItem():
                # Remove spacer items too
                pass
        
        # Add parameters based on current method
        method = self.selection_method_dropdown.currentText() if hasattr(self, 'selection_method_dropdown') else "Z-score"
        
        if method == "Z-score":
            self.method_params_layout.addWidget(QLabel("Threshold:"))
            self.z_spinbox = QDoubleSpinBox()
            self.z_spinbox.setRange(0.1, 5.0)
            self.z_spinbox.setSingleStep(0.1)
            self.z_spinbox.setValue(self.z_threshold)
            self.z_spinbox.valueChanged.connect(self.on_z_threshold_changed)
            self.method_params_layout.addWidget(self.z_spinbox)
            
        elif method == "Quantile Range":
            self.method_params_layout.addWidget(QLabel("Exclude outside:"))
            self.quantile_low_spinbox = QDoubleSpinBox()
            self.quantile_low_spinbox.setRange(0.0, 0.5)
            self.quantile_low_spinbox.setSingleStep(0.01)
            self.quantile_low_spinbox.setValue(self.quantile_low)
            self.quantile_low_spinbox.valueChanged.connect(self.on_quantile_changed)
            self.method_params_layout.addWidget(self.quantile_low_spinbox)
            
            self.method_params_layout.addWidget(QLabel("-"))
            self.quantile_high_spinbox = QDoubleSpinBox()
            self.quantile_high_spinbox.setRange(0.5, 1.0)
            self.quantile_high_spinbox.setSingleStep(0.01)
            self.quantile_high_spinbox.setValue(self.quantile_high)
            self.quantile_high_spinbox.valueChanged.connect(self.on_quantile_changed)
            self.method_params_layout.addWidget(self.quantile_high_spinbox)
            
        elif method == "Raw Range":
            self.method_params_layout.addWidget(QLabel("Exclude outside:"))
            self.range_low_spinbox = QDoubleSpinBox()
            self.range_low_spinbox.setRange(-999.0, 999.0)
            self.range_low_spinbox.setSingleStep(0.1)
            self.range_low_spinbox.setValue(self.range_low)
            self.range_low_spinbox.valueChanged.connect(self.on_range_changed)
            self.method_params_layout.addWidget(self.range_low_spinbox)
            
            self.method_params_layout.addWidget(QLabel("-"))
            self.range_high_spinbox = QDoubleSpinBox()
            self.range_high_spinbox.setRange(-999.0, 999.0)
            self.range_high_spinbox.setSingleStep(0.1)
            self.range_high_spinbox.setValue(self.range_high)
            self.range_high_spinbox.valueChanged.connect(self.on_range_changed)
            self.method_params_layout.addWidget(self.range_high_spinbox)
        
        # Always add stretch at the end to maintain consistent layout
        self.method_params_layout.addStretch()
        
    def on_method_changed(self):
        """Handle changes to cell selection method."""
        self.current_method = self.selection_method_dropdown.currentText()
        self.setup_method_parameters()
        self.on_selection_changed()
        
    def on_parameters_changed(self):
        """Handle changes to p-value selection."""
        pvalue_text = self.pvalue_dropdown.currentText()
        if "0.05" in pvalue_text:
            self.current_pvalue = 0.05
        elif "0.01" in pvalue_text:
            self.current_pvalue = 0.01
        elif "0.10" in pvalue_text:
            self.current_pvalue = 0.10
        self.update_statistics_display()  # Retrigger description generation
        
    def on_z_threshold_changed(self, value):
        """Handle changes to z-score threshold."""
        self.z_threshold = value
        self.on_selection_changed()  # Update both plot and statistics
        
    def on_quantile_changed(self):
        """Handle changes to quantile range."""
        if hasattr(self, 'quantile_low_spinbox') and hasattr(self, 'quantile_high_spinbox'):
            self.quantile_low = self.quantile_low_spinbox.value()
            self.quantile_high = self.quantile_high_spinbox.value()
            self.on_selection_changed()  # Update both plot and statistics
            
    def on_range_changed(self):
        """Handle changes to raw value range."""
        if hasattr(self, 'range_low_spinbox') and hasattr(self, 'range_high_spinbox'):
            self.range_low = self.range_low_spinbox.value()
            self.range_high = self.range_high_spinbox.value()
            self.on_selection_changed()  # Update both plot and statistics
            
    def get_high_activity_indices(self, values_array):
        """Get indices of high-activity cells based on current selection method."""
        if self.current_method == "Z-score":
            return select_high_indices(values_array, mode="zscore", z_thresh=self.z_threshold, robust=True, min_high=3)
        
        elif self.current_method == "Quantile Range":
            # Mark cells outside the quantile range
            low_threshold = np.nanpercentile(values_array, self.quantile_low * 100)
            high_threshold = np.nanpercentile(values_array, self.quantile_high * 100)
            outside_indices = np.where((values_array < low_threshold) | (values_array > high_threshold))[0]
            return outside_indices
        
        elif self.current_method == "Raw Range":
            # Mark cells outside the raw value range
            outside_indices = np.where((values_array < self.range_low) | (values_array > self.range_high))[0]
            return outside_indices
        
        return np.array([], dtype=int)
        
    def populate_event_dropdown(self):
        """Populate the event selection dropdown."""
        self.event_dropdown.clear()
        first_cell_data = next(iter(self.original_data.values()))
        event_columns = list(first_cell_data.columns)
        
        for event_col in event_columns:
            if "Event" in event_col:
                self.event_dropdown.addItem(event_col)
    
    def get_current_event(self):
        """Get the currently selected event."""
        return self.event_dropdown.currentText()
        
    def get_current_metric(self):
        """Get the currently selected metric."""
        return self.metric_dropdown.currentText()
        
    def on_selection_changed(self):
        """Handle dropdown selection changes."""
        self.plot_current_selection()
        self.update_statistics_display()
        
    def plot_current_selection(self):
        """Plot spatial map for the selected event and metric."""
        event_col = self.get_current_event()
        metric = self.get_current_metric()
        
        if not event_col or not metric:
            return
        
        # Remove old colorbars
        for cbar in self.colorbars:
            cbar.remove()
        self.colorbars.clear()
        
        # Clear previous plot
        self.ax_original.clear()
        
        # Get data for this event and metric
        if event_col not in self.spatial_data:
            self.ax_original.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=self.ax_original.transAxes)
            return
        
        original_data = self.spatial_data[event_col]['original'][metric]
        
        # Plot original spatial map only
        self.plot_spatial_map(self.ax_original, original_data, f'Original {event_col} - {metric}')
        
        # Refresh the plot
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_spatial_map(self, ax, data_dict, title):
        """Plot spatial map on given axis."""
        ax.set_title(title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        if not data_dict or not self.cell_positions:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get all cell positions and their values
        x_positions = []
        y_positions = []
        values = []
        
        for cell_id in self.parameters['selected_cells']:
            if cell_id in self.cell_positions:
                pos = self.cell_positions[cell_id]
                x_positions.append(pos[0])
                
                # Flip y-coordinate to match main widget orientation
                if self.image_height is not None:
                    y_flipped = self.image_height - pos[1]
                    y_positions.append(y_flipped)
                else:
                    y_positions.append(pos[1])
                
                # Get value for this cell, default to 0 if not active
                value = data_dict.get(cell_id, 0.0)
                values.append(value)
        
        if not x_positions:
            ax.text(0.5, 0.5, 'No cell positions available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create scatter plot with color-coded values
        scatter = ax.scatter(x_positions, y_positions, c=values, s=50, cmap='viridis', alpha=0.7)
        
        # Add red X markers for high-activity cells based on current method
        event_col = self.get_current_event()
        metric = self.get_current_metric()
        
        if event_col and metric:
            # Get high-activity cells using current selection method
            values_array = np.array(values)
            if len(values_array) > 0:
                high_indices = self.get_high_activity_indices(values_array)
                
                if len(high_indices) > 0:
                    # Get positions of high-activity cells
                    high_x = [x_positions[i] for i in high_indices]
                    high_y = [y_positions[i] for i in high_indices]
                    
                    # Create label based on method
                    if self.current_method == "Z-score":
                        label = f'High Activity (z≥{self.z_threshold}, n={len(high_indices)})'
                    elif self.current_method == "Quantile Range":
                        label = f'Outside Range ({self.quantile_low:.2f}-{self.quantile_high:.2f}, n={len(high_indices)})'
                    else:  # Raw Range
                        label = f'Outside Range ({self.range_low:.1f}-{self.range_high:.1f}, n={len(high_indices)})'
                    
                    # Add red X markers
                    ax.scatter(high_x, high_y, marker='x', c='red', s=100, linewidths=3, label=label)
                    
                    # Add legend
                    ax.legend(loc='upper right', fontsize=8)
        
        # Add colorbar and track it for removal later
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(self.get_current_metric())
        self.colorbars.append(cbar)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
    def update_statistics_display(self):
        """Update statistics display for the currently selected event and metric."""
        event_col = self.get_current_event()
        metric = self.get_current_metric()
        
        if not event_col or not metric:
            self.stats_text.setText("Select an event and metric to view statistics.")
            return
        
        if "spatial_metrics" not in self.statistics:
            self.stats_text.setText("No spatial statistics available.")
            return
        
        spatial_metrics = self.statistics["spatial_metrics"]
        if not spatial_metrics or "nnr_clustering" not in spatial_metrics:
            self.stats_text.setText("No NNR clustering results found.")
            return
        
        nnr_results = spatial_metrics["nnr_clustering"]
        key = f"{event_col}::{metric}"
        
        if key not in nnr_results:
            self.stats_text.setText(f"No NNR results for {event_col} - {metric}")
            return
        
        data = nnr_results[key]
        
        # Build formatted text for current selection
        text_lines = []
        text_lines.append(f"NNR CLUSTERING ANALYSIS")
        text_lines.append(f"Event: {event_col}")
        text_lines.append(f"Metric: {metric}")
        text_lines.append("=" * 40)
        text_lines.append("")
        text_lines.append("METHODOLOGY:")
        
        # Dynamic methodology description based on current method
        if self.current_method == "Z-score":
            text_lines.append(f"• High-activity cells: z-score ≥ {self.z_threshold} (marked with red X)")
        elif self.current_method == "Quantile Range":
            text_lines.append(f"• Marked cells: outside {self.quantile_low:.1%}-{self.quantile_high:.1%} range (red X)")
        else:  # Raw Range
            text_lines.append(f"• Marked cells: outside {self.range_low:.1f}-{self.range_high:.1f} range (red X)")
            
        text_lines.append("• NNR = NN_distance(marked) / NN_distance(all)")
        text_lines.append("• Lower NNR = More clustered")
        text_lines.append("")
        text_lines.append("RESULTS:")
        text_lines.append(f"Original NNR:      {data['original']:.4f}")
        text_lines.append(f"Shuffled Mean:      {data['shuffled_mean']:.4f}")
        text_lines.append(f"Shuffled Std:       {data['shuffled_std']:.4f}")
        text_lines.append(f"Z-Score:           {data['z_score']:.4f}")
        text_lines.append(f"High Activity Cells: {data['n_high_cells']} (red X markers)")
        text_lines.append(f"NN Distance (All):  {data['nn_all_cells']:.4f}")
        text_lines.append("")
        
        # Interpretation using current p-value threshold
        z_score = data['z_score']
        z_threshold = self.get_z_threshold_for_pvalue(self.current_pvalue)
        
        if abs(z_score) < z_threshold:
            interpretation = "Not significantly different from random"
            significance = ""
        elif z_score < -z_threshold:
            interpretation = "MORE CLUSTERED than random"
            significance = " (SIGNIFICANT)"
        else:
            interpretation = "MORE DISPERSED than random" 
            significance = " (SIGNIFICANT)"
        
        text_lines.append("INTERPRETATION:")
        text_lines.append(f"{interpretation}{significance}")
        text_lines.append(f"Significance level: p < {self.current_pvalue}")
        
        if abs(z_score) >= z_threshold:
            text_lines.append(f"Z-score threshold: ±{z_threshold:.2f}")
        
        self.stats_text.setText("\n".join(text_lines))
        
    def get_z_threshold_for_pvalue(self, pvalue):
        """Convert p-value to z-score threshold for two-tailed test."""
        if pvalue == 0.01:
            return 2.58  # 99% confidence
        elif pvalue == 0.10:
            return 1.65  # 90% confidence
        else:  # pvalue == 0.05
            return 1.96  # 95% confidence
        
    def show_full_statistics(self):
        """Show full statistics in a separate popup dialog window."""
        if not hasattr(self, 'statistics') or not self.statistics:
            return
            
        full_stats_window = FullStatisticsWindow(self.statistics, self)
        full_stats_window.exec_()  # Use exec_() for modal dialog or show() for non-modal
        
    def closeEvent(self, event):
        """Handle window close event."""
        self.window_closed.emit()
        event.accept()
        
    def show_window(self):
        """Show the visualization window."""
        self.show()
        self.raise_()
        self.activateWindow()


class FullStatisticsWindow(QDialog):
    """Separate popup dialog window to show full NNR statistics."""
    
    def __init__(self, statistics, parent=None):
        super().__init__(parent)
        self.statistics = statistics
        self.setModal(False)  # Allow interaction with parent window
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the full statistics window."""
        self.setWindowTitle("Full NNR Clustering Statistics")
        self.setGeometry(150, 150, 900, 700)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Complete Nearest Neighbor Ratio (NNR) Clustering Analysis")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "This analysis identifies cells with high activity (z-score ≥ 1.5) and calculates "
            "their spatial clustering using Nearest Neighbor Ratio. Lower NNR values indicate "
            "more spatial clustering than random."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("margin: 10px; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(desc_label)
        
        # Statistics text area
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setStyleSheet("font-family: 'Courier New', monospace; font-size: 10px;")
        
        # Generate full statistics text
        stats_text.setText(self.generate_full_statistics_text())
        layout.addWidget(stats_text)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        
        self.setLayout(layout)
        
    def generate_full_statistics_text(self):
        """Generate the complete statistics text."""
        if "spatial_metrics" not in self.statistics:
            return "No spatial statistics available."
        
        spatial_metrics = self.statistics["spatial_metrics"]
        if not spatial_metrics or "nnr_clustering" not in spatial_metrics:
            return "No NNR clustering results found."
        
        nnr_results = spatial_metrics["nnr_clustering"]
        
        # Build formatted text (same as before)
        text_lines = []
        text_lines.append("=" * 80)
        text_lines.append("NEAREST NEIGHBOR RATIO (NNR) CLUSTERING ANALYSIS")
        text_lines.append("=" * 80)
        text_lines.append("")
        text_lines.append("Lower NNR = More Clustering | Higher NNR = More Dispersed")
        text_lines.append("Z-score: Negative = More clustered than random, Positive = More dispersed")
        text_lines.append("")
        
        # Group by event
        events = {}
        for key, data in nnr_results.items():
            event, metric = key.split("::")
            if event not in events:
                events[event] = {}
            events[event][metric] = data
        
        for event_name in sorted(events.keys()):
            text_lines.append(f"{'='*60}")
            text_lines.append(f"EVENT: {event_name}")
            text_lines.append(f"{'='*60}")
            text_lines.append("")
            
            for metric_name in ["Average Amplitude", "Frequency", "Total Amplitude"]:
                if metric_name in events[event_name]:
                    data = events[event_name][metric_name]
                    text_lines.append(f"  {metric_name}:")
                    text_lines.append(f"    Original NNR:      {data['original']:.4f}")
                    text_lines.append(f"    Shuffled Mean:      {data['shuffled_mean']:.4f}")
                    text_lines.append(f"    Shuffled Std:       {data['shuffled_std']:.4f}")
                    text_lines.append(f"    Z-Score:           {data['z_score']:.4f}")
                    text_lines.append(f"    High Activity Cells: {data['n_high_cells']}")
                    text_lines.append(f"    NN Distance (All):  {data['nn_all_cells']:.4f}")
                    
                    # Interpretation
                    z_score = data['z_score']
                    if abs(z_score) < 1.96:
                        interpretation = "Not significantly different from random"
                    elif z_score < -1.96:
                        interpretation = "SIGNIFICANTLY MORE CLUSTERED than random"
                    else:
                        interpretation = "SIGNIFICANTLY MORE DISPERSED than random"
                    
                    text_lines.append(f"    Interpretation:     {interpretation}")
                    text_lines.append("")
            
            text_lines.append("")
        
        # Summary statistics
        text_lines.append(f"{'='*60}")
        text_lines.append("SUMMARY")
        text_lines.append(f"{'='*60}")
        
        significant_clustered = []
        significant_dispersed = []
        
        for key, data in nnr_results.items():
            z_score = data['z_score']
            if z_score < -1.96:
                significant_clustered.append(key.replace("::", " - "))
            elif z_score > 1.96:
                significant_dispersed.append(key.replace("::", " - "))
        
        text_lines.append(f"Significantly Clustered ({len(significant_clustered)}):")
        for item in significant_clustered:
            text_lines.append(f"  • {item}")
        
        text_lines.append("")
        text_lines.append(f"Significantly Dispersed ({len(significant_dispersed)}):")
        for item in significant_dispersed:
            text_lines.append(f"  • {item}")
        
        return "\n".join(text_lines)


def calculate_z_score(original_values: np.ndarray, shuffled_array: np.ndarray) -> float:
    """Calculate z-score comparing original to shuffled distribution."""
    original_mean = np.mean(original_values)
    shuffled_mean = np.mean(shuffled_array)
    shuffled_std = np.std(shuffled_array)
    
    if shuffled_std == 0:
        return 0.0
    
    z_score = (original_mean - shuffled_mean) / shuffled_std
    return z_score
