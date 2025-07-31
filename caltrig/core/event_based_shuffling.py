"""
Event-based shuffling utilities for calcium transient analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .event_based_utility import extract_event_based_data
from .shuffling import permute_itis_to_start_indices
from ..gui.pop_up_messages import ProgressWindow
from ..gui.sda_widgets import _precalculate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QMenuBar, QAction, QStyle, QApplication
from PyQt5.QtCore import pyqtSignal


def event_based_shuffle_analysis(
    session,
    selected_cells: List[int],
    event_type: str,
    window_size: int,
    lag: int,
    num_subwindows: int,
    num_shuffles: int
) -> Optional['VisualizeEventBasedShuffling']:
    """
    Perform event-based shuffling analysis on calcium transient data.
    
    This function reuses the window size preview logic for extracting event-based data
    and applies temporal shuffling similar to the 3D visualization approach.
    
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
        
    Returns:
    --------
    VisualizeEventBasedShuffling or None
        Visualization window object that can be displayed and stored in exploration widgets
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
    
    # Get ITI data for temporal shuffling (reusing shuffling.py logic)
    try:
        frame_start_dict, itis_dict = session.get_transient_frames_iti_dict(selected_cells)
        
        # Validate that we have ITI data for our selected cells
        if not itis_dict:
            print("Error: No ITI data available for selected cells")
            return None
            
        # Check if any selected cells have empty ITI data
        empty_cells = [cell_id for cell_id in selected_cells if cell_id not in itis_dict or len(itis_dict[cell_id]) == 0]
        if empty_cells:
            # Filter out cells with no data
            selected_cells = [cell_id for cell_id in selected_cells if cell_id in itis_dict and len(itis_dict[cell_id]) > 0]
            if not selected_cells:
                print("Error: No cells with valid transient data found")
                return None
            
    except Exception as e:
        print(f"Error getting ITI data: {e}")
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
                # Temporally shuffle the transients using existing function
                shuffled_frame_start = permute_itis_to_start_indices(itis_dict)
                
                # Create a temporary session copy with shuffled transient starts
                # We need to modify the session's transient data temporarily
                original_transient_starts = {}
                precalculated_values = _precalculate(session)
                
                # Store original values and replace with shuffled ones
                for cell_id in selected_cells:
                    if cell_id in precalculated_values['transient_info']:
                        original_transient_starts[cell_id] = precalculated_values['transient_info'][cell_id]['frame_start'].copy()
                        if cell_id in shuffled_frame_start:
                            # Make sure we don't have any index issues
                            shuffled_starts = shuffled_frame_start[cell_id]
                            if len(shuffled_starts) > 0:
                                precalculated_values['transient_info'][cell_id]['frame_start'] = np.array(shuffled_starts)
                            
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
                # Restore original values before continuing
                for cell_id in selected_cells:
                    if cell_id in original_transient_starts:
                        precalculated_values['transient_info'][cell_id]['frame_start'] = original_transient_starts[cell_id]
                continue
            
            # Restore original values
            for cell_id in selected_cells:
                if cell_id in original_transient_starts:
                    precalculated_values['transient_info'][cell_id]['frame_start'] = original_transient_starts[cell_id]
            
            shuffled_results.append(shuffled_data)
            
    except Exception as e:
        print(f"Error during shuffling: {e}")
        progress_window.close()
        return None
    
    finally:
        progress_window.close()
    
    # Calculate statistics
    statistics = calculate_event_shuffling_statistics(original_data, shuffled_results)
    
    # Create and return visualization window
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
            'event_indices': events.tolist()
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


def calculate_event_shuffling_statistics(original_data: Dict, shuffled_data_list: List[Dict]) -> Dict[str, Any]:
    """
    Calculate statistical measures comparing original and shuffled event-based data.
    Uses the same approach as existing shuffling functions: z-scores based on mean and std of shuffled distribution.
    
    Parameters:
    -----------
    original_data : Dict
        Original event-based data (cell_id -> DataFrame)
    shuffled_data_list : List[Dict]
        List of shuffled event-based data results
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing statistical measures for each cell and metric
    """
    
    if not shuffled_data_list:
        return {}
    
    statistics = {}
    
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
        self.metric_dropdown.addItems(["Average Amplitude", "Frequency", "Total Amplitude"])
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
        for cell_id in self.parameters['selected_cells']:
            self.cell_dropdown.addItem(f"Cell {cell_id}")
            
    def populate_event_dropdown(self):
        """Populate the event selection dropdown."""
        self.event_dropdown.clear()
        num_events = len(self.parameters['event_indices'])
        for i in range(min(num_events, 10)):  # Show first 10 events
            event_idx = self.parameters['event_indices'][i]
            self.event_dropdown.addItem(f"Event {i+1} (Frame {event_idx})")
            
    def get_current_cell_id(self):
        """Get the currently selected cell ID."""
        current_text = self.cell_dropdown.currentText()
        if current_text:
            return int(current_text.split()[1])  # Extract cell ID from "Cell X"
        return None
        
    def get_current_event_index(self):
        """Get the currently selected event index."""
        current_index = self.event_dropdown.currentIndex()
        if current_index >= 0 and current_index < len(self.parameters['event_indices']):
            return current_index
        return None
        
    def get_current_metric(self):
        """Get the currently selected metric."""
        return self.metric_dropdown.currentText()
        
    def on_selection_changed(self):
        """Handle dropdown selection changes."""
        self.plot_current_selection()
        
    def plot_current_selection(self):
        """Plot histogram for shuffled data with original value as a dashed vertical line."""
        cell_id = self.get_current_cell_id()
        event_idx = self.get_current_event_index()
        selected_metric = self.get_current_metric()
        
        if cell_id is None or event_idx is None or not selected_metric:
            return
            
        # Clear previous plot
        self.ax.clear()
        
        # Get original value for this cell, event, and metric
        original_value = None
        if cell_id in self.original_data:
            cell_data = self.original_data[cell_id]  # This is a pandas DataFrame
            event_col_name = f"Event {event_idx + 1}"
            if event_col_name in cell_data.columns and selected_metric in cell_data.index:
                original_value = cell_data.loc[selected_metric, event_col_name]
                
        # Get shuffled values for this cell, event, and metric
        shuffled_values = []
        for shuffle_result in self.shuffled_data:
            if cell_id in shuffle_result:
                cell_data = shuffle_result[cell_id]  # This is also a pandas DataFrame
                event_col_name = f"Event {event_idx + 1}"
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
        self.ax.set_title(f"{selected_metric}\nCell {cell_id}, Event {event_idx + 1}")
        self.ax.set_xlabel(f"{selected_metric} Value")
        self.ax.set_ylabel("Frequency")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
            
        # Update statistics display
        self.update_statistics_display(cell_id, event_idx, selected_metric)
        
        # Refresh the plot
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_statistics_display(self, cell_id, event_idx, selected_metric):
        """Update the statistics label with current selection stats."""
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
        
    def closeEvent(self, event):
        """Handle window close event."""
        self.window_closed.emit()
        event.accept()
        
    def show_window(self):
        """Show the visualization window."""
        self.show()
        self.raise_()
        self.activateWindow()


def calculate_z_score(original_values: np.ndarray, shuffled_array: np.ndarray) -> float:
    """Calculate z-score comparing original to shuffled distribution."""
    original_mean = np.mean(original_values)
    shuffled_mean = np.mean(shuffled_array)
    shuffled_std = np.std(shuffled_array)
    
    if shuffled_std == 0:
        return 0.0
    
    z_score = (original_mean - shuffled_mean) / shuffled_std
    return z_score
