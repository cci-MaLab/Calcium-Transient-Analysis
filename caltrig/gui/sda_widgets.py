# Spatial distribution analysis
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import xarray as xr
from .pop_up_messages import print_error
from matplotlib import cm
from typing import List, Optional
from pyvistaqt import QtInteractor
import pyvista as pv
import colorcet as cc
from concurrent.futures import ProcessPoolExecutor
from ..core.backend import DataInstance
import time
import math
import pickle

class CurrentVisualizationData():
    def __init__(self, data, max_height, min_height, start_frame, end_frame, x_start, x_end, y_start, y_end, scaling_factor=10):
        self.data = data
        self.max_height = round_away_from_zero(max_height, 1)
        self.min_height = round_away_from_zero(min_height, 1)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.scaling_factor = scaling_factor
        self.points = {"x": np.array([]), "y": np.array([])}
        self.cell_id_to_index = {}
        self.cell_id_coords = {}

    def get_3d_data(self, frame) -> dict:
        # Returns a dictionary containing the data for surface plot and the points of the centroids
        frame = (self.data[frame-self.start_frame] * self.scaling_factor).astype(np.float32)
        # Get the x, y and z values for the points
        points_coords = np.array([self.points["x"], self.points["y"], frame[self.points["y"], self.points["x"]]], dtype=np.float32).T if len(self.points["x"]) > 0 else np.array([], [], [])
        return {"frame": frame, "points_coords": points_coords}
    
    def update_points_list(self, points):
        self.cell_id_to_index = {cell_id: i for i, cell_id in enumerate(points.keys())}
        cell_ids = points.keys()
        points = points.values()
        x_coords, y_coords = zip(*points)
        # So it aligns with the data and PyVista convention we need to subtract the start values and flip the y and x values
        x_coords, y_coords = switch_to_3d_coordinates(x_coords, y_coords, self.x_start, self.y_start, self.x_end, self.y_end)     
        self.points = {"x": x_coords, "y": y_coords}
        # Keep a local dict cell_id to coords
        self.cell_id_coords = {cell_id: (x, y) for cell_id, x, y in zip(cell_ids, x_coords, y_coords)}

    def get_selected_points(self, selected_cells):
        # We need to create an array of the same size as the cell_ids but for rgb values. If 0 then red, if 1 then green
        selected_points = np.zeros(len(self.cell_id_to_index), np.uint8)
        for cell_id in selected_cells:
            selected_points[self.cell_id_to_index[cell_id]] = 1
        return selected_points
    
    def in_range(self, frame):
        return frame >= self.start_frame and frame <= self.end_frame-1
    
    def get_ranges(self):
        return [self.x_start, self.x_end, self.y_start, self.y_end, 0, self.max_height]
    
    def get_shape(self):
        return {"x": self.x_end - self.x_start, "y": self.y_end - self.y_start}
    
    def get_extent(self):
        x = self.data.shape[1]
        y = self.data.shape[2]
        z = self.max_height * self.scaling_factor
        return {"x": x, "y": y, "z": z}
    
    def __mul__(self, factor):
        self.scaling_factor = factor
        return self

    __rmul__ = __mul__

def round_away_from_zero(value, base=10):
    if value > 0:
        return math.ceil(value / base) * base
    else:
        return math.floor(value / base) * base

def switch_to_3d_coordinates(x, y, x_start=0, y_start=0, x_end=None, y_end=None):
    """
    The PyVista coordinates begin in the bottom left corner, while the 2d plane
    coordinates begin in the top left corner. Therefore, we need to first subtract
    the start values if they exist and then flip x and y values.
    """

    # If x or y are a list we need to convert them to numpy arrays
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x).astype(int)
    if isinstance(y, list) or isinstance(y, tuple):
        y = np.array(y).astype(int)
    
    x_start, y_start = int(x_start), int(y_start)
    
    x, y = x - x_start, y - y_start
    # Sometimes the value can end up in the peripheral area, we need to make sure it is within the bounds
    if x_end is not None:
        x = np.clip(x, 0, x_end-x_start-1)
    if y_end is not None:
        y = np.clip(y, 0, y_end-y_start-1)    
    return x, y

def base_visualization(serialized_data, start_frame=0, end_frame=50):
    unserialized_data = pickle.loads(serialized_data)
    
    # Extract the necessary data
    session = unserialized_data.get("session")
    precalculated_values = unserialized_data.get("precalculated_values", None)
    kwargs = unserialized_data.get("kwargs", {})
    data_type = kwargs.get("data_type", "C")
    window_size = kwargs.get("window_size", 1)
    cells_to_visualize = kwargs.get("cells_to_visualize", "All Cells")
    smoothing_size = kwargs.get("smoothing_size", 1)
    smoothing_type = kwargs.get("smoothing_type", "mean")
    cumulative = kwargs.get("cumulative", False)
    average = kwargs.get("average", False)
    normalize = kwargs.get("normalize", False)
    
    ids = session.get_cell_ids(cells_to_visualize)
    if window_size != 1:
        # We'll first check if the visualization data exists within precalculated, otherwise we'll calculate it
        name = f"{data_type}_window_{window_size}"
        name = name + "_cumulative" if cumulative else name
        name = name + "_normalized" if normalize else name
        name = name + "_averaged" if average else name
        if name in precalculated_values:
            xr_data = precalculated_values[name]
            signal = xr_data.sel(unit_id=ids).sel(frame=slice(start_frame, end_frame-1))
        else:
            xr_data = calculate_windowed_data(session, precalculated_values, data_type, window_size,
                                              cumulative=cumulative, normalize=normalize, average=average, name=name)
            precalculated_values[name] = xr_data
            signal = xr_data.sel(unit_id=ids).sel(frame=slice(start_frame, end_frame-1))
    else:
        if data_type in session.data:
            signal = session.data[data_type].sel(frame=slice(start_frame, end_frame-1)) # -1 since it is inclusive
        else:
            signal = precalculated_values[data_type].sel(frame=slice(start_frame, end_frame-1)) # -1 since it is inclusive
        signal = signal.sel(unit_id=ids)
    if smoothing_size != 1:
        if smoothing_type == "mean":
            signal = signal.rolling(min_periods=1, frame=smoothing_size, center=True).mean()
    signal = signal.values
    A = session.data["A"].sel(unit_id=ids).values
    A_flat = A.sum(axis=0)

    # Only use subset of A where values are positive, trim out 0 areas around the edges
    x_axis_sum = np.sum(A_flat, axis=0)
    y_axis_sum = np.sum(A_flat, axis=1)
    # Find first and last non-zero index
    x_axis_indices = np.where(x_axis_sum > 0)[0]
    y_axis_indices = np.where(y_axis_sum > 0)[0]
    x_start, x_end = x_axis_indices[0], x_axis_indices[-1]
    y_start, y_end = y_axis_indices[0], y_axis_indices[-1]
    A = A[:, y_start:y_end, x_start:x_end]

    Y = np.tensordot(signal, A, axes=([0], [0])).swapaxes(1, 2) # In order to maintain parity with the 2D visualization
    max_height = Y.max()
    min_height = Y.min()
    #Since we swapped the axes we need to swap the x and y values
    x_start, x_end, y_start, y_end = y_start, y_end, x_start, x_end
    
    CV = CurrentVisualizationData(Y, max_height, min_height, start_frame, end_frame, x_start, x_end, y_start, y_end)
    centroids = session.centroids_max
    # Include only the ones that rea in ids
    centroids = {id: centroids[id] for id in ids}
    CV.update_points_list(centroids)

    return CV

def calculate_windowed_data(session, precalculated_values, data_type, window_size, cumulative=False, normalize=False, average=False, name=""):
    E = session.data['E']
    unit_ids = E.unit_id.values

    if "C" in data_type:
        data_type = "C"
    elif "DFF" in data_type:
        data_type = "DFF"
    elif "Transient Count" in data_type:
        data_type = "Transient Count"
    else:
        raise ValueError("Data type not recognized")

    data = []
    for unit_id in unit_ids:
        if unit_id not in precalculated_values['transient_info']:
            data.append(np.zeros(E.shape[1]))
        else:
            unit_id_info = precalculated_values['transient_info'][unit_id]
            # Run through the unit_id_info and allocate the values to the respective bins
            no_of_bins = int(np.ceil(E.shape[1]/window_size))
            window_bins = [[] for _ in range(no_of_bins)]
            for start, c_val, dff_val in zip(unit_id_info['frame_start'], unit_id_info['C_values'], unit_id_info['DFF_values']):
                match data_type:
                    case "C":
                        window_bins[start//window_size].append(c_val)
                    case "DFF":
                        window_bins[start//window_size].append(dff_val)
                    case "Transient Count":
                        window_bins[start//window_size].append(1)
            # Replace any empty lists with 0
            for i, window_bin in enumerate(window_bins):
                if not window_bin:
                    window_bins[i] = [0]
            
            # To avoid weird things with average and count, we'll just sum the values
            if "Transient Count" == data_type:
                window_bins = [[np.sum(bin)] for bin in window_bins]

            # First Normalize the values
            if normalize:
                match data_type:
                    case "C":
                        total = unit_id_info['C_total']
                    case "DFF":
                        total = unit_id_info['DFF_total']
                    case "Transient Count":
                        total = np.sum(window_bins)
                for i in range(len(window_bins)):
                    for j in range(len(window_bins[i])):
                        window_bins[i][j] /= total
            
            if average:
                for i in range(len(window_bins)):
                    window_bins[i] = np.mean(window_bins[i])
            else:
                # Just sum the values
                for i in range(len(window_bins)):
                    window_bins[i] = np.sum(window_bins[i])
            window_bins = np.array(window_bins)

            if cumulative:
                window_bins = np.cumsum(window_bins)

            # Now convert the window_bins to the correct shape
            window_bins = np.repeat(window_bins, window_size)
            window_bins = window_bins[:E.shape[1]]
        
            data.append(window_bins)

    data = np.array(data)
    return xr.DataArray(data, coords=[unit_ids, np.arange(E.shape[1])], dims=['unit_id', 'frame'], name=name)

def calculate_single_value_windowed_data(session, precalculated_values, readout, window_size, name=""):
    unit_ids = list(precalculated_values['transient_info'].keys())
    E = session.data['E']
    no_of_bins = int(np.ceil(E.shape[1]/window_size))

    data = []
    for unit_id in unit_ids:
        unit_id_info = precalculated_values['transient_info'][unit_id]
        # Run through the unit_id_info and allocate the values to the respective bins
        window_bins = [[] for _ in range(no_of_bins)]
        for start, c_val, dff_val in zip(unit_id_info['frame_start'], unit_id_info['C_values'], unit_id_info['DFF_values']):
            match readout:
                case "Event Count Frequency":
                    window_bins[start//window_size].append(1)
                case "Average DFF Peak" | "Total DFF Peak":
                    window_bins[start//window_size].append(dff_val)
        # Replace any empty lists with 0
        for i, window_bin in enumerate(window_bins):
            if not window_bin:
                window_bins[i] = [0]
        
        # To avoid weird things with average and count, we'll just sum the values
        if "Event Count Frequency" == readout or "Total DFF Peak" == readout:
            window_bins = [[np.sum(bin)] for bin in window_bins]
    
        elif "Average DFF Peak" == readout:
            for i in range(len(window_bins)):
                window_bins[i] = np.mean(window_bins[i])
        window_bins = np.array(window_bins).flatten()
    
        data.append(window_bins)

    data = np.array(data)
    return xr.DataArray(data, coords=[unit_ids, np.arange(no_of_bins)], dims=['unit_id', 'window'], name=name)

def local_fpr(a_data: xr.DataArray, b_data: xr.DataArray, fpr: str):
    # Assign fpr to a specific func
    match fpr:
        case "B":
            func = lambda a, b: b
        case "B-A":
            func = lambda a, b: b - a
        case "(B-A)²":
            func = lambda a, b: (b - a) ** 2
        case "(B-A)/A":
            func = lambda a, b: np.divide(b - a, a, out=np.zeros_like(a, dtype=float), where=a != 0)
        case "|(B-A)/A|":
            func = lambda a, b: np.abs(np.divide(b - a, a, out=np.zeros_like(a, dtype=float), where=a != 0))
        case "B/A":
            func = lambda a, b: np.divide(b, a, out=np.zeros_like(a, dtype=float), where=a != 0)
        case _:
            raise ValueError(f"Unknown FPR type: {fpr}")
    
    result = xr.apply_ufunc(func, a_data, b_data)

    return result

def calculate_fpr(a_cells, b_cells, sv_win_data, fpr, sv_win_data_base=None, anchor=False):
    a_to_b_fpr = {}
    for a_cell in a_cells:
        for b_cell in b_cells:
            if a_cell == b_cell:
                continue
            if anchor:
                fpr_result = local_fpr(sv_win_data_base.sel(unit_id=a_cell), sv_win_data.sel(unit_id=b_cell), fpr)
            else:
                fpr_result = local_fpr(sv_win_data.sel(unit_id=a_cell), sv_win_data.sel(unit_id=b_cell), fpr)
            a_to_b_fpr[(a_cell, b_cell)] = fpr_result
    
    return a_to_b_fpr

def add_distance_to_fpr(fpr, session, a_cells, b_cells, shuffle=False):
    # We need to calculate the distance between the cells
    centroids = session.centroids
    new_centroids = {}
    if shuffle:
        # We need to anchor the a cell positions and only shuffle the b cell positions
        for a_cell in a_cells:
            selected_b_cells = list(b_cells)
            selected_b_cells.remove(a_cell)
            b_positions_shuffled = [centroids[cell] for cell in selected_b_cells]
            np.random.shuffle(b_positions_shuffled)
            new_centroids[a_cell] = {cell: b_positions_shuffled[i] for i, cell in enumerate(selected_b_cells)}
            new_centroids[a_cell][a_cell] = centroids[a_cell]
    else:
        new_centroids = {cell: centroids for cell in a_cells}

    distances = {}
    for a_cell, b_cell in fpr.keys():
        a_x, a_y = new_centroids[a_cell][a_cell]
        b_x, b_y = new_centroids[a_cell][b_cell]
        distance = np.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)
        distances[(a_cell, b_cell)] = distance
    
    fpr_with_distance = {}
    for (a_cell, b_cell), value in fpr.items():
        fpr_with_distance[(a_cell, b_cell)] = (value, distances[(a_cell, b_cell)])
    
    return fpr_with_distance

def gaussian_2d(shape=(7,7), sigma=1.5):
    """Generate a 2D Gaussian distribution centered at 1 with falloff."""
    m, n = [(s-1)/2 for s in shape]  # Center coordinates
    y, x = np.ogrid[-m:m+1, -n:n+1]
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gaussian / gaussian.max()  # Normalize to ensure center is 1     

def _forward_fill(y):
        prev = np.arange(len(y))
        prev[y == 0] = 0
        prev = np.maximum.accumulate(prev)
        return y[prev]

def _precalculate(session):
    precalculated_values = {}
    E = session.data['E']
    C = session.data['C']
    DFF = session.data['DFF']

    # Unit ids
    unit_ids = E.unit_id.values

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
                
class VisualizationWidget(QtInteractor):
    point_signal = pyqtSignal(int, int)
    def __init__(self, session: DataInstance, executor: Optional[ProcessPoolExecutor], chunk_size=50, visualization_generator=base_visualization, parent=None):
        super().__init__(parent)
        """
        Main widget for the PyVista for 3D visualization of any specified typed of data.

        Parameters
        ----------
        session : DataInstance
            The session object that contains all the relevant session data from the parent class.
        executor : ProcessPoolExecutor
            The executor to use to setup separate processes for loading the next chunk of data.
        chunk_length : int, optional
            The nearest multiple of frames to chunk the data into, by default 50. This to compensate for 
            the user potentially jumping to a random frame and for clarity we would like to start chunking
            at the nearest multiple of specified chunk_length.
        chunk_size : int, optional
            The size of the chunk to visualize, by default 50. We load 50 frames at a time and pre-load
            the next chunk when the user has reached the end of the current chunk. This is too avoid slow downs
            due to lazy loading of the data from disk.
        visualization_generator : function, optional
            The function to generate the visualization data, by default base_visualization.
        parent : QWidget, optional
            The parent widget, by default None.
        """
        self.chunk_size = chunk_size
        self.executor = executor
        self.processes = True if self.executor is not None else False
        self.session = session
        self.scaling_factor = 10
        self.current_frame = 0
        self.kwargs_func = {}
        self.selected_cells = []
        self.update_precalculated_values()
        # We need to serialize the data so we can speed up the process of submit a process to the executor
        self._update_serialize_data()
        
        self.visualization_generator = visualization_generator
        self.visualization_data = visualization_generator(self.serialized_data, start_frame=0, end_frame=50) * self.scaling_factor
        if self.processes:
            self.visualization_data_buffered = self.chunk_load(self.serialized_data, start_frame=50, end_frame=100)
        else:
            self.visualization_data_buffered = None
        # We need to keep track the cell id to index position
        self.cell_id_to_index = {cell_id: i for i, cell_id in enumerate(session.centroids_max.keys())}

        # All parameters for the visualization itself
        self.axes = None
        self.points_3d = None # The actual mlab points3d object
        self.points_coords = None # The coordinates of the points
        self.current_shape = self.visualization_data.data[0].shape
        self.parent = parent
        self.cells = None

        # Instantiate PyVista scene
        self.scalar_range = (-0.1, 1)
        self.ranges_changed = False
        self.background_color = 'black'
        self.points_3d = None
        self.cmap = "fire"
        self.arrows = {"params": ""}
        self.populate_3D_scene()

    def update_precalculated_values(self):
        self.precalculated_values = _precalculate(self.session)

    def populate_3D_scene(self):
        shape = self.visualization_data.get_shape()
        x, y = np.meshgrid(np.arange(shape["x"], dtype=np.float32), np.arange(shape["y"], dtype=np.float32))
        data_3d = self.visualization_data.get_3d_data(self.current_frame)
        frame, points_coords = data_3d["frame"], data_3d["points_coords"]
        self.grid = pv.StructuredGrid(x, y, frame)
        self.grid["scalars"] = frame.ravel(order='F')
        self.add_mesh(self.grid, lighting='flat', clim=self.scalar_range, scalar_bar_args=None, pickable=False)
        self.points_3d = pv.PolyData(points_coords)
        self.points_3d["colors"] = self.visualization_data.get_selected_points(self.selected_cells)
        self.add_mesh(self.points_3d, scalars="colors", render_points_as_spheres=False, point_size=10, cmap=['red', 'green'], scalar_bar_args=None)
        self.enable_point_picking(callback=self.receive_click, pickable_window=self.points_3d, show_point=False, show_message="Right Click or press 'P' to select point/cell", left_click=True)
        self._picking_text.GetTextProperty().SetColor(1,1,1)
        self.change_colormap(self.cmap)

        
    def set_data(self, visualization_data):
        self.visualization_data = visualization_data * self.scaling_factor
        self.readjust_ranges()
        self.reset_grid()
        self.cell_id_to_index = self.visualization_data.cell_id_to_index

    def readjust_ranges(self):
        min_height = self.visualization_data.min_height if self.visualization_data.min_height < self.scalar_range[0] else self.scalar_range[0]
        max_height = self.visualization_data.max_height if self.visualization_data.max_height > self.scalar_range[1] else self.scalar_range[1]
        new_scalar_range = (min(min_height, -0.1), max_height) # 0.01 to avoid the grid overlapping with the plane
        if new_scalar_range != self.scalar_range:
            self.scalar_range = new_scalar_range
            self.ranges_changed = True
    
    def chunk_load(self, serialized_data, start_frame, end_frame):
        return self.executor.submit(self.visualization_generator, serialized_data, start_frame=start_frame, end_frame=end_frame)

    def reset_grid(self):
        self.clear()
        self.disable_picking()
        self.populate_3D_scene()
    
    def set_frame(self, frame=0):
        self.check_frame(frame)  
        # Now we can set the frame
        frame_3d_data = self.visualization_data.get_3d_data(frame)
        frame_scalars = frame_3d_data["frame"].ravel(order='F')
        points_coords = frame_3d_data["points_coords"]
        self.grid.points[:,2] = frame_scalars
        self.grid["scalars"] = frame_scalars
        self.points_3d.points = points_coords
        self.render()

        self.current_frame = frame
        if self.ranges_changed:
            self.change_colormap(self.cmap)
            self.ranges_changed = False

    def check_frame(self, frame):
        """
        We are checking whether the frame is within the current or the next chunk:
         - If is within the current chunk we do nothing.
         - If it is within the next chunk we swap the current chunk with the next chunk and update the next chunk.
         - If it is outside of the next chunk we update both current and next chunk.

        Parameters:
        frame : int
            The frame to check.
        """
        if not self.visualization_data.in_range(frame):
            chunk_start = frame - frame % self.chunk_size
            next_chunk_start = chunk_start + self.chunk_size
            if self.visualization_data_buffered is None: # The next chunk wasn't instantiated. Update both
                self.visualization_data = self.visualization_generator(self.serialized_data, start_frame=chunk_start, end_frame=next_chunk_start) * self.scaling_factor
                if self.processes:
                    self.visualization_data_buffered = self.chunk_load(self.serialized_data, next_chunk_start, next_chunk_start+self.chunk_size)
            else:
                while not self.visualization_data_buffered.done():
                    time.sleep(0.01)
                # Time the following code
                self.visualization_data_buffered = self.visualization_data_buffered.result() * self.scaling_factor
                if not self.visualization_data_buffered.in_range(frame): # Not in range of the next chunk, we need to update both
                    self.visualization_data = self.visualization_generator(self.serialized_data, start_frame=chunk_start, end_frame=next_chunk_start) * self.scaling_factor
                    self.visualization_data_buffered = self.chunk_load(self.serialized_data, next_chunk_start, next_chunk_start+self.chunk_size)
                else: # In range of the next chunk, we need to swap the current chunk with the next chunk and update the next chunk
                    self.visualization_data = self.visualization_data_buffered
                    self.visualization_data_buffered = self.chunk_load(self.serialized_data, next_chunk_start, next_chunk_start+self.chunk_size)
            self.readjust_ranges()

    def change_colormap(self, name):
        self.cmap = name
        plane_lut = pv.LookupTable(cc.cm[name], n_values=256)
        actual_min = self.scalar_range[0] * self.scaling_factor
        actual_max = self.scalar_range[1] * self.scaling_factor
        plane_lut.scalar_range = (actual_min, actual_max)

        n_ticks = 5
        ticks = np.linspace(actual_min, actual_max, n_ticks)
        plane_lut.annotations = {
            float(t): f"{(t / self.scaling_factor):.0f}"
            for t in ticks
        }

        mapper = None
        for actor_name, actor in self.actors.items():
            if "Grid" in actor_name:
                mapper = actor.GetMapper()
                mapper.SetLookupTable(plane_lut)
                mapper.SetScalarRange(actual_min, actual_max)
                break

        for name in list(self.scalar_bars.keys()):
            self.remove_scalar_bar(name)

        bar_actor = self.add_scalar_bar(
            title='Spike Intensity',
            color='white',
            shadow=True,
            n_labels=n_ticks,
            mapper=mapper,
        )

        bar_actor.DrawTickLabelsOff()

        shape = self.visualization_data.get_shape() 
        self.show_grid(
            bounds=(0, shape["x"], 0, shape["y"], actual_min, actual_max),
            axes_ranges=(0, shape["x"], 0, shape["y"], self.scalar_range[0], self.scalar_range[1]),
            color='white'
        )
        self.render()

    def change_func(self, func, **kwargs):
        if "scaling" in kwargs:
            self.scaling_factor = kwargs["scaling"]
        self.visualization_generator = func
        self.kwargs_func = kwargs
        chunk_start = self.current_frame - self.current_frame % self.chunk_size

        self._update_serialize_data()
        self.scalar_range = (-0.1, 1)
        self.set_data(self.visualization_generator(self.serialized_data, start_frame=chunk_start, end_frame=chunk_start+self.chunk_size))
        self.set_frame(self.current_frame)  

    def _update_serialize_data(self):
        self.serialized_data = pickle.dumps({"session": self.session, "precalculated_values": self.precalculated_values, "kwargs": self.kwargs_func})

    def update_selected_cells(self, cells):
        # Remove cells that are rejected
        cells = self.session.prune_rejected_cells(cells)
        self.selected_cells = cells
        self.points_3d["colors"] = self.visualization_data.get_selected_points(self.selected_cells)
        self.render()
    
    def receive_click(self, picked_point):
        if picked_point is None:
            return
        x, y, _ = picked_point
        # Add the start values to the x and y values
        x, y = int(x + self.visualization_data.x_start), int(y + self.visualization_data.y_start)
        self.point_signal.emit(x, y)

    def extract_cofiring_data(self, window_size, **kwargs):
        name = f"cofiring_{window_size}"
        name += "_shared_A" if kwargs["shareA"] else ""
        name += "_shared_B" if kwargs["shareB"] else ""
        name += kwargs["direction"]
        if name not in self.precalculated_values:
            self.change_cofiring_window(window_size, **kwargs)
        return self.precalculated_values[name]
    
    def change_cofiring_window(self, window_size, visualize=False, nums_to_visualize="Verified Cells",
                               **kwargs):
        
        name = f"cofiring_{window_size}"
        name += "_shared_A" if kwargs["shareA"] else ""
        name += "_shared_B" if kwargs["shareB"] else ""
        name += kwargs["direction"]
        unit_ids = self.session.get_cell_ids(nums_to_visualize, verified=True)

        if len(unit_ids) < 2:
            print_error("Not Enough Cells for Cofiring", extra_info="Make sure that you have at least two verified cells for cofiring.", severity=QMessageBox.Warning)
            return unit_ids

        if name not in self.precalculated_values:
            # Cells to Number
            cofiring_data_cells = {}
            # Number to Cells
            cofiring_data_number = {}

            verified_unit_ids = self.session.get_cell_ids("Verified Cells")
            
            for unit_id in verified_unit_ids:
                for unit_id_2 in verified_unit_ids:
                    if unit_id == unit_id_2:
                        continue
                    unit1_starts = self.precalculated_values['transient_info'][unit_id]['frame_start']
                    unit2_starts = self.precalculated_values['transient_info'][unit_id_2]['frame_start']
                    value = check_cofiring(unit1_starts, unit2_starts, window_size, **kwargs)
                    cofiring_data_cells[(unit_id, unit_id_2)] = value
                    if value not in cofiring_data_number:
                        cofiring_data_number[value] = []
                    cofiring_data_number[value].append((unit_id, unit_id_2))
            self.precalculated_values[name] = {"cells": cofiring_data_cells, "number": cofiring_data_number}

        if visualize:
            # This means the checkbox is checked and we need to visualize the data
            self.visualize_arrows(unit_ids, window_size, **kwargs)
        
        return self.precalculated_values[name]

    def remove_arrow(self, ids):
        if ids in self.arrows:
            self.remove_actor(self.arrows[ids])
            self.arrows.pop(ids)

    def visualize_arrows(self, nums_to_visualize, window_size, cofiring_nums=set(), cofiring_cells=set(), shareA=True, shareB=True, direction="bidirectional", **kwargs):
        
        name = f"cofiring_{window_size}"
        name += "_shared_A" if shareA else ""
        name += "_shared_B" if shareB else ""
        name += direction

        params = "_".join([str(val) for val in [window_size, shareA, shareB, direction]])

        self.remove_cofiring(check_params=params)
        self.arrows["params"] = params

        if name not in self.precalculated_values:
            self.change_cofiring_window(window_size, shareA=shareA, shareB=shareB, 
                                        nums_to_visualize=nums_to_visualize, direction=direction,
                                          **kwargs)
            return
        # We need to visualize the arrows
        cell_id_coords = self.visualization_data.cell_id_coords
        if len(cell_id_coords) == 0:
            return

        max_cofiring = max(self.precalculated_values[name]["number"].keys())
        colormap = cm.get_cmap("rainbow")
        visualized_arrows = []
        for id1 in nums_to_visualize:
            for id2 in nums_to_visualize:
                if (id1, id2) in self.precalculated_values[name]["cells"]:
                    value = self.precalculated_values[name]["cells"][(id1, id2)]
                else:
                    self.remove_arrow((id1, id2))
                    continue
                if id1 not in cell_id_coords or id2 not in cell_id_coords:
                    continue
                if value not in cofiring_nums and "all" not in cofiring_nums:
                    self.remove_arrow((id1, id2))
                    continue
                if id1 not in cofiring_cells and id2 not in cofiring_cells and "all" not in cofiring_nums:
                    self.remove_arrow((id1, id2))
                    continue

                x1, y1 = cell_id_coords[id1]
                x2, y2 = cell_id_coords[id2]
                z_offset = 0.5 # We need to offset the arrows slightly above the surface
                if value > 0:
                    if (id1, id2) in self.arrows:
                        visualized_arrows.append((id1, id2))
                        continue
                    normalized_value = value / max_cofiring
                    color = colormap(normalized_value)
                    color = (color[0], color[1], color[2])
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    tip_length = (5 + 10 * normalized_value) / distance
                    tip_radius = (2 + 4 * normalized_value) / distance
                    shaft_radius = (1 + 2 * normalized_value) / distance
                    mesh = pv.Arrow(start=(x1, y1, z_offset),
                                    direction=(x2-x1, y2-y1, z_offset),
                                    tip_length=tip_length,
                                    tip_radius=tip_radius,
                                    shaft_radius=shaft_radius,
                                    scale="auto",
                                    )
                    arrow_actor = self.add_mesh(mesh, color=color, show_scalar_bar=False, reset_camera=False)
                    self.arrows[(id1, id2)] = arrow_actor
                    visualized_arrows.append((id1, id2))
        
        # Remove any arrows that are not visualized
        for id in list(self.arrows.keys()):
            if id == "params":
                continue
            if id not in visualized_arrows:
                self.remove_arrow(id)

        # Reset the gridlines
        shape = self.visualization_data.get_shape() 
        self.show_grid(bounds=(0, shape["x"], 0, shape["y"], self.scalar_range[0] * self.scaling_factor, self.scalar_range[1] * self.scaling_factor),
                       axes_ranges=(0, shape["x"], 0, shape["y"], self.scalar_range[0], self.scalar_range[1]), color='white')
        self.render()

    def remove_cofiring(self, check_params=None):
        if check_params is not None:
            if self.arrows["params"] == check_params:
                return
        for ids in list(self.arrows.keys()):
            self.remove_arrow(ids)
        self.arrows = {"params": ""}
        shape = self.visualization_data.get_shape() 
        self.show_grid(bounds=(0, shape["x"], 0, shape["y"], self.scalar_range[0] * self.scaling_factor, self.scalar_range[1] * self.scaling_factor),
                       axes_ranges=(0, shape["x"], 0, shape["y"], self.scalar_range[0], self.scalar_range[1]), color='white')
        self.render()

    def closeEvent(self, event):
        self.clear()
        pv.close_all()
        super().closeEvent(event)
        

class VisualizationAdvancedWidget(QtInteractor):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        """
        Visualization for the advanced components of the visualization, where
        we'll summarize per cell a readout in each window. The window size
        will be specified by the user. Initially we'll render an empty grid
        and then it will be updated per user specifications.
        """

        # Create an empty grid
        x, y = np.meshgrid(np.arange(100), np.arange(100))
        x, y = x.astype(np.float32), y.astype(np.float32)
        z = np.zeros_like(x, dtype=np.float32)
        self.grid = pv.StructuredGrid(x, y, z)
        self.add_mesh(self.grid, scalar_bar_args=None, pickable=False)
        self.background_color = 'black'
        self.scalar_range = (0, 5)
        self.ranges_changed = False
        self.scaling_factor = 10
        self.data_grid = None
        self.grid_bounds = (0, 100, 0, 100, -5, 80)
        self.change_colormap('fire')
        self.session = session
        self.update_precalculated_values()

    def update_precalculated_values(self):
        self.precalculated_values = _precalculate(self.session)

    def update_scaling_factor(self, scaling):
        """
        Update the scaling factor for the visualization. This will change the z-axis of the grid.
        """
        self.scaling_factor = scaling
        self.update_current_window(self.current_window+1)
        self.change_colormap()

    def change_colormap(self, name=None):
        # Update the colormap for the plane (StructuredGrid)
        if name is not None:
            self.cmap = name
        
        # Create a LookupTable for the plane
        plane_lut = pv.LookupTable(cc.cm[self.cmap], n_values=256)
        
        # Scale the values for visual representation but keep original for display
        actual_min = self.scalar_range[0] * self.scaling_factor
        actual_max = self.scalar_range[1] * self.scaling_factor
        
        # Use scaled values for color mapping
        plane_lut.scalar_range = (actual_min, actual_max)
        
        # Create annotations to show actual unscaled values
        n_ticks = 5
        ticks = np.linspace(actual_min, actual_max, n_ticks)
        plane_lut.annotations = {
            float(t): f"{(t / self.scaling_factor):.2f}"  # Convert back to unscaled for display
            for t in ticks
        }

        # Apply to the grid
        mapper = None
        for actor_name in self.actors.keys():
            if "Grid" in actor_name:
                mapper = self.actors[actor_name].GetMapper()
                mapper.SetLookupTable(plane_lut)
                mapper.SetScalarRange(actual_min, actual_max)  # Use scaled range for colors
                break

        # Update the scalar bar
        for scalar_bar_name in list(self.scalar_bars.keys()):
            self.remove_scalar_bar(scalar_bar_name)
        
        bar_actor = self.add_scalar_bar(
            title='Intensity',
            color='white',
            shadow=True,
            n_labels=n_ticks,
            mapper=mapper,
        )
        
        # Disable default tick labels since we use annotations
        bar_actor.DrawTickLabelsOff()
        
        # The key part: separate visual scaling from displayed values
        # Use SCALED values for bounds (visual) but UNSCALED for axes_ranges (display)
        self.show_grid(
            bounds=(self.grid_bounds[0], self.grid_bounds[1], 
                    self.grid_bounds[2], self.grid_bounds[3], 
                    actual_min - 0.1, actual_max),  # Use scaled values for visual bounds
            axes_ranges=(self.grid_bounds[0], self.grid_bounds[1], 
                        self.grid_bounds[2], self.grid_bounds[3], 
                        self.scalar_range[0], self.scalar_range[1]),  # Use unscaled values for display
            color='white'
        )
        
        self.render()
        

    def readjust_ranges(self):
        # Work with unscaled values
        min_height = self.data_grid[self.current_window].min() if self.data_grid[self.current_window].min() < self.scalar_range[0] else self.scalar_range[0]
        max_height = self.data_grid[self.current_window].max() if self.data_grid[self.current_window].max() > self.scalar_range[1] else self.scalar_range[1]
        new_scalar_range = (min(min_height, -0.1), max_height)  # Use unscaled values
        
        if new_scalar_range != self.scalar_range:
            self.scalar_range = new_scalar_range
            self.ranges_changed = True

    def set_data(self, a_cells, b_cells, window_size, readout, fpr, scaling = 10):
        """
        Set the data for the advanced visualization. This will be used to visualize
        the data in a 2D grid where the x and y axis are the cell ids and the color
        of the grid is the readout value for the window size.

        Parameters
        ----------
        session : DataInstance
            The session object that contains all the relevant session data from the parent class.
        a_cells : list
            The cell ids for the A cells.
        b_cells : list
            The cell ids for the B cells.
        window_size : int
            The window size for the readout.
        readout : str
            The readout to calculate for each window, can be 'Event Count Frequency', 'Average DFF Peak', 'Total Dff'
        fpr : float
            The further processed readout for the readout. 
        scaling : int
            The scaling factor for the readout, by default 10. This changes the z-axis of the grid.
        """
        self.scaling_factor = scaling
        self.scalar_range = (0, 5)
        # 1.) Iterating through a cells, retrieve the footprint with the relative positions of cell bs
        a_b_relative_centers = {}
        x_start, x_end, y_start, y_end = 0, 0, 0, 0 # So we can constrain the grid to the smallest possible size
        for a_cell in a_cells:
            # Retrieve the center of the a cell
            a_center = self.session.centroids[a_cell]
            b_relative_centers = {}
            # Iterate through the b cells and retrieve the relative positions to a_center
            for b_cell in b_cells:
                if a_cell == b_cell:
                    continue
                b_center = self.session.centroids[b_cell]
                b_relative_center = np.array(b_center) - np.array(a_center)
                x_start, x_end, y_start, y_end = min(x_start, b_relative_center[0]), max(x_end, b_relative_center[0]), min(y_start, b_relative_center[1]), max(y_end, b_relative_center[1])
                b_relative_centers[b_cell] = b_relative_center
            a_b_relative_centers[a_cell] = b_relative_centers
        
        # 2.) Calculate the readout for each window
        sv_win_data = calculate_single_value_windowed_data(self.session, self.precalculated_values, readout, window_size)

        # 2.1) Check if a_cells and b_cells are in sv_win_data
        missing_cells = set()
        for a_cell in a_cells:
            if a_cell not in sv_win_data.coords['unit_id'].values:
                missing_cells.add(a_cell)
        
        for b_cell in b_cells:
            if b_cell not in sv_win_data.coords['unit_id'].values:
                missing_cells.add(b_cell)
        
        if len(missing_cells) > 0:
            print_error("Missing Transients", extra_info=f"The following cells have no registered transients: {missing_cells}\n" + 
                         "Deselect the cells from the A Cell and B Cell list or register transients", severity=QMessageBox.Warning)
            return None

        # 3.) Iterate through the a cells and b cells and calculate fpr
        a_to_b_fpr = calculate_fpr(a_cells, b_cells, sv_win_data, fpr)
        
        # 4.) Now we are going to create a numpy array with the values above of dim num_win, (x_end-x_start), (y_end-y_start)
        # We'll add a little bit of padding, so we can encompass any cell that we input
        padding = 20
        window_size = sv_win_data.shape[1]
        grid = np.zeros((window_size, 2*int(x_end-x_start+padding), 2*int(y_end-y_start+padding)))
        # get the center of each plane
        center_x = int(x_end-x_start+padding)
        center_y = int(y_end-y_start+padding)

        # Iterate through num_win and fill in the values
        for win in range(window_size):
            for a_cell in a_cells:
                for b_cell in b_cells:
                    if a_cell == b_cell:
                        continue
                    relative_center = a_b_relative_centers[a_cell][b_cell]
                    x, y = center_x + relative_center[0], center_y + relative_center[1]
                    value = gaussian_2d() * a_to_b_fpr[(a_cell, b_cell)].sel(window=win).values
                    grid[win, int(x-3):int(x+4), int(y-3):int(y+4)] = value
        
        self.bin_size = window_size
        self.data_grid = grid  # Store UNSCALED data
        self.current_window = 0

        # Clear current grid
        self.clear()
        x, y = np.meshgrid(np.arange(grid.shape[1], dtype=np.float32), np.arange(grid.shape[2], dtype=np.float32))
        self.grid = pv.StructuredGrid(x, y, grid[0].T * self.scaling_factor)  # Apply scaling here
        self.grid["scalars"] = (grid[0].T * self.scaling_factor).ravel(order='F')  # And here
        self.add_mesh(self.grid, scalar_bar_args=None, pickable=False)

        # Create the marker for the center
        marker = create_x_marker((center_x, center_y, 0), size=15.0, color="green", line_width=20)
        self.add_mesh(marker, color='green', pickable=False)

        # Store both UNSCALED and SCALED range for proper display
        unscaled_min, unscaled_max = grid[0].min(), grid[0].max()
        self.scalar_range = (unscaled_min, unscaled_max)  # Unscaled for axis labels
        
        # Set bounds for the grid - use scaled values for visual height
        self.grid_bounds = (0, grid.shape[1], 0, grid.shape[2], 
                            unscaled_min * self.scaling_factor - 0.5, 
                            unscaled_max * self.scaling_factor)

        self.readjust_ranges()
        self.change_colormap()
        
        return self.bin_size

    def update_current_window(self, window):
        self.current_window = window-1
        # Get unscaled data
        current_data = self.data_grid[self.current_window]
        
        # Apply scaling only for visual representation
        scaled_data = current_data * self.scaling_factor
        
        # Update the grid with scaled values
        self.grid.points[:,2] = scaled_data.T.ravel(order='F')
        self.grid["scalars"] = scaled_data.T.ravel(order='F')
        
        # Update scalar ranges - store unscaled values
        self.scalar_range = (current_data.min(), current_data.max())
        
        self.readjust_ranges()
        self.change_colormap()
        self.render()
            

    def closeEvent(self, event):
        self.clear()
        pv.close_all()
        super().closeEvent(event)

def check_cofiring(A_starts: List[int], B_starts: List[int], window_size: int, shareA:bool=True, shareB:bool=True, direction:str="bidirectional", omit_first=False, **kwargs):
    """
    Check if two cells are cofiring with each other. This is generally done
    by checking if the start of a transient in one cell is within the window
    size of the start of a transient in another cell. If the `shareA` this 
    means that a transient in cell A can be shared with multiple transients
    in cell B. The same goes for `shareB`, where a transient in cell B can be
    shared with multiple transients in cell A. The `direction` parameter indicates
    whether the window size looks forward, backward or bidirectional with respect
    to the temporal axis.

    Parameters
    ----------
    A_starts : list
        The starts of the transients in cell A.
    B_starts : list
        The starts of the transients in cell B.
    window_size : int
        The window size to check for cofiring.
    shareA : bool
        Whether a transient in cell A can be shared with multiple transients in cell B.
    shareB : bool
        Whether a transient in cell B can be shared with multiple transients in cell A.
    direction : str
        The direction to check for cofiring. Can be 'forward', 'backward' or 'bidirectional'.
    omit_first : bool
        Whether to omit the first transient in the cofiring check. This is to compensate
        for the fact that when calculating itis and then shuffling the data, the first transient
        has no previous transient to compare to, therefore it may artificially inflate the cofiring
        values when compared to the shuffled data.

    Returns
    -------
    num_cofiring : int
        The number of cofiring events between the two cells.
    """

    num_cofiring = 0
    A_starts_used = set()
    B_starts_used = set()
    # We will check kwargs to see if we have used connections
    # If yes that means we want to avoid double counting the same A->B and B->A transients
    connections_used = kwargs.get("connections_used", None)
    if connections_used is not None:
        # Extract cell ids
        A_id = kwargs.get("A_id")
        B_id = kwargs.get("B_id")


    def check_overlap(start1, start2):
        if direction == "forward":
            return 0 <= start2 - start1 <= window_size
        elif direction == "backward":
            return 0 <= start1 - start2 <= window_size
        else:
            return abs(start1 - start2) <= (window_size // 2)


    # This is a brute force method, but it should be fine for the number of transients
    # Might be worth optimizing this in the future
    for i, startA in enumerate(A_starts):
        for j, startB in enumerate(B_starts):
            if check_overlap(startA, startB):
                if omit_first and i == 0 or j == 0:
                    continue
                if not shareA:
                    if startA in A_starts_used:
                        continue
                if not shareB:
                    if startB in B_starts_used:
                        continue
                
                A_starts_used.add(startA)
                B_starts_used.add(startB)
                if connections_used is not None:
                    # For convenience we will store the starts in ascending order
                    starting_frames = (startA, startB) if startA < startB else (startB, startA)
                    # Same goes for ids
                    connection_id = (A_id, B_id) if A_id < B_id else (B_id, A_id)
                    if starting_frames in connections_used.get(connection_id, []):
                        continue

                    if starting_frames not in connections_used.setdefault(connection_id, []):
                        connections_used[connection_id].append(starting_frames)


                                
                    
                num_cofiring += 1

    return num_cofiring

def create_x_marker(center, size=1.0, color="green", line_width=5):
    """
    Create an "X" marker using two crossing lines in PyVista.

    Parameters:
    - center (tuple or list): The (x, y, z) coordinates of the marker center.
    - size (float): The size of the "X" marker.
    - color (str): The color of the lines.
    - line_width (int): The width of the lines.

    Returns:
    - pv.MultiBlock: A MultiBlock containing the two line segments forming the "X".
    """
    center = np.array(center)

    # Define the endpoints of the X
    p1 = center + np.array([-size, -size, 0])
    p2 = center + np.array([size, size, 0])
    p3 = center + np.array([-size, size, 0])
    p4 = center + np.array([size, -size, 0])

    # Create line segments
    line1 = pv.Line(p1, p2)
    line2 = pv.Line(p3, p4)

    # Combine into a MultiBlock
    marker = pv.MultiBlock([line1, line2])
    
    # Assign colors and line width
    marker[0]["color"] = color
    marker[1]["color"] = color
    marker[0]["line_width"] = line_width
    marker[1]["line_width"] = line_width

    return marker