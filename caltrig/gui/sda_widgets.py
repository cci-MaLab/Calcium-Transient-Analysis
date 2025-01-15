# Spatial distribution analysis

from PyQt5.QtWidgets import  QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox
import numpy as np
from scipy.interpolate import interp1d
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
import pickle
from matplotlib.colors import ListedColormap

class CurrentVisualizationData():
    def __init__(self, data, max_height, start_frame, end_frame, x_start, x_end, y_start, y_end, scaling_factor=10):
        self.data = data
        self.max_height = max_height
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
        x_coords, y_coords = switch_to_3d_coordinates(x_coords, y_coords, self.x_start, self.y_start)     
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

def switch_to_3d_coordinates(x, y, x_start=0, y_start=0):
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
            max_height = xr_data.max().values
            signal = xr_data.sel(unit_id=ids).sel(frame=slice(start_frame, end_frame-1))
        else:
            xr_data = calculate_windowed_data(session, precalculated_values, data_type, window_size,
                                              cumulative=cumulative, normalize=normalize, average=average, name=name)
            precalculated_values[name] = xr_data
            max_height = xr_data.max().values
            signal = xr_data.sel(unit_id=ids).sel(frame=slice(start_frame, end_frame-1))
    else:
        if data_type in session.data:
            max_height = session.data[data_type].max().values
            signal = session.data[data_type].sel(frame=slice(start_frame, end_frame-1)) # -1 since it is inclusive
        else:
            max_height = precalculated_values[data_type].max().values
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

    #Since we swapped the axes we need to swap the x and y values
    x_start, x_end, y_start, y_end = y_start, y_end, x_start, x_end
    
    CV = CurrentVisualizationData(Y, max_height, start_frame, end_frame, x_start, x_end, y_start, y_end)
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


                
class PyVistaWidget(QtInteractor):
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
        self.precalculated_values = self._precalculate()
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
        self.scalar_range = (0, 50)
        self.background_color = 'black'
        self.points_3d = None
        self.cmap = "fire"
        self.arrows = {"params": ""}
        self.populate_3D_scene()

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
        self.enable_point_picking(callback=self.receive_click, use_mesh=True, pickable_window=self.points_3d, show_point=False, show_message="Right Click or press 'P' to select point/cell", left_click=True)
        self._picking_text.GetTextProperty().SetColor(1,1,1)
        self.change_colormap(self.cmap)

    def _precalculate(self):
        precalculated_values = {}
        E = self.session.data['E']
        C = self.session.data['C']
        DFF = self.session.data['DFF']

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
                C_cumulative_events[i] = self._forward_fill(C_cumulative_events[i])
                DFF_cumulative_events[i] = self._forward_fill(DFF_cumulative_events[i])
                frequency[i] = self._forward_fill(frequency[i])

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
        precalculated_values['Frequency'] = xr.DataArray(frequency, coords=[unit_ids, np.arange(C.shape[1])], dims=['unit_id', 'frame'], name='frequency')
        precalculated_values['transient_info'] = transient_info
        
        return precalculated_values

    def _forward_fill(self, y):
        prev = np.arange(len(y))
        prev[y == 0] = 0
        prev = np.maximum.accumulate(prev)
        return y[prev]
        
    def set_data(self, visualization_data):
        self.visualization_data = visualization_data * self.scaling_factor
        self.reset_grid()
        self.cell_id_to_index = self.visualization_data.cell_id_to_index
    
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

    def change_colormap(self, name):
        # Update the colormap for the plane (StructuredGrid)
        self.cmap = name
        plane_lut = pv.LookupTable(cc.cm[name])  # Create a LookupTable for the plane
        plane_lut.scalar_range = self.scalar_range  # Apply scalar range

        mapper = None
        for actor_name in self.actors.keys():
            if "Grid" in actor_name:
                mapper = self.actors[actor_name].GetMapper()
                mapper.SetLookupTable(plane_lut)
                mapper.SetScalarRange(self.scalar_range)
                break



        # Update the scalar bar for the plane
        for scalar_bar_name in list(self.scalar_bars.keys()):
            self.remove_scalar_bar(scalar_bar_name)
        self.add_scalar_bar(
            title='Spike Intensity',
            color='white',
            shadow=True,
            n_labels=5,
            fmt='%.0f',
            mapper=mapper,
        )
        shape = self.visualization_data.get_shape() 
        self.show_grid(bounds=(0, shape["x"], 0, shape["y"], -20, 150), color='white')
        self.render()

    def change_func(self, func, **kwargs):
        if "scaling" in kwargs:
            self.scaling_factor = kwargs["scaling"]
        self.visualization_generator = func
        self.kwargs_func = kwargs
        chunk_start = self.current_frame - self.current_frame % self.chunk_size

        self._update_serialize_data()
        self.set_data(self.visualization_generator(self.serialized_data, start_frame=chunk_start, end_frame=chunk_start+self.chunk_size))
        self.set_frame(self.current_frame)  

    def _update_serialize_data(self):
        self.serialized_data = pickle.dumps({"session": self.session, "precalculated_values": self.precalculated_values, "kwargs": self.kwargs_func})

    def update_selected_cells(self, cells):
        self.selected_cells = cells
        self.points_3d["colors"] = self.visualization_data.get_selected_points(self.selected_cells)
        self.render()
    
    def receive_click(self, mesh, index):
        if mesh is None:
            return
        x, y, _ = mesh.points[index]
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
                if value not in cofiring_nums and "all" not in cofiring_nums:
                    self.remove_arrow((id1, id2))
                    continue
                if id1 not in cofiring_cells and id2 not in cofiring_cells and "all" not in cofiring_nums:
                    self.remove_arrow((id1, id2))
                    continue

                x1, y1 = cell_id_coords[id1]
                x2, y2 = cell_id_coords[id2]
                z_offset = 0.5 # We need to offset the arrows slightly above the surface
                if value > 1:
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
        self.show_grid(bounds=(0, shape["x"], 0, shape["y"], -20, 150), color='white')
        self.render()

    def remove_cofiring(self, check_params=None):
        if check_params is not None:
            if self.arrows["params"] == check_params:
                return
        for ids in list(self.arrows.keys()):
            self.remove_arrow(ids)
        self.arrows = {"params": ""}
        shape = self.visualization_data.get_shape() 
        self.show_grid(bounds=(0, shape["x"], 0, shape["y"], -20, 150), color='white')
        self.render()
        
   
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
                num_cofiring += 1

    return num_cofiring