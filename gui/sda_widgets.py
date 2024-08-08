# Spatial distribution analysis

from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi import mlab 
from PyQt5.QtWidgets import  QWidget, QVBoxLayout, QPushButton
import numpy as np
from scipy.interpolate import interp1d

class SDAWindowWidget(QWidget):
    def __init__(self, session, name, main_window_ref, parent=None):
        super().__init__() 
        self.session = session
        self.name = name
        self.main_window_ref = main_window_ref

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        C = session.data["C"].sel(frame=slice(0,1999)).values
        A = session.data["A"].values
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

        Y = np.tensordot(C, A, axes=([0], [0]))

        self.mayavi_widget = MayaviQWidget(Y)
        self.mayavi_widget.setObjectName("3D Cell Overview")

        # Add a simple button to start the animation
        self.button = QPushButton("Start/Stop Animation")
        self.button.clicked.connect(self.mayavi_widget.start_stop_animation)

        # Add a panel to the side for different options and utilities

        self.layout.addWidget(self.mayavi_widget)
        self.layout.addWidget(self.button)


class Visualization(HasTraits):
    # Signal to indicate scene has been activated
    scene = Instance(MlabSceneModel, ())
    def __init__(self, visualization_data):
        HasTraits.__init__(self)
        self.visualization_data = visualization_data
        self.axes = None
        self.points_3d = None

    def update_frame(self, data):
        self.plot.mlab_source.set(scalars=data)

    def update_points(self, points_coords):
        if not len(points_coords[0]) == 0:
            if self.points_3d is None:
                self.points_3d = mlab.points3d(*points_coords, color=(1, 1, 1), mode="point", scale_mode="none", scale_factor=1.0)
                self.points_3d.actor.property.point_size = 10
                #self.points_3d.actor.property.render_points_as_spheres = True
            else:
                self.points_3d.mlab_source.set(x=points_coords[0], y=points_coords[1], z=points_coords[2])
    
    def reset_frame(self, frame_no=0):
        frame = self.visualization_data.data[0]
        x = np.linspace(0, frame.shape[0], frame.shape[0])
        y = np.linspace(0, frame.shape[1], frame.shape[1])
        self.plot.mlab_source.reset(x=x, y=y, scalars=self.visualization_data.data[frame_no])

    def update_data(self, visualization_data, frame=0):
        self.visualization_data = visualization_data
        self.reset_frame(frame)
        self.axes = mlab.axes(extent=self.visualization_data.get_extent(), ranges=self.visualization_data.get_ranges(), xlabel='Width', ylabel='Height', zlabel='Spike Intensity')

        
    @on_trait_change('scene.activated')
    def initial_plot(self):
        self.plot = mlab.surf(self.visualization_data.data[0], colormap='hot')
        self.axes = mlab.axes(extent=self.visualization_data.get_extent(), ranges=self.visualization_data.get_ranges(), xlabel='Width', ylabel='Height', zlabel='Spike Intensity')
        self.scene.scene.background = (1, 1, 1)
        self.scene.scene.foreground = (0, 0, 0)
        self.scene.anti_aliasing_frames = 0
        

    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                      height=250, width=300, show_label=False),
                resizable=True # We need this to resize with the parent widget
                )
    
    def set_frame(self, frame):
        data_dict = self.visualization_data.get_3d_data(frame)
        self.update_frame(data_dict["frame"])
        self.update_points(data_dict["points_coords"])

class CurrentVisualizationData():
    def __init__(self, data, start_frame, end_frame, x_start, x_end, y_start, y_end, scaling_factor=10):
        self.data = data
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.scaling_factor = scaling_factor
        self.points = {"x": np.array([]), "y": np.array([])}

    def get_3d_data(self, frame) -> dict:
        # Returns a dictionary containing the data for surface plot and the points of the centroids
        frame = self.data[frame-self.start_frame] * self.scaling_factor
        # Get the x, y and z values for the points
        points_coords = (self.points["x"], self.points["y"], frame[self.points["x"], self.points["y"]]) if len(self.points["x"]) > 0 else ([], [], [])
        return {"frame": frame, "points_coords": points_coords}
    
    def update_points_list(self, points):
        points = points.values()
        x_coords, y_coords = zip(*points)

        y_coords, x_coords = np.array(x_coords).round().astype(int) - int(self.x_start), np.array(y_coords).round().astype(int) - int(self.y_start) # They need to be switched around due prior flipping
        # Finally y_coords needs to be flipped with respect to its axis
        # TODO
        self.points = {"x": x_coords, "y": y_coords}

    
    def in_range(self, frame):
        return frame >= self.start_frame and frame <= self.end_frame-1
    
    def get_ranges(self):
        return [self.x_start, self.x_end, self.y_start, self.y_end, 0, self.data.max()]
    
    def get_extent(self):
        x = self.data.shape[1]
        y = self.data.shape[2]
        z = self.data.max() * self.scaling_factor
        return [0, x, 0, y, 0, z]
    
    def __mul__(self, factor):
        self.scaling_factor = factor
        return self

    __rmul__ = __mul__

def base_visualization(session, precalculated_values=None, data_type="C", start_frame=0, end_frame=200, **kwargs):
    if data_type in session.data:
        signal = session.data[data_type].sel(frame=slice(start_frame, end_frame-1)).values # -1 since it is inclusive
    else:
        signal = precalculated_values[data_type].sel(frame=slice(start_frame, end_frame-1)).values # -1 since it is inclusive
    A = session.data["A"].values
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

    Y = np.flip(np.tensordot(signal, A, axes=([0], [0])).swapaxes(1, 2), 2) # In order to maintain parity with the 2D visualization

    # Since we did the prior we need to flip x_start, y_start etc...
    x_start, x_end, y_start, y_end = y_start, y_end, x_start, x_end

    CV = CurrentVisualizationData(Y, start_frame, end_frame, x_start, x_end, y_start, y_end)
    CV.update_points_list(session.centroids)

    return CV


class MayaviQWidget(QWidget):
    def __init__(self, session, chunk_length=50, window_size=200, visualization_data=CurrentVisualizationData(np.random.rand(1, 608, 608), 0, 0, 0, 608, 0, 608), visualization_generator=base_visualization):
        super().__init__()
        self.chunk_length = chunk_length
        self.window_size = window_size
        self.session = session
        self.scaling_factor = 10
        self.visualization_data = visualization_data * self.scaling_factor
        self.anim = None
        self.visualization = Visualization(self.visualization_data)
        self.visualization_generator = visualization_generator
        layout = QVBoxLayout(self)
        self.ui = self.visualization.edit_traits(parent=self,
                                                  kind='subpanel').control
        self.current_frame = 0
        layout.addWidget(self.ui)
        self.kwargs = {}

        # We need to precalculate some data related to E to not cause slow down during visualization
        self.precalculated_values = self._precalculate(session)

        # Convert the numpy arrays to xarray DataArrays, copy over coords and dims from C
        for key, value in self.precalculated_values.items():
            self.precalculated_values[key] = session.data['C'].copy(data=value)

    def _precalculate(self, session):
        precalculated_values = {}
        E = self.session.data['E'].values
        C = self.session.data['C'].values
        DFF = self.session.data['DFF'].values

        # Output values
        C_based_events = np.zeros(C.shape)
        C_cumulative_events = np.zeros(C.shape)
        DFF_based_events = np.zeros(DFF.shape)
        DFF_cumulative_events = np.zeros(DFF.shape)

        for i, row in enumerate(E):
            events = np.nan_to_num(row, nan=0) # Sometimes saving errors can cause NaNs
            indices = events.nonzero()
            if indices[0].any():
                # Split up the indices into groups
                split_indices = np.split(indices[0], np.where(np.diff(indices[0]) != 1)[0]+1)
                # Now Split the indices into pairs of first and last indices
                split_indices = [(indices_group[0], indices_group[-1]+1) for indices_group in split_indices]

                C_based_total = 0
                DFF_based_total = 0
                indices_to_remove = []
                for j, (start, end) in enumerate(split_indices):
                    C_based_val = abs(C[i, start:end].max() - C[i, start:end].min())
                    DFF_based_val = abs(DFF[i, start:end].max() - DFF[i, start:end].min())

                    if C_based_val == 0 or DFF_based_val == 0:
                        indices_to_remove.append(j)
                        continue # Can occur due to erroneous data, skip these

                    C_based_events[i, start:end] = C_based_val
                    DFF_based_events[i, start:end] = DFF_based_val

                    C_based_total += C_based_val
                    DFF_based_total += DFF_based_val

                    C_cumulative_events[i, start:end] = C_based_total
                    DFF_cumulative_events[i, start:end] = DFF_based_total
                
                # Remove the erroneous indices
                split_indices = [split_indices[j] for j in range(len(split_indices)) if j not in indices_to_remove]
                
                # Normalize the values by the total in both cases
                C_based_events[i] /= C_based_total
                DFF_based_events[i] /= DFF_based_total
                C_cumulative_events[i] /= C_based_total
                C_cumulative_events[i][0] = 0.000001 # So we don't have interpolation issues
                C_cumulative_events[i][-1] = 1
                DFF_cumulative_events[i] /= DFF_based_total
                DFF_cumulative_events[i][0] = 0.000001
                DFF_cumulative_events[i][-1] = 1

                # Interpolate the values to fill in the gaps for cumulative events
                C_cumulative_events[i] = self._interpolate(C_cumulative_events[i])
                DFF_cumulative_events[i] = self._interpolate(DFF_cumulative_events[i])

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


        
        precalculated_values['C_base'] = C_based_events
        precalculated_values['C_cumulative'] = C_cumulative_events
        precalculated_values['DFF_base'] = DFF_based_events
        precalculated_values['DFF_cumulative'] = DFF_cumulative_events
        
        return precalculated_values
                    
    def _interpolate(self, y):
        x = np.arange(len(y))
        idx = np.nonzero(y)
        interp = interp1d(x[idx],y[idx])

        return interp(x)
        

    def set_data(self, visualization_data):
        self.visualization_data = visualization_data * self.scaling_factor
        self.visualization.update_data(self.visualization_data)
    
    def set_frame(self, frame):
        if not self.visualization_data.in_range(frame):
            # We are out of range and we need to update the data, first we need to find the nearest chunk, which is a multiple of chunk_length
            chunk_start = frame - frame % self.chunk_length
            self.set_data(self.visualization_generator(self.session, precalculated_values=self.precalculated_values, start_frame=chunk_start, end_frame=chunk_start+self.window_size, **self.kwargs))
        self.visualization.set_frame(frame)
        self.current_frame = frame



    def change_colormap(self, colormap):
        self.visualization.plot.module_manager.scalar_lut_manager.lut_mode = colormap

    def change_func(self, func, **kwargs):
        if "scaling" in kwargs:
            self.scaling_factor = kwargs["scaling"]
        self.visualization_generator = func
        self.kwargs = kwargs
        chunk_start = self.current_frame - self.current_frame % self.chunk_length
        self.set_data(self.visualization_generator(self.session, precalculated_values=self.precalculated_values,  start_frame=chunk_start, end_frame=chunk_start+self.window_size, **kwargs))
        self.set_frame(self.current_frame)



    
