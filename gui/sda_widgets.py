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

    def update_frame(self, data):
        self.plot.mlab_source.set(scalars=data)
    
    def reset_frame(self, frame=0):
        frame = self.visualization_data.data[0]
        x = np.linspace(0, frame.shape[0], frame.shape[0])
        y = np.linspace(0, frame.shape[1], frame.shape[1])
        self.plot.mlab_source.reset(x=x, y=y, scalars=self.visualization_data.data[frame])

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
        self.update_frame(self.visualization_data.get_frame(frame))

class CurrentVisualizationData():
    def __init__(self, data, start_frame, end_frame, x_start, x_end, y_start, y_end):
        self.data = data
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

    def get_frame(self, frame):
        return self.data[frame-self.start_frame]
    
    def in_range(self, frame):
        return frame >= self.start_frame and frame <= self.end_frame-1
    
    def get_ranges(self):
        return [self.x_start, self.x_end, self.y_start, self.y_end, 0, self.data.max()]
    
    def get_extent(self):
        x = self.data.shape[1]
        y = self.data.shape[2]
        z = self.data.max()
        return [0, x, 0, y, 0, z]
    
    def __mul__(self, factor):
        self.data *= factor
        return self

    __rmul__ = __mul__

def base_visualization(session, data_type="C", start_frame=0, end_frame=200, **kwargs):
    signal = session.data[data_type].sel(frame=slice(start_frame, end_frame-1)).values # -1 since it is inclusive
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

    return CurrentVisualizationData(Y, start_frame, end_frame, x_start, x_end, y_start, y_end)

def normalized_events_visualization(session, data_type="C", start_frame=0, end_frame=200, **kwargs):
    signal = session.data[data_type].sel(frame=slice(start_frame, end_frame-1)).values # -1 since it is inclusive
    E = session.data["E"].values
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

    # Look at the E matrix and extrapolate the start and end

    Y = np.flip(np.tensordot(signal, A, axes=([0], [0])).swapaxes(1, 2), 2) # In order to maintain parity with the 2D visualization

    return CurrentVisualizationData(Y, start_frame, end_frame, x_start, x_end, y_start, y_end)


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
        self.precalculate_values = self._precalculate(session)

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
                split_indices = np.split(indices, np.where(np.diff(indices) != 1)[0]+1)
                # Now Split the indices into pairs of first and last indices
                split_indices = [(indices_group[0], indices_group[-1]+1) for indices_group in split_indices]

                C_based_total = 0
                DFF_based_total = 0
                for start, end in split_indices:
                    C_based_val = abs(C[i, start:end].max() - C[i, start:end].min())
                    DFF_based_val = abs(DFF[i, start:end].max() - DFF[i, start:end].min())

                    C_based_events[i, start:end] = C_based_val
                    DFF_based_events[i, start:end] = DFF_based_val

                    C_based_total += C_based_val
                    DFF_based_total += DFF_based_val

                    C_cumulative_events[i, start:end] = C_based_total
                    DFF_cumulative_events[i, start:end] = DFF_based_total
                
                # Normalize the values by the total in both cases
                C_based_events[i] /= C_based_total
                DFF_based_events[i] /= DFF_based_total
                C_cumulative_events[i] /= C_based_total
                DFF_cumulative_events[i] /= DFF_based_total

                # Interpolate the values to fill in the gaps for cumulative events
                C_cumulative_events[i] = interp1d(np.arange(C_cumulative_events.shape[1])[indices], C_cumulative_events[i][indices])
                DFF_cumulative_events[i] = interp1d(np.arange(DFF_cumulative_events.shape[1])[indices], DFF_cumulative_events[i][indices])

                # We'll simulate decay for the base events by taking the last value and multiplying it by 0.95
                for start, end in split_indices:
                    last_val_C = C_based_events[i, end-1] * 0.95
                    last_val_DFF = DFF_based_events[i, end-1] * 0.95

                    i = end
                    while last_val_C > 1 and C_based_events[i] == 0 and i < C_based_events.shape[1]:
                        C_based_events[i] = last_val_C
                        last_val_C *= 0.95
                        i += 1
                    
                    i = end
                    while last_val_DFF > 1 and DFF_based_events[i] == 0 and i < DFF_based_events.shape[1]:
                        DFF_based_events[i] = last_val_DFF
                        last_val_DFF *= 0.95
                        i += 1
        
        precalculated_values['C_based_events'] = C_based_events
        precalculated_values['C_cumulative_events'] = C_cumulative_events
        precalculated_values['DFF_based_events'] = DFF_based_events
        precalculated_values['DFF_cumulative_events'] = DFF_cumulative_events
        
        return precalculated_values
                    


    def set_data(self, visualization_data):
        self.visualization_data = visualization_data * self.scaling_factor
        self.visualization.update_data(self.visualization_data)
    
    def set_frame(self, frame):
        if not self.visualization_data.in_range(frame):
            # We are out of range and we need to update the data, first we need to find the nearest chunk, which is a multiple of chunk_length
            chunk_start = frame - frame % self.chunk_length
            self.set_data(self.visualization_generator(self.session, start_frame=chunk_start, end_frame=chunk_start+self.window_size, **self.kwargs))
        self.visualization.set_frame(frame)
        self.current_frame = frame



    def change_colormap(self, colormap):
        try:
            self.visualization.plot.module_manager.scalar_lut_manager.lut_mode = colormap
        except:
            # Capitalize the first letter
            colormap = colormap.capitalize()
            self.visualization.plot.module_manager.scalar_lut_manager.lut_mode = colormap

    def change_func(self, func, **kwargs):
        self.visualization_generator = func
        self.kwargs = kwargs
        chunk_start = self.current_frame - self.current_frame % self.chunk_length
        self.set_data(self.visualization_generator(self.session, start_frame=chunk_start, end_frame=chunk_start+self.window_size))
        self.set_frame(self.current_frame)



    
