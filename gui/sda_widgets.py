# Spatial distribution analysis

from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi import mlab 
from PyQt5.QtWidgets import  QWidget, QVBoxLayout, QPushButton
import numpy as np

class SDAWindowWidget(QWidget):
    def __init__(self, session, name, main_window_ref, parent=None):
        super().__init__() 
        self.session = session
        self.name = name
        self.main_window_ref = main_window_ref
        self.anim = None

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

        # Add a simple button to start the animation
        self.button = QPushButton("Start/Stop Animation")
        self.button.clicked.connect(self.start_stop_animation)

               


        self.mayavi_widget = MayaviQWidget(Y)
        self.mayavi_widget.setObjectName("3D Cell Overview")
        self.layout.addWidget(self.mayavi_widget)
        self.layout.addWidget(self.button)

    def start_stop_animation(self):
        if not self.anim:
            self.anim = self.mayavi_widget.visualization.animation()
        
        if self.anim.timer.IsRunning():
            self.anim._stop_fired()
        else:
            self.anim._start_fired()

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    def __init__(self, data):
        HasTraits.__init__(self)
        self.data = data
        self.current_idx = 0
        self.length = len(data)
        
    def update_data(self, data):
        self.plot.mlab_source.scalars = data
        
    @on_trait_change('scene.activated')
    def initial_plot(self):
        self.plot = mlab.surf(self.data[self.current_idx], colormap='viridis')
        self.scene.background=(1,1,1)

    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                      height=250, width=300, show_label=False),
                resizable=True # We need this to resize with the parent widget
                )
    

    @mlab.animate(delay=10, ui=False)
    def animation(self):        
        while 1:
            self.current_idx = (self.current_idx + 1) % self.length
            self.update_data(self.data[self.current_idx])
            yield



class MayaviQWidget(QWidget):
    def __init__(self, data):
        super().__init__()
        scaling_factor = 10
        self.data = data * scaling_factor
        self.visualization = Visualization(self.data)
        layout = QVBoxLayout(self)
        self.ui = self.visualization.edit_traits(parent=self,
                                                  kind='subpanel').control
        layout.addWidget(self.ui)