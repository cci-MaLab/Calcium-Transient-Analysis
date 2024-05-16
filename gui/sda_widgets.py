# Spatial distribution analysis

from PyQt5.QtWidgets import QWidget, QVBoxLayout

class SDAWindowWidget(QWidget):
    def __init__(self):
        super().__init__() 

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)