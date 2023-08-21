from PyQt5.QtWidgets import (QApplication, QMainWindow, QStyle, QFileDialog, QMessageBox, QAction, QDialog,
                            QLabel)
from PyQt5.QtCore import (QThreadPool)
from custom_widgets import LoadingDialog
import sys
sys.path.insert(0, ".")
from backend import open_minian

class MainWindow(QMainWindow):


    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Cell Clustering Tool")
        self.setMinimumSize(600, 400)

        # Threading
        self.threadpool = QThreadPool()

        # Data stuff
        self.data = []

        # Menu Bar
        pixmapi = QStyle.StandardPixmap.SP_DirIcon
        button_action = QAction(self.style().standardIcon(pixmapi), "&Load Data", self)
        button_action.setStatusTip("Select a Folder to load in data")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action)

        self.show()

    def printError(self, s):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("I have a question!")
        if isinstance(s, tuple):
            text = f"For path {s[1]} the following error occurred:\n {s[0]}"
        else:
            text = s
        dlg.setText(text)
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.exec()


    def addData(self):
        #self.data.append(data)
        print("finished")


    def onMyToolBarButtonClick(self, s):
        fname = QFileDialog.getExistingDirectory(
            self,
            "Open File",
        )

        # Cannot do the stuff below as it segfaults :(
        '''
        worker = Worker(open_minian, fname, self.data)
        self.threadpool.start(worker)
        worker.signals.error.connect(self.printError)
        worker.signals.finished.connect(self.addData)

        # Execute
        self.threadpool.start(worker)
        '''
        self.setWindowTitle("Loading...")

        self.data.append(open_minian(fname))

        self.setWindowTitle("Cell Clustering Tool")



        




app = QApplication([])
window = MainWindow()
app.exec()