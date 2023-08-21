from PyQt5.QtWidgets import (QApplication, QMainWindow, QStyle, QFileDialog, QMessageBox, QAction,
                            QLabel, QVBoxLayout)
from PyQt5.QtCore import (QThreadPool)
from custom_widgets import LoadingDialog
import sys
import os
sys.path.insert(0, ".")
from backend import SessionFeature

class MainWindow(QMainWindow):


    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Cell Clustering Tool")
        self.setMinimumSize(600, 400)

        # Threading
        self.threadpool = QThreadPool()

        # Data stuff
        self.data = []
        self.path_list = set()

        # Menu Bar
        pixmapi = QStyle.StandardPixmap.SP_DirIcon
        button_action = QAction(self.style().standardIcon(pixmapi), "&Load Data", self)
        button_action.setStatusTip("Select a Folder to load in data")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action)

        # Layouts
        layout_main = QVBoxLayout()
        

        self.show()

        if os.path.isfile('paths.txt'):
            if os.path.getsize('paths.txt') != 0:
                dlg = LoadingDialog()
                if dlg.exec():
                    self.setWindowTitle("Loading...")
                    file = open('paths.txt', 'r')
                    lines = file.readlines()
                    for fname in lines:
                        self.data.append(SessionFeature(fname))

                    self.setWindowTitle("Cell Clustering Tool")
        else:
            with open('paths.txt', 'w') as fp:
                pass



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
        if fname != '' and fname not in self.path_list:
            self.setWindowTitle("Loading...")

            self.data.append(SessionFeature(fname))

            self.setWindowTitle("Cell Clustering Tool")

            self.path_list.add(fname)

            with open('paths.txt', 'a') as f:
                f.writelines([fname])
        




app = QApplication([])
window = MainWindow()
app.exec()