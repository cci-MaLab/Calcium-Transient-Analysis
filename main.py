from PyQt6.QtWidgets import (QApplication, QMainWindow, QStyle, QFileDialog, QMessageBox)
from PyQt6.QtGui import (QAction)
from threads import Worker


class MainWindow(QMainWindow):


    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Cell CLustering Tool")
        self.setMinimumSize(600, 400)

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

    def threadComplete(self):
        print("THREAD COMPLETE!")

    def addData(self, data):
        self.data.append(data)


    def onMyToolBarButtonClick(self, s):
        fname = QFileDialog.getExistingDirectory(
            self,
            "Open File",
        )
        
        # Over here Haoying you should call your load_data function from backend.py using the Worker class
        # Something like this:
        '''
        worker = Worker(load_data, fname)
        self.threadpool.start(worker)
        worker.signals.error.connect(self.printError)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.threadComplete)

        # Execute
        self.threadpool.start(worker)
        '''
        




app = QApplication([])
window = MainWindow()
app.exec()