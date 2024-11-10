'''
In order to get mouse move event when left mouse button is pressed, we need to override a few pyqtgraph classes.
This is due to the fact that the GraphicsScene class only emits the position of the mouse without any information
whether a button is pressed or not. Theoretically this could be a bit of resource hog, however the mouse pressing
should occur rarely enough to not be a problem.
'''


from pyqtgraph import GraphicsScene, GraphicsView, ImageView, ViewBox, getConfigOption
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from pyqtgraph.widgets.HistogramLUTWidget import HistogramLUTWidget
from pyqtgraph.widgets.PlotWidget import PlotWidget
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from pyqtgraph.graphicsItems.ROI import ROI
from pyqtgraph.graphicsItems.InfiniteLine import InfiniteLine
from pyqtgraph.graphicsItems.LinearRegionItem import LinearRegionItem
from pyqtgraph.graphicsItems.VTickGroup import VTickGroup
from pyqtgraph.SignalProxy import SignalProxy
from time import perf_counter_ns

getMillis = lambda: perf_counter_ns() // 10 ** 6

class GraphicsSceneOverride(GraphicsScene):
    sigMousePressMove = QtCore.Signal(object)
    sigMousePressAltMove = QtCore.Signal(object)
    sigMouseMoved = QtCore.Signal(object)
    sigMouseRelease = QtCore.Signal(object)
    sigMouseReleaseAlt = QtCore.Signal(object)
    def __init__(self, *args, **kwargs):
        super(GraphicsSceneOverride, self).__init__(*args, **kwargs)


    def mouseMoveEvent(self, ev):
        super(GraphicsSceneOverride, self).mouseMoveEvent(ev)
        if (ev.buttons() & QtCore.Qt.MouseButton.LeftButton):
            self.sigMousePressMove.emit(ev.scenePos())
        elif (ev.buttons() & QtCore.Qt.MouseButton.RightButton):
            self.sigMousePressAltMove.emit(ev.scenePos())
        else:
            self.sigMouseMoved.emit(ev.scenePos())

    def mouseReleaseEvent(self, ev):
        super(GraphicsSceneOverride, self).mouseReleaseEvent(ev)
        if (ev.button() == QtCore.Qt.MouseButton.LeftButton):
            self.sigMouseRelease.emit(ev)
        elif (ev.button() == QtCore.Qt.MouseButton.RightButton):
            self.sigMouseReleaseAlt.emit(ev)
        

class PlotROI(ROI):
    def __init__(self, size):
        ROI.__init__(self, pos=[0,0], size=size) #, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])
        self.addRotateHandle([0, 0], [0.5, 0.5])


class GraphicsViewOverride(GraphicsView):    

    def __init__(self, parent=None, useOpenGL=None, background='default'):        
        self.closed = False
        
        QtWidgets.QGraphicsView.__init__(self, parent)
        
        # This connects a cleanup function to QApplication.aboutToQuit. It is
        # called from here because we have no good way to react when the
        # QApplication is created by the user.
        # See pyqtgraph.__init__.py
        from pyqtgraph import _connectCleanup
        _connectCleanup()
        
        if useOpenGL is None:
            useOpenGL = getConfigOption('useOpenGL')
        
        self.useOpenGL(useOpenGL)
        self.setCacheMode(self.CacheModeFlag.CacheBackground)
        
        ## This might help, but it's probably dangerous in the general case..
        #self.setOptimizationFlag(self.DontSavePainterState, True)
        
        self.setBackgroundRole(QtGui.QPalette.ColorRole.NoRole)
        self.setBackground(background)
        
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        
        
        self.lockedViewports = []
        self.lastMousePos = None
        self.setMouseTracking(True)
        self.aspectLocked = False
        self.range = QtCore.QRectF(0, 0, 1, 1)
        self.autoPixelRange = True
        self.currentItem = None
        self.clearMouse()
        self.updateMatrix()
        # GraphicsScene must have parent or expect crashes!
        self.sceneObj = GraphicsSceneOverride(parent=self)
        self.setScene(self.sceneObj)
        
        ## by default we set up a central widget with a grid layout.
        ## this can be replaced if needed.
        self.centralWidget = None
        self.setCentralItem(QtWidgets.QGraphicsWidget())
        self.centralLayout = QtWidgets.QGraphicsGridLayout()
        self.centralWidget.setLayout(self.centralLayout)
        
        self.mouseEnabled = False
        self.scaleCenter = False  ## should scaling center around view center (True) or mouse click (False)
        self.clickAccepted = False


class ImageViewOverride(ImageView):
    def __init__(self, parent=None, name="ImageView", view=None, imageItem=None, 
                 levelMode='mono', *args):
        QtWidgets.QWidget.__init__(self, parent, *args)
        self._imageLevels = None  # [(min, max), ...] per channel image metrics
        self.levelMin = None    # min / max levels across all channels
        self.levelMax = None
        
        self.name = name
        self.image = None
        self.axes = {}
        self.imageDisp = None
        self.ui = Ui_Form_Override()
        self.ui.setupUi(self)
        self.scene = self.ui.graphicsView.scene()
        self.last_pos = [1, 1]

        self.scene.sigMouseReleaseAlt.connect(self.update_last_pos)
        
        
        self.ignorePlaying = False
        
        if view is None:
            self.view = ViewBox()
        else:
            self.view = view
        self.ui.graphicsView.setCentralItem(self.view)
        self.view.setAspectLocked(True)
        self.view.invertY()
        
        self.menu = None
        
        self.ui.normGroup.hide()

        self.roi = PlotROI(10)
        self.roi.setZValue(20)
        self.view.addItem(self.roi)
        self.roi.hide()
        self.normRoi = PlotROI(10)
        self.normRoi.setPen('y')
        self.normRoi.setZValue(20)
        self.view.addItem(self.normRoi)
        self.normRoi.hide()
        self.roiCurves = []
        self.timeLine = InfiniteLine(0, movable=True)
        if getConfigOption('background')=='w':
            self.timeLine.setPen((20, 80, 80, 200))
        else:
            self.timeLine.setPen((255, 255, 0, 200))
        self.timeLine.setZValue(1)
        self.ui.roiPlot.addItem(self.timeLine)
        self.ui.splitter.setSizes([self.height()-35, 35])

        # init imageItem and histogram
        if imageItem is None:
            self.imageItem = ImageItem()
        else:
            self.imageItem = imageItem
            self.setImage(imageItem.image, autoRange=False, autoLevels=False, transform=imageItem.transform())
        self.view.addItem(self.imageItem)
        self.currentIndex = 0
        
        self.ui.histogram.setImageItem(self.imageItem)
        self.ui.histogram.setLevelMode(levelMode)
        
        # make splitter an unchangeable small grey line:
        s = self.ui.splitter
        s.handle(1).setEnabled(False)
        s.setStyleSheet("QSplitter::handle{background-color: grey}")
        s.setHandleWidth(2)

        self.ui.roiPlot.hideAxis('left')
        self.frameTicks = VTickGroup(yrange=[0.8, 1], pen=0.4)
        self.ui.roiPlot.addItem(self.frameTicks, ignoreBounds=True)
        
        self.keysPressed = {}
        self.playTimer = QtCore.QTimer()
        self.playRate = 0
        self.fps = 1 # 1 Hz by default
        self.lastPlayTime = 0
        
        self.normRgn = LinearRegionItem()
        self.normRgn.setZValue(0)
        self.ui.roiPlot.addItem(self.normRgn)
        self.normRgn.hide()
            
        ## wrap functions from view box
        for fn in ['addItem', 'removeItem']:
            setattr(self, fn, getattr(self.view, fn))

        ## wrap functions from histogram
        for fn in ['setHistogramRange', 'autoHistogramRange', 'getLookupTable', 'getLevels']:
            setattr(self, fn, getattr(self.ui.histogram, fn))

        self.timeLine.sigPositionChanged.connect(self.timeLineChanged)
        self.ui.roiBtn.clicked.connect(self.roiClicked)
        self.roi.sigRegionChanged.connect(self.roiChanged)
        #self.ui.normBtn.toggled.connect(self.normToggled)
        self.ui.menuBtn.clicked.connect(self.menuClicked)
        self.ui.normDivideRadio.clicked.connect(self.normRadioChanged)
        self.ui.normSubtractRadio.clicked.connect(self.normRadioChanged)
        self.ui.normOffRadio.clicked.connect(self.normRadioChanged)
        self.ui.normROICheck.clicked.connect(self.updateNorm)
        self.ui.normFrameCheck.clicked.connect(self.updateNorm)
        self.ui.normTimeRangeCheck.clicked.connect(self.updateNorm)
        self.playTimer.timeout.connect(self.timeout)
        
        self.normProxy = SignalProxy(self.normRgn.sigRegionChanged, slot=self.updateNorm)
        self.normRoi.sigRegionChangeFinished.connect(self.updateNorm)
        
        self.ui.roiPlot.registerPlot(self.name + '_ROI')
        self.view.register(self.name)
        
        self.noRepeatKeys = [QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_PageUp, QtCore.Qt.Key.Key_PageDown]
        
        self.roiClicked() ## initialize roi plot to correct shape / visibility

    def update_last_pos(self, ev):
        point = self.getImageItem().mapFromScene(ev.scenePos())
        self.last_pos = [int(point.x()), int(point.y())]

class Ui_Form_Override(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(726, 588)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.splitter = QtWidgets.QSplitter(Form)
        self.splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView = GraphicsViewOverride(self.layoutWidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 2, 1)
        self.histogram = HistogramLUTWidget(self.layoutWidget)
        self.histogram.setObjectName("histogram")
        self.gridLayout.addWidget(self.histogram, 0, 1, 1, 2)
        self.roiBtn = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.roiBtn.sizePolicy().hasHeightForWidth())
        self.roiBtn.setSizePolicy(sizePolicy)
        self.roiBtn.setCheckable(True)
        self.roiBtn.setObjectName("roiBtn")
        self.gridLayout.addWidget(self.roiBtn, 1, 1, 1, 1)
        self.menuBtn = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.menuBtn.sizePolicy().hasHeightForWidth())
        self.menuBtn.setSizePolicy(sizePolicy)
        self.menuBtn.setObjectName("menuBtn")
        self.gridLayout.addWidget(self.menuBtn, 1, 2, 1, 1)
        self.roiPlot = PlotWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.roiPlot.sizePolicy().hasHeightForWidth())
        self.roiPlot.setSizePolicy(sizePolicy)
        self.roiPlot.setMinimumSize(QtCore.QSize(0, 40))
        self.roiPlot.setObjectName("roiPlot")
        self.gridLayout_3.addWidget(self.splitter, 0, 0, 1, 1)
        self.normGroup = QtWidgets.QGroupBox(Form)
        self.normGroup.setObjectName("normGroup")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.normGroup)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.normSubtractRadio = QtWidgets.QRadioButton(self.normGroup)
        self.normSubtractRadio.setObjectName("normSubtractRadio")
        self.gridLayout_2.addWidget(self.normSubtractRadio, 0, 2, 1, 1)
        self.normDivideRadio = QtWidgets.QRadioButton(self.normGroup)
        self.normDivideRadio.setChecked(False)
        self.normDivideRadio.setObjectName("normDivideRadio")
        self.gridLayout_2.addWidget(self.normDivideRadio, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.normGroup)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.normGroup)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.normGroup)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)
        self.normROICheck = QtWidgets.QCheckBox(self.normGroup)
        self.normROICheck.setObjectName("normROICheck")
        self.gridLayout_2.addWidget(self.normROICheck, 1, 1, 1, 1)
        self.normXBlurSpin = QtWidgets.QDoubleSpinBox(self.normGroup)
        self.normXBlurSpin.setObjectName("normXBlurSpin")
        self.gridLayout_2.addWidget(self.normXBlurSpin, 2, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.normGroup)
        self.label_8.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 2, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.normGroup)
        self.label_9.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 2, 3, 1, 1)
        self.normYBlurSpin = QtWidgets.QDoubleSpinBox(self.normGroup)
        self.normYBlurSpin.setObjectName("normYBlurSpin")
        self.gridLayout_2.addWidget(self.normYBlurSpin, 2, 4, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.normGroup)
        self.label_10.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignTrailing|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 2, 5, 1, 1)
        self.normOffRadio = QtWidgets.QRadioButton(self.normGroup)
        self.normOffRadio.setChecked(True)
        self.normOffRadio.setObjectName("normOffRadio")
        self.gridLayout_2.addWidget(self.normOffRadio, 0, 3, 1, 1)
        self.normTimeRangeCheck = QtWidgets.QCheckBox(self.normGroup)
        self.normTimeRangeCheck.setObjectName("normTimeRangeCheck")
        self.gridLayout_2.addWidget(self.normTimeRangeCheck, 1, 3, 1, 1)
        self.normFrameCheck = QtWidgets.QCheckBox(self.normGroup)
        self.normFrameCheck.setObjectName("normFrameCheck")
        self.gridLayout_2.addWidget(self.normFrameCheck, 1, 2, 1, 1)
        self.normTBlurSpin = QtWidgets.QDoubleSpinBox(self.normGroup)
        self.normTBlurSpin.setObjectName("normTBlurSpin")
        self.gridLayout_2.addWidget(self.normTBlurSpin, 2, 6, 1, 1)
        self.gridLayout_3.addWidget(self.normGroup, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "PyQtGraph"))
        self.roiBtn.setText(_translate("Form", "ROI"))
        self.menuBtn.setText(_translate("Form", "Menu"))
        self.normGroup.setTitle(_translate("Form", "Normalization"))
        self.normSubtractRadio.setText(_translate("Form", "Subtract"))
        self.normDivideRadio.setText(_translate("Form", "Divide"))
        self.label_5.setText(_translate("Form", "Operation:"))
        self.label_3.setText(_translate("Form", "Mean:"))
        self.label_4.setText(_translate("Form", "Blur:"))
        self.normROICheck.setText(_translate("Form", "ROI"))
        self.label_8.setText(_translate("Form", "X"))
        self.label_9.setText(_translate("Form", "Y"))
        self.label_10.setText(_translate("Form", "T"))
        self.normOffRadio.setText(_translate("Form", "Off"))
        self.normTimeRangeCheck.setText(_translate("Form", "Time range"))
        self.normFrameCheck.setText(_translate("Form", "Frame"))