from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QHBoxLayout, QWidget, QComboBox, QTableWidget, 
                             QTableWidgetItem, QMainWindow, QGridLayout, QPushButton,QMenu,QGroupBox,QCheckBox)
from PyQt5 import QtCore
import pyqtgraph as pg
from core.genetic_algorithm import Genetic_Algorithm  
import pandas as pd
import numpy as np


class GAWindowWidget(QWidget):
    def __init__(self, main_ref: QMainWindow, ga: Genetic_Algorithm, parent=None):
        super().__init__(parent)
        self.main_ref = main_ref
        self.ga = ga
        self.name = "Genetic Algorithm"
        self.length = len(self.ga.examples)
        self.setWindowTitle("Genetic Algorithm Results")
        self.cocaine_view = pg.GraphicsLayoutWidget(title="Cocaine")
        self.saline_view = pg.GraphicsLayoutWidget(title="Saline")
        self.current_page = 0

        # Trace Visualization
        layout_trace = QHBoxLayout()
        layout_trace.addWidget(self.cocaine_view)
        layout_trace.addWidget(self.saline_view)

        # Dropdown
        layout_dropdown = QHBoxLayout()
        self.rank_dropdown = QComboBox()
        self.rank_dropdown.addItems([f"Rank {i+1}" for i in range(self.length)])
        self.rank_dropdown.setCurrentIndex(0)
        self.rank_dropdown.currentIndexChanged.connect(self.update)
        layout_dropdown.addWidget(self.rank_dropdown)
        

        button_next_page = QPushButton("Next Page")
        layout_dropdown.addWidget(button_next_page)
        button_next_page.clicked.connect(self.update_traces)


        # Table of top 5 values
        layout_table = QVBoxLayout()
        label_table = QLabel(f"Top {self.length} Parameters for {ga.event_type}")
        table = QTableWidget()
        table.setRowCount(self.length)
        table.setColumnCount(5)
        table.verticalHeader().setVisible(False)
        data = {"Rank": [f"Rank {i+1}" for i in range(self.length)],
                "Fitness": ['{:.4f}'.format(ga._best_fitness[i].item()) for i in range(self.length)],
                "PreBinNum": [str(ga.preBinNum[i].item()) for i in range(self.length)],
                "PostBinNum": [str(ga.postBinNum[i].item()) for i in range(self.length)],
                "Bin Size": [str(ga.binSize[i].item()) for i in range(self.length)]}
        horHeaders = []
        for n, key in enumerate(data.keys()):
            horHeaders.append(key)
            for m, item in enumerate(data[key]):
                newitem = QTableWidgetItem(item)
                table.setItem(m, n, newitem)
        table.setHorizontalHeaderLabels(horHeaders)
        layout_table.addWidget(label_table)
        layout_table.addWidget(table)

        # Button for table
        button_table = QPushButton('Show detail',self)
        layout_button_table = QHBoxLayout()
        layout_button_table.addWidget(button_table)      
        button_table.clicked.connect(self.showDetails)
   
        button_figure = QPushButton('Show figure',self)
        layout_button_table.addWidget(button_figure)      
        button_table.clicked.connect(self.showfigure)
        
        # Table of result
        layout_res_table = QVBoxLayout()
        self.res_table = QTableWidget()
        layout_res_table.addWidget(self.res_table)
        
        # Histogram
        # mouseID, day, session, unit_id
        self.hist_view = pg.GraphicsLayoutWidget(title="histogram")
        layout_hist = QVBoxLayout()
        layout_hist.addWidget(self.hist_view)
        # self.histogram
        
        
        self.update()

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(layout_dropdown)
        layout.addLayout(layout_trace)
        layout.addLayout(layout_table)
        layout.addLayout(layout_button_table)
        # layout.addLayout(layout_res_table)
        # layout.addLayout(layout_hist)
        self.setLayout(layout)
    
    def showDetails(self):
        self.detail_win = GADetailTableWindowWidget(self,self.ga)
        self.detail_win.show()
    
    def showfigure(self):
        self.figure_win = GADetailFigureWindowWidget(self,self.ga,res_df=self.res_df)
        self.figure_win.show()

    def update_traces(self):
        if self.current_page<self.total_pages:
            self.current_page+=1
        self.update()


    def update(self):
        index = self.rank_dropdown.currentIndex()
        cocaine_traces = self.ga.traces[index]["Cocaine"]
        saline_traces = self.ga.traces[index]["Saline"]
        # # Example version
        # cocaine_traces = self.ga.examples[index]["Cocaine"]
        # saline_traces = self.ga.examples[index]["Saline"]
        cocaine_traces_number = len(cocaine_traces)
        print(cocaine_traces_number)
        # There are 20 traces in one page
        traces_per_page = 20
        self.total_pages = cocaine_traces_number//traces_per_page

        self.cocaine_view.clear()
        self.saline_view.clear()
        # We'll set a label that spans the middle of the grid
        cocaine_label = pg.LabelItem(justify='center')
        cocaine_label.setText("Cocaine Traces")
        cocaine_page_label = pg.LabelItem(justify = 'center')
        cocaine_page_label.setText("Current Page/Total Page: {}/{}".format(self.current_page+1,self.total_pages+1))
        saline_label = pg.LabelItem(justify='center')
        saline_label.setText("Saline Traces")
        saline_page_label = pg.LabelItem(justify = 'center')
        saline_page_label.setText("Current Page/Total Page: {}/{}".format(self.current_page+1,self.total_pages+1))
        self.cocaine_view.addItem(cocaine_label, row=0, col=0)
        self.cocaine_view.addItem(cocaine_page_label, row = 0, col = 1)
        self.saline_view.addItem(saline_label, row=0, col=0)
        self.saline_view.addItem(saline_page_label, row = 0, col = 1)
        # In both cases we'll create a 5 by 4 grid
        for i in range(5):
            for j in range(4):
                if self.current_page*traces_per_page+i*4+j< cocaine_traces_number:
                    cocaine_item = self.cocaine_view.addPlot(row=i+1, col=j)
                    cocaine_item.plot(self.ga.xvalues[index]["Cocaine"][self.current_page*traces_per_page+i*4+j], cocaine_traces[self.current_page*traces_per_page+i*4+j])
                    saline_item = self.saline_view.addPlot(row=i+1, col=j)
                    saline_item.plot(self.ga.xvalues[index]["Saline"][self.current_page*traces_per_page+i*4+j], saline_traces[self.current_page*traces_per_page+i*4+j])
        
        # Example version
        # for i in range(5):
        #     for j in range(4):
        #         if self.current_page*traces_per_page+i*4+j< cocaine_traces_number:
        #             cocaine_item = self.cocaine_view.addPlot(row=i+1, col=j)
        #             cocaine_item.plot(self.ga.example_xvalues[index]["Cocaine"][self.current_page*traces_per_page+i*4+j], cocaine_traces[self.current_page*traces_per_page+i*4+j])
        #             saline_item = self.saline_view.addPlot(row=i+1, col=j)
        #             saline_item.plot(self.ga.example_xvalues[index]["Saline"][self.current_page*traces_per_page+i*4+j], saline_traces[self.current_page*traces_per_page+i*4+j])
        
        # # result table
        # self.res_df = self.ga.calculate_data(self.ga.preBinNum[index],self.ga.postBinNum[index],self.ga.binSize[index],'RNFS')
        # print(self.res_df.dtypes)
        # self.res_table.setColumnCount(len(self.res_df.columns))
        # self.res_table.setHorizontalHeaderLabels(self.res_df.columns)
        # self.res_table.setRowCount(len(self.res_df.index))
        # for i in range(len(self.res_df.index)):
        #     for j in range(len(self.res_df.columns)):
        #         self.res_table.setItem(i, j, QTableWidgetItem(str(self.res_df.iloc[i,j])))

        
        
    
    def closeEvent(self, event):
        super(GAWindowWidget, self).closeEvent(event)
        self.main_ref.remove_window(self.name)


class GAGenerationScoreWindowWidget(QWidget):
    def __init__(self, main_ref: QMainWindow, ga: Genetic_Algorithm, parent=None):
        super().__init__(parent)
        self.main_ref = main_ref
        self.ga = ga
        self.name = "Genetic Algorithm Average Score"
        self.setWindowTitle("Genetic Algorithm Average Score.")

        self.layout_plot = QGridLayout()
        self.plot_win = pg.PlotWidget()
        self.plot_win.showGrid(x=True,y=True)
        
        self.layout_plot.addWidget(self.plot_win)
        self.plot_win.setYRange(max=1,min=0)
        self.plot_win.addLegend()
        average_accuracy = self.plot_win.plot(self.ga.curve,pen='r',name = 'average accuracy')
        average_f1Curve = self.plot_win.plot(self.ga.f1Curve,pen='g',name = 'average F1 score')

        self.plot_win.setLabel('bottom','Generation')
        self.plot_win.setLabel('left','Score')
        # self.plot_view.addLegend()
        # Layout
        layout = QVBoxLayout()
        layout.addLayout(self.layout_plot)
        ax = self.plot_win.getAxis('right')
        ticks = [x for x in range(ga.max_generation)]
        ax.setTicks([[(v, str(v)) for v in ticks ]])
        self.setLayout(layout)
        print('curve')
        
        # self.update()


    # def update(self):
    #     self.timer = QtCore.QTimer(self)
    #     self.timer.timeout.connect(self.get_generation_info)
    #     self.timer.start(60000)

    # def get_generation_info(self):
    #     self.plot_plt.plot().setData(self.ga.curve,pen='r')


class GADetailTableWindowWidget(QWidget):
    def __init__(self, main_ref: QMainWindow, ga: Genetic_Algorithm, parent=None):
        super(GADetailTableWindowWidget,self).__init__(parent)
        self.main_ref = main_ref
        self.ga = ga
        self.length = len(self.ga.examples)
        self.resize(200,200)
        self.name = "Detail Table"
        self.setWindowTitle("Detail Table")

        # Dropdown for rank
        layout_dropdown = QHBoxLayout()
        self.rank_dropdown = QComboBox()
        self.rank_dropdown.addItems([f"Rank {i+1}" for i in range(self.length)])
        self.rank_dropdown.setCurrentIndex(0)
        self.rank_dropdown.currentIndexChanged.connect(self.update)
        layout_dropdown.addWidget(self.rank_dropdown)

        # dropdown for events
        layout_event_dropdown = QHBoxLayout()
        self.event_dropdown = QComboBox()
        self.event_dropdown.addItems(["IALP","ALP","RNFS","ALP_Timeout"])
        self.event_dropdown.setCurrentIndex(0)
        self.event_dropdown.currentIndexChanged.connect(self.update)
        layout_dropdown.addWidget(self.event_dropdown)    


        # Table of result
        layout_res_table = QVBoxLayout()
        self.res_table = QTableWidget()
        self.res_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.res_table.customContextMenuRequested.connect(self.generateMenu)
        layout_res_table.addWidget(self.res_table)

        self.update()

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(layout_dropdown)
        layout.addLayout(layout_event_dropdown)
        layout.addLayout(layout_res_table)
        self.setLayout(layout)

    def generateMenu(self):
        menu = QMenu()

    def update(self):
        index = self.rank_dropdown.currentIndex()
        event_type = self.event_dropdown.currentText()

        # result table
        self.res_df = self.ga.calculate_data(self.ga.preBinNum[index],self.ga.postBinNum[index],self.ga.binSize[index],event_type)
        print(self.res_df.dtypes)
        self.res_table.setColumnCount(len(self.res_df.columns))
        self.res_table.setHorizontalHeaderLabels(self.res_df.columns)
        self.res_table.setRowCount(len(self.res_df.index))
        for i in range(len(self.res_df.index)):
            for j in range(len(self.res_df.columns)):
                self.res_table.setItem(i, j, QTableWidgetItem(str(self.res_df.iloc[i,j])))



class GADetailFigureWindowWidget(QWidget):
    def __init__(self, main_ref: QMainWindow, ga: Genetic_Algorithm,res_df, parent=None):
        super(GADetailFigureWindowWidget,self).__init__(parent)
        self.main_ref = main_ref
        self.ga = ga
        self.length = len(self.ga.examples)
        self.resize(200,200)
        self.name = "Figure"
        self.res_df = res_df
        self.setWindowTitle("Figure")

        #checkboxs
        groupBox = QGroupBox("Define group")
        groupBox.setFlat(False)
        layout_groupBox = QHBoxLayout()
        groupBox.setLayout(layout_groupBox)
        

        self.checkbox_group = QCheckBox("Group")
        self.checkbox_group.setChecked(True)
        
        self.checkbox_day = QCheckBox("Day")
        self.checkbox_day.setChecked(False)

        self.checkbox_session = QCheckBox("Session")
        self.checkbox_session.setChecked(False)

        layout_groupBox.addWidget(self.checkbox_group)
        layout_groupBox.addWidget(self.checkbox_day)
        layout_groupBox.addWidget(self.checkbox_session)


        # start button
        button_start = QPushButton()
        button_start.setText("Update")
        layout_button = QHBoxLayout()
        layout_button.addWidget(button_start)
        button_start.clicked.connect(self.update_group)

        # figure type
        layout_dropdown = QHBoxLayout()
        label_dropdown = QLabel("Figure Type: ")
        self.dropdown = QComboBox()
        self.dropdown.addItems(["Histogram","Line"])
        layout_dropdown.addWidget(label_dropdown)
        layout_dropdown.addWidget(self.dropdown)

        # Figure
        self.figure_view = pg.GraphicsLayoutWidget(title = "figure")
        layout_figure = QVBoxLayout()
        layout_figure.addWidget(self.figure_view)

        layout = QVBoxLayout()
        layout.addWidget(groupBox)
        layout.addLayout(layout_dropdown)
        layout.addLayout(layout_button)
        layout.addLayout(layout_figure)
        self.setLayout(layout)

    def update_group(self):
        color_list = ['b','g','r','c','m','y','k','w','d','l','s']
        
        self.figure_view.clear()
        
        checks = []
        if self.checkbox_group.isChecked() == True:
            checks.append('group')
        if self.checkbox_day.isChecked() == True:
            checks.append('day')
        if self.checkbox_session.isChecked() == True:
            checks.append('session')
        head_hist = self.res_df.columns.values[6:]
        x_dict_line = dict(enumerate(head_hist))
        x_axis_line = [(i,list(head_hist)[i]) for i in range(len(head_hist))] 
        binNumberAxis_line = pg.AxisItem(orientation='bottom')
        binNumberAxis_line.setTicks([x_axis_line,x_dict_line.items()]) 
        

        # Histogram
        if self.dropdown.currentText()=='Histogram':            
            print(checks)
            if not checks:
                hist_plot = self.figure_view.addPlot(axisItems={'bottom': binNumberAxis_line})
                hist_plot.addLegend()

                ave = self.res_df[head_hist].mean()
                sem = self.res_df[head_hist].sem()
                bar = pg.BarGraphItem(x = np.asarray(list(x_dict_line.keys())), width = 0.5, height=ave, pen='w', brush=(0,0,255,150))
                errorbar = pg.ErrorBarItem(x = np.asarray(list(x_dict_line.keys())), y = np.asarray(ave), top = np.asarray(sem), bottom = np.asarray(sem),beam = 0.5, pen = 'w')
                hist_plot.addItem(bar)
                hist_plot.addItem(errorbar)
                
            else:
                
                hist = self.res_df
                ave = hist.groupby(checks)[head_hist].mean()

                group_number = len(hist.groupby(checks))
                x_dict_hist = {}
                for idx, binNumber in enumerate(head_hist):
                    for n in range(group_number):
                        x_dict_hist[idx*group_number+n] = binNumber
                x_axis_hist = [list(x_dict_hist.items())[i] for i in range(group_number//2,len(list(x_dict_hist.items())),group_number)] 
                binNumberAxis_host = pg.AxisItem(orientation='bottom')
                binNumberAxis_host.setTicks([x_axis_hist,x_dict_hist.items()]) 
                hist_plot = self.figure_view.addPlot(axisItems={'bottom': binNumberAxis_host})
                hist_plot.addLegend()
                print(ave)
                print(ave.index.values)
                sem = hist.groupby(checks)[head_hist].sem()
                print(sem)


                x = np.array([i for i in range(len(head_hist))])

                for i, group_name in enumerate(ave.index.values):
                    
                    bar = pg.BarGraphItem(x=x*group_number+i,width = 1,height = ave.loc[group_name,:],pen = 'w', brush = color_list[i],name = group_name)
                    
                    # bar = pg.BarGraphItem(x=np.asarray(x)+i*1/len(ave.index.values),width = 1/len(ave.index.values),height = ave.loc[group_name,:],pen = 'w', brush = color_list[i],name = group_name)
                    hist_plot.addItem(bar)
                    # legend.addItem(bar, str(group_name))
                for i, group_name in enumerate(sem.index.values):
                    errorbar = pg.ErrorBarItem(x = x*group_number+i, y = np.asarray(ave.loc[group_name,:]), top = np.asarray(sem.loc[group_name,:]), bottom = np.asarray(sem.loc[group_name,:]),beam = 1/(3*(len(ave.index.values))), pen = 'w')
                    # errorbar = pg.ErrorBarItem(x = np.asarray(x)+i*1/len(ave.index.values), y = np.asarray(ave.loc[group_name,:]), top = np.asarray(sem.loc[group_name,:]), bottom = np.asarray(sem.loc[group_name,:]),beam = 1/(3*(len(ave.index.values))), pen = 'w')
                    hist_plot.addItem(errorbar)
                # hist_plot.addLegend()
        elif self.dropdown.currentText() == 'Line':
            line_plot = self.figure_view.addPlot(axisItems={'bottom': binNumberAxis_line})
            line_plot.addLegend()
            if not checks:
                ave = self.res_df[head_hist].mean()
                sem = self.res_df[head_hist].sem()
                # x = []
                # for i in range(len(head_hist)):
                #     x.append(i)
                line_plot.plot(list(x_dict_line.keys()),ave)
                errorbar = pg.ErrorBarItem(x = np.asarray(list(x_dict_line.keys())), y = np.asarray(ave), top = np.asarray(sem), bottom = np.asarray(sem),beam = 0.5, pen = 'w')
                line_plot.addItem(errorbar)
                # legend.addItem(bar, 'total')
            else:
                hist = self.res_df
                ave = hist.groupby(checks)[head_hist].mean()
                print(ave)
                print(ave.index.values)
                sem = hist.groupby(checks)[head_hist].sem()
                print(sem)

                
                x = []
                for i in range(len(head_hist)):
                    x.append(i)
                for i, group_name in enumerate(ave.index.values):
                    line_plot.plot(x,ave.loc[group_name,:],pen = color_list[i], name = group_name)
                    # legend.addItem(bar, str(group_name))
                for i, group_name in enumerate(sem.index.values):
                    errorbar = pg.ErrorBarItem(x = np.asarray(list(x_dict_line.keys())), y = np.asarray(ave.loc[group_name,:]), top = np.asarray(sem.loc[group_name,:]), bottom = np.asarray(sem.loc[group_name,:]),beam = 1/(3*(len(ave.index.values))), pen = 'w')
                    line_plot.addItem(errorbar)
                
            
                
    def update_figure(self):
        pass

        