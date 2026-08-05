import numpy as np
import pandas as pd
from pathlib import Path
import os

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import (QApplication, 
    QWidget, 
    QFrame, 
    QHBoxLayout,
    QSplitter, 
    QTableView,
    QStyledItemDelegate,
    QItemDelegate,
    QCheckBox,
    QPushButton,
    QGridLayout,
    QAbstractItemView,
    QLabel,
    QStatusBar,
    QVBoxLayout,
    QWidget)

from PyQt5.QtCore import Qt

import time
import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

plt.rcParams["font.family"] = "calibri"
plt.rcParams["font.size"] = 14

BOTTOM_TABLE_LABELS = ["Frequency", "Damping", "Order", "Type"]
BOTTOM_TABLE_UNITS = ["Hz", "%", "/", "/"]

class ReadOnlyDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        return


class FloatDelegate(QItemDelegate):
    def __init__(self, decimals, parent=None):
        QItemDelegate.__init__(self, parent=parent)
        self.nDecimals = decimals

    def paint(self, painter, option, index):
        value = index.model().data(index, Qt.EditRole)
        try:
            number = float(value)
            painter.drawText(option.rect, Qt.AlignCenter, "{:.{}f}".format(number, self.nDecimals))
        except :
            QItemDelegate.paint(self, painter, option, index)


class App(QtWidgets.QMainWindow):
    def __init__(self, modal_id, colors = ["#A80F0A", "#E0423D", "#428D8F", "#2A787A"]):
        super(App, self).__init__()

        self.modal_id = modal_id

        self.colors = colors
        self.title = "Stability chart"
        self.point_size = 20
        self.pick_radius_size = 15
        self.click_time_old = time.time() 
        self.time_between_clicks = 0.6 # seconds
        self.FRF_transparency = 0.5
        self.show_cursor = True

        wid = QWidget(self)
        self.setCentralWidget(wid)
        hbox = QHBoxLayout()
        wid.setLayout(hbox)

        # Data definition
        self.selected_poles = []
        self.selected_ind = [] # indices of selected poles with respect to the order of the imputed poles 
        self.get_modal_data()

        self.freq_cmif = self.modal_id.freq#np.load("freq.npy")
        self.FRF_matrix = self.modal_id.FRF #np.load("Y.npy")
        self.poles = self.modal_id.stab_plot#np.load("for_plot.npy") # poles import

        self.FRF_cmif = np.linalg.svd(self.FRF_matrix)[1]

        #Model order / frequency / damping / pole type
        # Pole type:
        #   -   0 ... stable frequency, stable damping
        #   -   1 ... stable frequency, unstable samping
        #   -   2 ... unstable frequency, unstable damping

        reference_index = np.arange(self.poles.shape[0])
        all_0 = self.poles[:, -1]==3. # stab. freq, damp, mpf
        all_1 = self.poles[:, -1]==2. # stab. freq and damp
        all_2 = self.poles[:, -1]==1. # stab. freq
        all_3 = self.poles[:, -1]==0. # unstab.
        self.reference_index_0 = reference_index[all_0]
        self.reference_index_1 = reference_index[all_1]
        self.reference_index_2 = reference_index[all_2]
        self.reference_index_3 = reference_index[all_3]

        self.x_data_0 = self.poles[all_0, 1] # get frequency at stab. freq, damp, mpf
        self.y_data_0 = self.poles[all_0, 0] # get model order at stab. freq, damp, mpf
        self.damp_data_0 = self.poles[all_0, 2] # get damping at stab. freq, damp, mpf

        self.x_data_1 = self.poles[all_1, 1] # get frequency at stab. freq and damp
        self.y_data_1 = self.poles[all_1, 0] # get model order at stab. freq and damp
        self.damp_data_1 = self.poles[all_1, 2] # get damping at stab. freq and damp

        self.x_data_2 = self.poles[all_2, 1] # get frequency at stab. freq
        self.y_data_2 = self.poles[all_2, 0] # get model order at stab. freq
        self.damp_data_2 = self.poles[all_2, 2] # get damping at stab. freq

        self.x_data_3 = self.poles[all_3, 1] # get frequency at unstab.
        self.y_data_3 = self.poles[all_3, 0] # get model order at unstab.
        self.damp_data_3 = self.poles[all_3, 2] # get damping at unstab.

        self.no_out, self.no_input = self.FRF_matrix.shape[1:]

        self.style = """
        QSplitter::handle {
            image: url(:/images/splitter.png);
        }

        QSplitter::handle:horizontal {
            width: 2px;
        }

        QSplitter::handle:vertical {
            height: 2px;
        }

        QSplitter::handle:pressed {
            image: url(:/images/splitter_pressed.png);
        }
        """

        # Window geometry
        self.top = 150
        self.left = 300
        self.width = 1000
        self.height = 600

        verticalSpacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding) 

        # Objects layout
        #table
        self.tableView = QTableView()
        self.tableView.setFrameShape(QFrame.StyledPanel)
        self.table_model = QtGui.QStandardItemModel(0, 5)
        self.table_model.setSortRole(QtCore.Qt.UserRole)
        self.table_model.setHorizontalHeaderLabels(['Freq. [Hz]', 'Damp. [%]', 'Order', "Type", "Index"])
        self.tableView.setSortingEnabled(True)
        self.tableView.setModel(self.table_model)
        self.tableView.setColumnHidden(4, True) # for index to be exported
        self.float_delegate_3 = FloatDelegate(3, self.tableView)
        self.float_delegate_0 = FloatDelegate(0, self.tableView)
        header = self.tableView.horizontalHeader()       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch) # Stretch or ResizeToContents
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        delegate = ReadOnlyDelegate(self.tableView)
        self.tableView.setItemDelegateForColumn(0, delegate)
        self.tableView.setItemDelegateForColumn(1, delegate)
        self.tableView.setItemDelegateForColumn(2, delegate)
        self.tableView.setItemDelegateForColumn(3, delegate)        

        # always select full row
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)

        # FRF table:
        self.tableView_FRF = QTableView()
        self.tableView_FRF.setFrameShape(QFrame.StyledPanel)
        self.table_model_FRF = QtGui.QStandardItemModel(self.no_out, self.no_input)
        # self.no_input = 20
        self.table_model_FRF.setHorizontalHeaderLabels([str(_) for _ in np.arange(self.no_input)+1])
        self.tableView_FRF.setModel(self.table_model_FRF)
        
        header_FRF = self.tableView_FRF.horizontalHeader()   
        for i in range(self.no_input):
            header_FRF.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents) # ResizeToContents   Stretch
            self.tableView_FRF.setItemDelegateForColumn(i, delegate)

        # Delete button
        self.BtnDeleteSelected = QPushButton("Delete selected poles")
        self.BtnDeleteSelected.clicked.connect(self.delete_rows)

        self.BtnDeleteAll = QPushButton("Delete all")
        self.BtnDeleteAll.clicked.connect(self.delete_all)

        self.BtnExport = QPushButton("Export data")
        self.BtnExport.clicked.connect(self.export_data)

        editTableLayout = QHBoxLayout()
        editTableLayout.addWidget(self.BtnDeleteSelected)
        editTableLayout.addWidget(self.BtnDeleteAll)
        editTableLayout.addWidget(self.BtnExport)

        editTableWidget = QWidget()
        editTableWidget.setLayout(editTableLayout)

        # FRF display option
        displyFRFoptionsLayout = QVBoxLayout()
        displyFRFsuboptionsLayout = QHBoxLayout()
        self.display_cmif = QCheckBox("Display CMIF")
        self.display_cmif.setChecked(True)
        self.display_cmif.stateChanged.connect(self.plot_cmif)

        self.Btn_clear_FRF_display = QPushButton("Clear desplayed FRFs")
        self.Btn_clear_FRF_display.clicked.connect(self.clear_displayed_FRF)

        displyFRFsuboptionsLayout.addWidget(self.display_cmif)
        displyFRFsuboptionsLayout.addWidget(self.Btn_clear_FRF_display)
        displyFRFsuboptionsWidget = QWidget()
        displyFRFsuboptionsWidget.setLayout(displyFRFsuboptionsLayout)

        displyFRFoptionsLayout.addWidget(QLabel("Plot FRF:"))
        displyFRFoptionsLayout.addWidget(displyFRFsuboptionsWidget)

        # Check boxes poles to show
        optionsLayout = QVBoxLayout()

        self.option_0 = QCheckBox("Stable")
        self.option_1 = QCheckBox("Stable frequency and damping")
        self.option_2 = QCheckBox("Stable frequency")
        self.option_3 = QCheckBox("Unstable")
        self.option_0.stateChanged.connect(self.state_changed_0)
        self.option_1.stateChanged.connect(self.state_changed_1)
        self.option_2.stateChanged.connect(self.state_changed_2)
        self.option_3.stateChanged.connect(self.state_changed_3)
        self.option_0.setChecked(True)
        self.option_1.setChecked(True)
        self.option_2.setChecked(True)
        self.option_3.setChecked(False)

        optionsLayout.addItem(verticalSpacer)
        optionsLayout.addStretch()
        optionsLayout.addWidget(QLabel("Poles to show: "))
        optionsLayout.addWidget(self.option_0)
        optionsLayout.addWidget(self.option_1)
        optionsLayout.addWidget(self.option_2)
        optionsLayout.addWidget(self.option_3)

        optionsWidget = QWidget()
        optionsWidget.setLayout(optionsLayout)

        displyFRFoptionsWidget = QWidget()
        displyFRFoptionsWidget.setLayout(displyFRFoptionsLayout)

        # Types of reconstruction TODO!
        reconstructionLayout = QVBoxLayout()
        self.reconstruction_one = QCheckBox("Reconstruction one")
        self.reconstruction_two = QCheckBox("Reconstruction two")
        self.resonstruction_three = QCheckBox("Reconstruction three")
        # self.reconstruction_one.stateChanged.connect(self.state_changed_one)
        # self.reconstruction_two.stateChanged.connect(self.state_changed_two)
        # self.reconstruction_two.stateChanged.connect(self.state_changed_two)
        self.reconstruction_one.setChecked(True)
        self.reconstruction_two.setChecked(False)
        self.resonstruction_three.setChecked(False)

        reconstructionLayout.addItem(verticalSpacer)
        reconstructionLayout.addStretch()
        # reconstructionLayout.addWidget(QLabel("Type of reconstruction: "))
        # reconstructionLayout.addWidget(self.reconstruction_one)
        # reconstructionLayout.addWidget(self.reconstruction_two)
        # reconstructionLayout.addWidget(self.resonstruction_three)

        reconstructionWidget = QWidget()
        reconstructionWidget.setLayout(reconstructionLayout)

        # Info panel
        bottom_layout = QHBoxLayout()
        sub_info_layout = QVBoxLayout()
        info_layout = QGridLayout()
        self.row_0 = QLabel(" ")
        self.row_1 = QLabel(" ")
        self.row_2 = QLabel(" ")
        self.row_3 = QLabel(" ")
        self.row_0.setFixedSize(200, 16)
        self.row_1.setFixedSize(200, 16)
        self.row_2.setFixedSize(200, 16)
        self.row_3.setFixedSize(200, 16)
        self.row_0.setStyleSheet("border: 1px solid black;")
        self.row_1.setStyleSheet("border: 1px solid black;")
        self.row_2.setStyleSheet("border: 1px solid black;")
        self.row_3.setStyleSheet("border: 1px solid black;")
        self.row_0.setAlignment(Qt.AlignCenter)
        self.row_1.setAlignment(Qt.AlignCenter)
        self.row_2.setAlignment(Qt.AlignCenter)
        self.row_3.setAlignment(Qt.AlignCenter)

    
        for i, (label, unit) in enumerate(zip(BOTTOM_TABLE_LABELS, BOTTOM_TABLE_UNITS)):
            info_layout.addWidget(QLabel(label+": "), i, 0)
            info_layout.addWidget(QLabel(unit), i, 2)
        info_layout.addWidget(self.row_0, 0, 1)
        info_layout.addWidget(self.row_1, 1, 1)
        info_layout.addWidget(self.row_2, 2, 1) 
        info_layout.addWidget(self.row_3, 3, 1)
        infoWidget = QWidget()
        infoWidget.setLayout(info_layout)

        sub_info_layout.addItem(verticalSpacer)
        sub_info_layout.addStretch()
        sub_info_layout.addWidget(infoWidget)
        sub_info_Widget = QWidget()
        sub_info_Widget.setLayout(sub_info_layout)

        additional_options_layout = QVBoxLayout()
        self.show_selected_poles = QCheckBox("Selected poles")
        self.show_selected_poles.stateChanged.connect(self.state_changed_selected_poles)
        self.show_selected_poles.setChecked(True)
        
        self.show_cursor = QCheckBox("Show cursor")
        self.show_cursor.stateChanged.connect(self.show_hide_cursor)
        self.BtnAutoscaleLogAxis = QPushButton("Autoscale Amplitude")
        self.BtnAutoscaleLogAxis.clicked.connect(self.autoscale_log_axis)

        additional_options_layout.addWidget(self.show_selected_poles)
        additional_options_layout.addWidget(self.show_cursor)
        additional_options_layout.addWidget(self.BtnAutoscaleLogAxis)
        
        additional_options_Widget = QWidget()
        additional_options_Widget.setLayout(additional_options_layout)

        self.logo = QLabel(self)
        # loading image
        #set the pyFBS logo
        icon = str(Path(__file__).parents[1]) + os.sep + "data" + os.sep + "logo_new.png"
        self.pixmap = QtGui.QPixmap(icon)
        self.logo.setPixmap(self.pixmap)
        self.layout().setAlignment(self.logo, QtCore.Qt.AlignCenter)

        # self.logo.resize(int(self.pixmap.width()),
        #                   int(self.pixmap.height()))
        
        bottom_layout.addWidget(optionsWidget)
        bottom_layout.addWidget(reconstructionWidget)
        bottom_layout.addWidget(sub_info_Widget)
        bottom_layout.addWidget(additional_options_Widget)
        bottom_layout.addItem(verticalSpacer)
        bottom_layout.addStretch()# to push everything on the left size
        bottom_layout.addWidget(self.logo)
        # bottom_layout().setAlignment(self.logo, QtCore.Qt.AlignCenter)
        bottomWidget = QWidget()
        bottomWidget.setLayout(bottom_layout)

        # Plot
        content_plot = QWidget()

        # Split screen to 3 parts
        splitter0 = QSplitter(Qt.Vertical)
        splitter0.addWidget(self.tableView)
        splitter0.addWidget(editTableWidget)
        splitter0.addWidget(displyFRFoptionsWidget)
        splitter0.addWidget(self.tableView_FRF)
        splitter0.setSizes([500, 1, 1, 500])

        splitter1 = QSplitter(Qt.Vertical)
        splitter1.addWidget(content_plot)
        splitter1.addWidget(bottomWidget)
        splitter1.setStyleSheet(self.style)
        splitter1.setSizes([1000,1])
        
        splitter2 = QSplitter(Qt.Horizontal)
        splitter2.addWidget(splitter0)
        splitter2.addWidget(splitter1)
        splitter2.setStyleSheet(self.style)
        splitter2.setSizes([300,1000])

        hbox.addWidget(splitter2)

        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

        # plot
        # initialize plots
        self.fig, self.axlog = plt.subplots(facecolor='#F0F0F0', figsize=(10, 5)) #gray
        self.axlog.set_facecolor("#E1E1E1")
        self.fig.set_layout_engine('tight')

        self.ax = self.axlog.twinx()
        
        # self.ax.yaxis.set_label_position("right")
        
        self.ax.set_yticks(np.unique(self.poles[:, 0]))
        self.ax.set_axisbelow(True)
        self.ax.grid()

        # switch axis to have polynomial order on the left, while picking still works
        self.ax.set_ylabel("Polynomial order")
        self.ax.yaxis.set_label_position("left")
        self.ax.yaxis.tick_left()
        self.axlog.yaxis.set_label_position("right")
        self.axlog.yaxis.tick_right()
        self.axlog.set_xlabel('Frequency [Hz]')

        # plot data
        self.plot_cmif()
        self.plot_FRF = self.axlog.semilogy([], [])
        self.data_3 = self.ax.scatter(self.x_data_3, self.y_data_3, s=int(self.point_size/2), color=self.colors[0], marker='o', 
                                        picker=1, pickradius=self.pick_radius_size, 
                                        label='Unstable', alpha=0.8)
        self.data_2 = self.ax.scatter(self.x_data_2, self.y_data_2, s=self.point_size, color=self.colors[1], marker='o', 
                                        picker=1, pickradius=self.pick_radius_size,
                                        label='Stable freq.', alpha=0.8)
        self.data_1 = self.ax.scatter(self.x_data_1, self.y_data_1, s=self.point_size, color=self.colors[2], marker='o', 
                                        picker=1, pickradius=self.pick_radius_size,
                                        label='Stable freq. and damp.', alpha=0.8)
        self.data_0 = self.ax.scatter(self.x_data_0, self.y_data_0, s=self.point_size*1.2, color=self.colors[3], marker='o', 
                                        picker=1, pickradius=self.pick_radius_size, 
                                        edgecolors='k', linewidths=1, label='Stable', alpha=0.8) #
        
        self.selected_poles_plot = self.ax.scatter([], [], color='#FF0009', marker="x", s=self.point_size*5, lw=3)
        self.vlines = self.ax.scatter([], [], marker = '|', linewidths = 1, s=10**12, color='k')
        self.ax.legend(loc='lower right', ncol=4, framealpha=0.6)

        self.data_3.set_visible(self.option_3.isChecked())
        self.fig.canvas.draw() # redraw graph

        self.horizontal_line = self.ax.axhline(color='k', lw=0.5, ls='--')
        self.vertical_line = self.ax.axvline(color='k', lw=0.5, ls='--')
        self.set_cross_hair_visible(self.show_cursor.isChecked())
     
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

        # plot to widget
        self.plotWidget = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(content_plot)  
        lay.setContentsMargins(0, 0, 0, 0)      
        lay.addWidget(self.plotWidget)
        # add toolbar
        self.addToolBar(QtCore.Qt.TopToolBarArea, NavigationToolbar(self.plotWidget, self))
        
        # add status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        self.selected_FRF = []
        selection_model = self.tableView_FRF.selectionModel()
        selection_model.selectionChanged.connect(self.on_selectionChanged)

        self.get_plot_limits()
        self.showMaximized()
        
    def autoscale_log_axis(self):
        self.axlog.autoscale(axis='y')
        self.fig.canvas.draw() # redraw graph
        
    def plot_cmif(self):
        self.axlog.clear()
        if self.display_cmif.isChecked():
            self.cmif_plot = self.axlog.semilogy(self.freq_cmif, np.abs(self.FRF_cmif), c='k')
        
        self.axlog.set_ylabel("Amplitude")
        self.axlog.set_xlabel('Frequency [Hz]')
        self.axlog.yaxis.set_label_position("right")
        self.axlog.yaxis.tick_right()
        self.axlog.set_yscale('log')
        try:
            self.axlog.set_xlim(*self.x_lim_axlog)
            self.axlog.set_ylim(*self.y_lim_axlog)
            self.ax.set_xlim(*self.x_lim_ax)
            self.ax.set_ylim(*self.y_lim_ax)
        except: pass
        self.fig.canvas.draw() # redraw graph

    def clear_displayed_FRF(self):
        self.tableView_FRF.clearSelection()
        
    @QtCore.pyqtSlot('QItemSelection', 'QItemSelection')
    def on_selectionChanged(self, selected, deselected):
        for ix in selected.indexes():
            loc = [ix.row(), ix.column()]
            if loc in self.selected_FRF:
                pass
            else:
                self.selected_FRF.append(loc)

        for ix in deselected.indexes():
            loc = [ix.row(), ix.column()]
            if loc in self.selected_FRF:
                self.selected_FRF.remove(loc)
        self.update_plot()

    def find_axis(self, data, event):
        try:
            cont, ind = data.contains(event)
        except:
            cont, ind = 0, 0
        return cont, ind

    def set_cross_hair_visible(self, visible):
            need_redraw = self.horizontal_line.get_visible() != visible
            self.horizontal_line.set_visible(visible)
            self.vertical_line.set_visible(visible)
            return need_redraw

    def show_hide_cursor(self):
        if self.show_cursor.isChecked():
            self.set_cross_hair_visible(True)
            self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(False)
            self.ax.figure.canvas.draw()

    def hover(self, event):
        if self.show_cursor.isChecked():
            if not event.inaxes:
                need_redraw = self.set_cross_hair_visible(False)
                if need_redraw:
                    self.ax.figure.canvas.draw()
            else:
                self.set_cross_hair_visible(True)
                x, y = event.xdata, event.ydata
                # update the line positions
                self.horizontal_line.set_ydata(y)
                self.vertical_line.set_xdata(x)
                self.ax.figure.canvas.draw()
        
        if event.inaxes!=None:
            cont_0, ind_0 = self.find_axis(self.data_0, event)
            cont_1, ind_1 = self.find_axis(self.data_1, event)
            cont_2, ind_2 = self.find_axis(self.data_2, event)
            cont_3, ind_3 = self.find_axis(self.data_3, event)
            if (cont_0 or cont_1 or cont_2 or cont_3): 
                if cont_0:
                    self.row_3.setText("Stable")
                    ind = ind_0
                    self.x_data, self.y_data, self.damp = self.x_data_0, self.y_data_0, self.damp_data_0
                elif cont_1:
                    self.row_3.setText("Stable frequency and damping")
                    ind = ind_1
                    self.x_data, self.y_data, self.damp = self.x_data_1, self.y_data_1, self.damp_data_1
                elif cont_2:
                    self.row_3.setText("Stable frequency")
                    ind = ind_2
                    self.x_data, self.y_data, self.damp = self.x_data_2, self.y_data_2, self.damp_data_2
                elif cont_3:
                    self.row_3.setText("Unstable")
                    ind = ind_3
                    self.x_data, self.y_data, self.damp = self.x_data_3, self.y_data_3, self.damp_data_3
                ind = ind["ind"][0]
                self.row_0.setText(str(np.round(self.x_data[ind], 3)))
                self.row_1.setText(str(np.round(self.damp[ind], 5)))
                self.row_2.setText(str(int(self.y_data[ind])))
        
    def add_pole_to_data_base(self, ref_ind):
        self.get_selected_ind()
        if int(ref_ind) not in self.selected_ind:
            self.add = True
        
    def onpick(self, event):
        self.get_plot_limits()
        self.add = False
        # To avoid multiple points selection
        self.click_time = time.time()
        if self.click_time-self.click_time_old > self.time_between_clicks:
            self.click_time_old = np.copy(self.click_time)
            # to ensure that only one point is added; not sure anymore if it does something :)
            all_ind = []
            for ind in event.ind:
                all_ind.append(ind)
            ind = all_ind[0]
            if event.artist == self.data_0:
                self.x_data, self.y_data, self.damp = self.x_data_0, self.y_data_0, self.damp_data_0
                self.pole_type = "Stable"
                ref_ind = self.reference_index_0[ind]
                self.add_pole_to_data_base(ref_ind)
            elif event.artist == self.data_1:
                self.x_data, self.y_data, self.damp = self.x_data_1, self.y_data_1, self.damp_data_1
                self.pole_type = "Stable freq. and damp."
                ref_ind = self.reference_index_1[ind]
                self.add_pole_to_data_base(ref_ind)
            elif event.artist == self.data_2:
                self.x_data, self.y_data, self.damp = self.x_data_2, self.y_data_2, self.damp_data_2
                self.pole_type = "Stable freq."
                ref_ind = self.reference_index_2[ind]
                self.add_pole_to_data_base(ref_ind)
            elif event.artist == self.data_3:
                self.x_data, self.y_data, self.damp = self.x_data_3, self.y_data_3, self.damp_data_3
                self.pole_type = "Unstable"
                ref_ind = self.reference_index_3[ind]
                self.add_pole_to_data_base(ref_ind)
            if self.add == True:
                self.selected_poles = np.array([self.x_data[ind], self.damp[ind], self.y_data[ind], ref_ind]).flatten()
                self.update_table()
                self.update_plot()
                self.statusBar.showMessage("New pole was selected - freq: "+str(np.round(self.selected_poles[0], 2))+ " Hz.")
            else:
                if event.mouseevent.button == 1:
                    self.statusBar.showMessage("This pole was already selected. If you want to remove it, use right click.")
                elif event.mouseevent.button == 3:
                    self.table_model.removeRow(list(self.selected_ind).index(int(ref_ind)))
                    self.update_plot()
                    self.get_selected_ind()
                    self.statusBar.showMessage(f"The poles were deleted.")
                    self.statusBar.showMessage("Pole was deleted.")
        else:
            self.click_time_old = np.copy(self.click_time)

    def update_FRFS(self):
        self.get_plot_limits()
        self.plot_cmif()
        if len(np.array(self.selected_FRF).shape) == 2:
            FRF = np.abs(np.array(self.FRF_matrix[:, [np.array(self.selected_FRF)[:, 0]], [np.array(self.selected_FRF)[:, 1]]]))
            self.plot_FRF = self.axlog.semilogy(self.freq_cmif, np.abs(FRF).reshape(FRF.shape[0], int(FRF.shape[1]*FRF.shape[2])), alpha=self.FRF_transparency)
            self.fig.canvas.draw() # redraw graph

    def update_plot(self):
        if self.show_selected_poles.isChecked():
            try:
                self.get_data_from_table()
                self.selected_poles_plot.set_offsets(np.array([self.x_y_poles[:, 0], self.x_y_poles[:, 1]]).T)
                self.vlines.set_offsets(np.array([self.x_y_poles[:, 0], np.zeros_like(self.x_y_poles[:, 0])]).T)
            except:
                self.selected_poles_plot.set_offsets(np.array([[], []]).T)
                self.vlines.set_offsets(np.array([[], []]).T)
            self.fig.canvas.draw() # redraw graph
        self.update_FRFS()
        
    def update_table(self):
        data = self.selected_poles
        one_row = []
        for i in range(3):
            column = QtGui.QStandardItem(str(data[i]))
            column.setData(float(data[i]), QtCore.Qt.UserRole)
            one_row.append(column)
        
        plole_type_column = QtGui.QStandardItem(str(self.pole_type))
        plole_type_column.setData(str(self.pole_type), QtCore.Qt.DisplayRole)
        one_row.append(plole_type_column)

        column = QtGui.QStandardItem(str(data[-1]))
        column.setData(int(data[-1]), QtCore.Qt.UserRole)
        one_row.append(column)

        self.table_model.appendRow(one_row)
        self.tableView.setItemDelegateForColumn(0, self.float_delegate_3) # to show only 3 decimal places
        self.tableView.setItemDelegateForColumn(1, self.float_delegate_3) # to show only 3 decimal places
        self.tableView.setItemDelegateForColumn(2, self.float_delegate_0)# to show only 0 decimal places

        self.get_selected_ind()

    def state_changed_0(self):
        try: # to avoid the error on start of the program
            self.data_0.set_visible(self.option_0.isChecked())
            self.fig.canvas.draw() # redraw graph
            message_pole = "visible" if self.option_0.isChecked() else "hidden"
            self.statusBar.showMessage(f"Stabile poles are now {message_pole}.")
        except:
            pass
    
    def state_changed_1(self):
        try: # to avoid the error on start of the program
            self.data_1.set_visible(self.option_1.isChecked())
            self.fig.canvas.draw() # redraw graph
            message_pole = "visible" if self.option_1.isChecked() else "hidden"
            self.statusBar.showMessage(f"Poles with stabile frequency and damping are now {message_pole}.")
        except:
            pass
    
    def state_changed_2(self):
        try: # to avoid the error on start of the program
            self.data_2.set_visible(self.option_2.isChecked())
            self.fig.canvas.draw() # redraw graph
            message_pole = "visible" if self.option_2.isChecked() else "hidden"
            self.statusBar.showMessage(f"Poles with stabile frequency are now {message_pole}.")
        except:
            pass
    
    def state_changed_3(self):
        try: # to avoid the error on start of the program
            self.data_3.set_visible(self.option_3.isChecked())
            self.fig.canvas.draw() # redraw graph
            message_pole = "visible" if self.option_3.isChecked() else "hidden"
            self.statusBar.showMessage(f"Unstable poles are now {message_pole}.")
        except:
            pass

    def state_changed_selected_poles(self):
        try: # to avoid the error on start of the program
            self.selected_poles_plot.set_visible(self.show_selected_poles.isChecked())
            # self.vlines.set_visible(self.show_selected_poles.isChecked())
            self.fig.canvas.draw() # redraw graph
            message_pole = "visible" if self.show_selected_poles.isChecked() else "hidden"
            self.statusBar.showMessage(f"Already selected poles are now {message_pole}.")
        except:
            pass

    def delete_rows(self):
        index_list = []    
        indices = []                                                      
        for model_index in self.tableView.selectionModel().selectedRows():
            indices.append(model_index.row()) 
            index = QtCore.QPersistentModelIndex(model_index)         
            index_list.append(index)
        for index in index_list:                                      
            self.table_model.removeRow(index.row())
            
        self.update_plot()
        self.get_selected_ind()
        self.statusBar.showMessage(f"The poles were deleted.")

    def delete_all(self):                                             
        no_rows = self.table_model.rowCount()
        for i in sorted(np.arange(no_rows), reverse=True):                              
            self.table_model.removeRow(i)
        self.update_plot()
        self.get_selected_ind()
        self.statusBar.showMessage(f"The poles were deleted.")

    def export_data(self):
        self.get_data_from_table()
        if len(self.all_selected_poles) > 0:
            name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File',"output.xlsx", "Excel *.xlsx")
            print(name)
            if name:
                pd.DataFrame({"Freq. [Hz]": self.all_selected_poles[:, 0],
                "Damp [%]": self.all_selected_poles[:, 1],
                "Polinomial order": self.all_selected_poles[:, 2],
                "Pole type": self.all_selected_poles[:, 3],}).to_excel(name)
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle("File saved")
                msg.setText(f"Table data was saved to {name}.")
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec()
        else:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("Unable to export empty table.")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg.exec()

    def get_selected_ind(self):
        self.get_data_from_table()
        if self.all_selected_poles.size>0:
            self.selected_ind = np.array(list(map(int, self.all_selected_poles[:, -1])))
        else:
            self.selected_ind = []
        
        self.get_modal_data()

    def get_plot_limits(self):
        self.x_lim_ax = self.ax.get_xlim()
        self.y_lim_ax = self.ax.get_ylim()
        self.x_lim_axlog = self.axlog.get_xlim()
        self.y_lim_axlog = self.axlog.get_ylim()

    def get_data_from_table(self):
        x_y_poles=[]
        all_selected_poles = []
        for i in range(self.table_model.rowCount()):               
            x_y_poles.append([
                float(self.table_model.item(i,0).text()), float(self.table_model.item(i,2).text())])
            all_selected_poles.append([
                float(self.table_model.item(i,0).text()), 
                float(self.table_model.item(i,1).text()), 
                float(self.table_model.item(i,2).text()),
                str(self.table_model.item(i,3).text()),
                int(float(self.table_model.item(i,4).text())),
                ])
        self.x_y_poles = np.array(x_y_poles)
        self.all_selected_poles = np.asarray(all_selected_poles)

    def get_modal_data(self):
        selected_poles_id = []
        selected_mpf_id = []
        nat_freq = []
        damp_ratio = []

        for index_ in self.selected_ind:
            pole_, mpf_ = self.modal_id.pL_from_index(index_)
            selected_poles_id.append(pole_)
            selected_mpf_id.append(mpf_)

            nat_freq_, damp_ratio_, _, __ = self.modal_id.transform_poles(pole_, mpf_.T, 1, self.modal_id.freq)

            nat_freq.append(nat_freq_[0])
            damp_ratio.append(damp_ratio_[0])
            
        self.modal_id.selected_poles = np.asarray(selected_poles_id)
        self.modal_id.selected_mpf = np.asarray(selected_mpf_id).T
        self.modal_id.nat_freq = np.asarray(nat_freq)
        self.modal_id.damp_ratio = np.asarray(damp_ratio)
    