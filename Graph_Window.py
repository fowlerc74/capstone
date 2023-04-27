from pyspark.ml.feature import VectorAssembler
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np
from new_menu import *

# Window that will show the graph(s) options to pick the ML algorithms
# and also implement the hovering, filtering, and coordination features.
class graphWindow(QWidget):

    def __init__(self, csv):
        super().__init__()
        # Sets up the data frame and prints the schema to CL
        self.sdf = setup(csv)
        self.columns = get_columns(self.sdf)
        # Window Title
        self.setWindowTitle("Graphing Window")
        self.resize(900, 700)
        # Grid Layout
        self.gLayout = QGridLayout()
        # Adds a menu bar 
        menubar = QMenuBar()
        self.gLayout.addWidget(menubar, 0, 0)
        # Change csv files -> close spark and setup again
        actionFile = menubar.addMenu("File")
        actionFile.addAction("New")
        # add the machine learning options to pick from.
        actionML = menubar.addMenu("ML Options")

        linear_button = QAction("Linear Regression", actionML)
        linear_button.triggered.connect(self.linear_action)
    
        kmeans_button = QAction("K-Means", actionML)
        kmeans_button.triggered.connect(self.kmeans_window)
        
        actionML.addAction(linear_button)
        actionML.addAction(kmeans_button)
        # View
        # # Holds Filter, Reset, etc.
        self.setLayout(self.gLayout)
        
    # Displays linear regression graph and compares to the actual 
    # daily precipitation values
    def linear_action(self):
        # Displays the data frame schema for testing purposes
        self.sdf.printSchema()
        # Saves the return values from the menu.
        pred, daily_rain, date_arr = get_linear_plot(self.sdf)

        # Prints return values for testing
        print(pred)
        print(daily_rain)
        print(date_arr)

        # Makes an HBox layout
        h_layout = QHBoxLayout()
        # Makes a graph widget 
        self.graphWidget = pg.PlotWidget()
        # Add a border around the graph
        self.graphWidget.setStyleSheet("border: 5px solid blue;")
        # Sets the background white for the graph
        self.graphWidget.setBackground('w')
        # Makes a red line for the graph -- Prediction
        pen = pg.mkPen(color = (255, 0, 0))
        # Makes a blue line for the graph -- Actual rain values
        blue = pg.mkPen(color = (0, 0, 255))
        # Sets the title of the graph
        self.graphWidget.setTitle("Linear Regression")
        # Labels both x and y
        self.graphWidget.setLabel('left', 'Rain')
        self.graphWidget.setLabel('bottom', 'Days')
        # Plots the data for both the prediction and actual values
        self.graphWidget.plot(date_arr, pred, pen = pen)
        self.graphWidget.plot(date_arr, daily_rain, pen = blue)
        # Adds the graph to the HBox layout and adds it to the grid layout
        h_layout.addWidget(self.graphWidget)
        self.gLayout.addLayout(h_layout, 0, 0)
    

    def kmeans_window(self):
        self.grid_k_layout = QGridLayout()
        self.kmeans_label = QLabel("K-Means Options")
        self.kmeans_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.kmeans_label.setFont(QFont('Times', 30))
        self.grid_k_layout.addWidget(self.kmeans_label, 1, 1)

        self.column_label = QLabel("Choose a element to perform K-Means on:")
        self.grid_k_layout.addWidget(self.column_label, 3, 1)
        self.combobox1 = QComboBox()
        self.combobox1.addItems(self.columns)  
        self.grid_k_layout.addWidget(self.combobox1, 4, 1)
        self.column1 = None
        self.combobox1.activated.connect(self.set_kColumn1)
        self.combobox2 = QComboBox()
        self.combobox2.addItems(self.columns)  
        self.grid_k_layout.addWidget(self.combobox2, 5, 1)
        self.column2 = None
        self.vector = None
        self.combobox2.activated.connect(self.set_kColumn2)

        self.kNum = QLabel("Number of Clusters:")
        self.grid_k_layout.addWidget(self.kNum, 6, 1)
        self.textbox = QLineEdit()
        self.textbox.setValidator(QIntValidator())
        self.k = None
        self.grid_k_layout.addWidget(self.textbox, 7, 1)

        # Enter and Cancel buttons
        self.enter = QPushButton("Enter", self)
        self.cancel = QPushButton("Cancel", self) 
        self.grid_k_layout.addWidget(self.enter, 8, 0)
        self.grid_k_layout.addWidget(self.cancel, 8, 2)
        self.enter.clicked.connect(self.check_setup)
        self.cancel.clicked.connect(self.close)

        self.k_results = QVBoxLayout()
        self.k_results_label = QLabel('K-Means results')
        self.k_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.k_results_label.setFont(QFont('Times', 30))
        self.k_results.addWidget(self.k_results_label)

        self.gLayout.addLayout(self.grid_k_layout, 0, 1)


    # For now does 2 columns
    def set_kColumn1(self):
        self.column1 = self.combobox1.currentText()
    
    def set_kColumn2(self):
        self.column2 = self.combobox2.currentText()
    
    def set_clusters(self):
        self.k = int(self.textbox.text())
        
    def check_setup(self):
        self.set_clusters()
        self.vector_assembler()
        if (self.k != None and self.column1 != None and self.column2 != None and self.vector != None):
            output, silhouette, clusters = get_kmeans(self.sdf, self.k, self.vector)
            predict_select = output.select('prediction').toPandas()
            
            self.kmeans_output = output

            feature_select = output.select(self.column1, self.column2).toPandas()
            

            self.kmeans_action(predict_select, feature_select, clusters, silhouette)

    def vector_assembler(self):
        self.vector = VectorAssembler(inputCols = [self.column1, self.column2],
                                         outputCol = 'features')

    # The clustering of data in a new scatter plot graph and displays to the graph window.
    def kmeans_action(self, predict_arr, features_arr, clusters, silhouette_score):
        # Creates a graph widget and set the background white.
        self.graphWidget = pg.PlotWidget(self, background='w')
        self.sil_graph = pg.PlotWidget(self, background = 'w')
        # Change the prediction array into a numpy array.
        predict_list = list(predict_arr['prediction'])
        predict_arr = np.array(predict_list)

        # Change the features array into a numpy array -- Just does Daily Precipitation.
        # Make another option box to pick which feature to choose.
        features_arr = np.array(features_arr)

        silhouette_score = np.array(silhouette_score)


        self.create_table()

        # Get the amount of clusters.
        clusters = clusters.getK()
        # Creates the scatter plot.
        for i in range(clusters):
            brush = QBrush(pg.intColor(i, clusters, alpha = 150))
            pen_color = QColor(pg.intColor(i, clusters))
            self.graphWidget.scatterPlot(features_arr[predict_arr == i], symbolBrush = brush, pen = pen_color)
        
        self.sil_graph.setTitle('K-Means Silhouette Score')
        self.sil_graph.setLabel('left', 'Cost')
        self.sil_graph.setLabel('bottom', '# of Clusters')
        self.sil_graph.plot(range(2,10), silhouette_score, pen = pg.mkPen('b', width = 3))

        # Adds the graph to the HBox layout and adds it to the grid layout
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.graphWidget)
        h_layout.addWidget(self.sil_graph)
        self.gLayout.addLayout(h_layout, 0, 0)
    
    def create_table(self):
        self.tableWidget = QTableWidget()
        hold_column_names = [self.column1, self.column2, 'Prediction']
        
        self.tableWidget.setColumnCount(len(hold_column_names))
        self.tableWidget.setRowCount(len(self.column1))
        self.tableWidget.setHorizontalHeaderLabels(hold_column_names)


        self.k_results.addWidget(self.tableWidget)
        self.gLayout.addLayout(self.k_results, 1, 1)
                


