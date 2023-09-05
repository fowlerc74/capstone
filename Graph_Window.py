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
        self.resize(1400, 700)
        # Grid Layout
        self.layout = QGridLayout()
        # Set the dimensions of the grid
        self.layout.setColumnStretch(0, 3)
        self.layout.setColumnStretch(1, 1)
        self.layout.setRowStretch(0, 5)
        self.layout.setRowStretch(1,4)
        # Adds graph window
        self.graph_win = QGridLayout()
        self.layout.addLayout(self.graph_win, 0, 0)
        # Adds filter window
        self.filter_win = QVBoxLayout()
        self.layout.addLayout(self.filter_win, 0, 1)
        # Adds variable window
        # self.var_win = QHBoxLayout()
        # self.layout.addLayout(self.var_win, 1, 0)
        # Adds ML option corner
        self.ml_win = QGridLayout()
        self.layout.addLayout(self.ml_win, 1, 1)
        # Change csv files -> close spark and setup again
        # actionFile = mloption.addMenu("File")
        # actionFile.addAction("New")

        # Set a title in the graph window as a placeholder
        # title = QLabel("Graph")
        # self.graph_win.addWidget(title)
        # Set a title in the filters window as a placeholder
        # title = QLabel("Filters")
        # self.filter_win.addWidget(title)
        # Set a title in the variables window as a placeholder
        # title = QLabel("Variables")
        # self.var_win.addWidget(title)

        # Sets all of the buttons for the linear regression algorithms
        linear_button = QPushButton("Linear Regression")
        linear_button.clicked.connect(self.linear_action)
    
        kmeans_button = QPushButton("K-Means")
        kmeans_button.clicked.connect(self.kmeans_window)

        gaussian_button = QPushButton("Gaussian Mixture")
        gaussian_button.clicked.connect(self.gaussian_window)

        pca_button = QPushButton("Principal Component Analysis")
        pca_button.clicked.connect(self.pca_window)
        
        # Adds the buttons to the ML window
        self.ml_win.addWidget(linear_button, 0, 0)
        self.ml_win.addWidget(kmeans_button, 0, 1)
        self.ml_win.addWidget(gaussian_button, 1, 0)
        self.ml_win.addWidget(pca_button, 1, 1) 

        # Sets the current layout to the one just built
        self.setLayout(self.layout)

    """
    This code would clean up the init method significantly, but it won't let me call other
    methods from the init method
    def init_filter_window(self):
        title = QLabel("Filters")
        self.filterWin.addWidget(title)
    def init_var_window(self):
        title = QLabel("Variables")
        self.filterWin.addWidget(title)
    def init_ml_window(self):
        linear_button = QAction("Linear Regression", self.mlWin)
        linear_button.triggered.connect(self.linear_action)
    
        kmeans_button = QAction("K-Means", self.mlWin)
        kmeans_button.triggered.connect(self.kmeans_window)
        gaussian_button = QAction("Gaussian Mixture", self.mlWin)
        gaussian_button.triggered.connect(self.gaussian_window)
        pca_button = QAction("Principal Component Analysis", self.mlWin)
        pca_button.triggered.connect(self.pca_window)
        
        self.mlWin.addAction(linear_button)
        self.mlWin.addAction(kmeans_button)
        self.mlWin.addAction(gaussian_button)
        self.mlWin.addAction(pca_button) 
    """
        
    # Displays linear regression graph and compares to the actual 
    # daily precipitation values
    def linear_action(self):
        # Saves the return values from the menu.
        pred, daily_rain, date_arr = get_linear_plot(self.sdf)

        # Makes an HBox layout
        self.h_layout = QHBoxLayout()
        # Makes a graph widget 
        self.graph_widget = pg.PlotWidget()
        # Add a border around the graph
        self.graph_widget.setStyleSheet("border: 5px solid blue;")
        # Sets the background white for the graph
        self.graph_widget.setBackground('w')
        # Makes a red line for the graph -- Prediction
        pen = pg.mkPen(color = (255, 0, 0))
        # Makes a blue line for the graph -- Actual rain values
        blue = pg.mkPen(color = (0, 0, 255))
        # Sets the title of the graph
        self.graph_widget.setTitle("Linear Regression")
        # Labels both x and y
        self.graph_widget.setLabel('left', 'Rain')
        self.graph_widget.setLabel('bottom', 'Days')
        # Plots the data for both the prediction and actual values
        self.graph_widget.plot(date_arr, pred, pen = pen)
        self.graph_widget.plot(date_arr, daily_rain, pen = blue)
        # Adds the graph to the HBox layout and adds it to the grid layout
        self.h_layout.addWidget(self.graph_widget)
        self.graph_win.addLayout(self.h_layout, 0, 0)

    # Displays options that the user will use to start the initial setup for 
    # K-Means to run properly.
    def kmeans_window(self):
        self.grid_k_layout = QGridLayout()
        self.k_results = QVBoxLayout()
    
        self.k_means_label = QLabel("K-Means Options")
        self.k_means_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.k_means_label.setFont(QFont('Times', 30))
        self.grid_k_layout.addWidget(self.k_means_label, 1, 1)

        self.column_label = QLabel("Variables:")
        self.column1 = None
        self.column2 = None
        self.vector = None
        self.combobox1 = QComboBox()
        self.combobox2 = QComboBox()
        self.combobox1.addItems(self.columns)
        self.combobox2.addItems(self.columns)
        self.grid_k_layout.addWidget(self.column_label, 3, 1)
        self.grid_k_layout.addWidget(self.combobox1, 4, 1)
        self.grid_k_layout.addWidget(self.combobox2, 5, 1)
        self.combobox1.activated.connect(self.set_kColumn1)
        self.combobox2.activated.connect(self.set_kColumn2)

        self.k_num = QLabel("Number of Clusters:")
        self.k = None
        self.textbox = QLineEdit()
        self.textbox.setValidator(QIntValidator())
        self.grid_k_layout.addWidget(self.k_num, 6, 1)
        self.grid_k_layout.addWidget(self.textbox, 7, 1)

        # Enter and Cancel buttons
        self.enter = QPushButton("Run K-Means", self)
        self.grid_k_layout.addWidget(self.enter, 8, 0)
        self.enter.clicked.connect(self.check_setup)
        
        self.filter_win.addLayout(self.grid_k_layout)


    # Sets the first column to perform K-Means on
    def set_kColumn1(self):
        self.column1 = self.combobox1.currentText()
    
    # Sets the second column to perform K-Means on
    def set_kColumn2(self):
        self.column2 = self.combobox2.currentText()
    
    # Sets the number of clusters.
    def set_clusters(self):
        self.k = int(self.textbox.text())
    
    # Checks that all values are valid before running K-Means and returns the values
    # to be used for making both the cluster and silhouette graph  
    def check_setup(self):
        self.set_clusters()
        self.vector_assembler()
        if (self.k != None and self.column1 != None and self.column2 != None and self.vector != None):
            output, silhouette, clusters = get_kmeans(self.sdf, self.k, self.vector)
            predict_select = output.select('prediction').toPandas()

            feature_select = output.select(self.column1, self.column2).toPandas()
            

            self.kmeans_action(predict_select, feature_select, clusters, silhouette)
    
    # Makes a vector from what the user selected in the K-Means option window
    def vector_assembler(self):
        self.vector = VectorAssembler(inputCols = [self.column1, self.column2],
                                         outputCol = 'features')

    # The clustering of data in a new scatter plot graph and displays to the graph window.
    def kmeans_action(self, predict_arr, features_arr, clusters, silhouette_score):
        # Creates a graph widget and set the background white.
        self.graph_widget = pg.PlotWidget(self, background='w')
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
            self.graph_widget.scatterPlot(features_arr[predict_arr == i], symbolBrush = brush, pen = pen_color)

        self.graph_widget.setTitle('K-Means Clustering')
        self.graph_widget.setLabel('left', self.column1)
        self.graph_widget.setLabel('bottom', self.column2)

        self.sil_graph.setTitle('K-Means Silhouette Score')
        self.sil_graph.setLabel('left', 'Cost')
        self.sil_graph.setLabel('bottom', '# of Clusters')
        self.sil_graph.plot(range(2,10), silhouette_score, pen = pg.mkPen('b', width = 3))

        self.graph_win.addWidget(self.graph_widget, 0, 0)
        self.graph_win.addWidget(self.sil_graph, 0, 1)

        # Adds the graph to the HBox layout and adds it to the grid layout
        h_layout = QHBoxLayout()
        # h_layout.addWidget(self.graph_widget)
        # h_layout.addWidget(self.sil_graph)
        self.graph_win.addLayout(h_layout, 0, 0)
        # self.layout.addLayout(h_layout, 0, 0)
    
    # This creates a table to view all points in the K-Means cluster graph
    # and view which cluster the point belongs to.
    def create_table(self):
        self.table_widget = QTableWidget()
        hold_column_names = [self.column1, self.column2, 'Prediction']
        
        self.table_widget.setColumnCount(len(hold_column_names))
        self.table_widget.setRowCount(len(self.column1))
        self.table_widget.setHorizontalHeaderLabels(hold_column_names)


        self.k_results.addWidget(self.table_widget)
        self.layout.addLayout(self.k_results, 1, 0)
    

    # Creates the gaussian distribution window, not implemented yet
    def gaussian_window(self):
        pass

    # Creates the window for the PCA visualization and all filters
    def pca_window(self):
        # Create the options box
        self.pca_options = QVBoxLayout()
        # Create the label for the number of components
        num_comp_label = QLabel("Number of Components:")
        self.pca_options.addWidget(num_comp_label)
        # Create the textbox for the number of components, and make it only take ints
        self.pca_textbox = QLineEdit()
        self.pca_textbox.setValidator(QIntValidator())
        self.pca_options.addWidget(self.pca_textbox)
        # Create the button to run PCA
        run_PCA = QPushButton("Run PCA", self)
        run_PCA.clicked.connect(self.run_pca)
        self.pca_options.addWidget(run_PCA)
        # Add the PCA options to the filter window
        self.filter_win.addLayout(self.pca_options)
        # Doesn't run PCA until runPCA button pressed

    # When the runPCA button is pressed, creates the visualization
    def run_pca(self):
        # The number of modified values to output after PCA runs
        NUM_OUTPUT_VALUES = 5 # temporary until PCA visualization gets built
        # Create the window for PCA
        self.pca_win = QVBoxLayout()
        # Set the number of components from the textbox
        self.num_comp = int(self.pca_textbox.text())
        # If the number of components
        if self.num_comp != None and self.num_comp > 0:
            # Run PCA, get the model and data out
            model, data = pca(self.sdf, self.num_comp, NUM_OUTPUT_VALUES)
            # Add label for text output
            pca_label = QLabel("Principal Component Analysis")
            self.pca_win.addWidget(pca_label)
            # Print out the explained variances
            variances = QLabel("Explained Variances:" + str(model.explainedVariance))
            self.pca_win.addWidget(variances)
            # Print out label for datapoints
            first_str = "First " + str(NUM_OUTPUT_VALUES) + " Data Points:"
            first_values_label = QLabel(first_str)
            self.pca_win.addWidget(first_values_label)
            # Loop through output data and print
            for out in data: 
                values = QLabel(str(out.output))
                self.pca_win.addWidget(values)

            # Add to the graph window, default to bottom left
            self.graph_win.addLayout(self.pca_win, 1, 0)
