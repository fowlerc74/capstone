from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import Qt
from start_spark import *
from k_means import *
from linear import *
from pca import *
from clear_layout import *


# Window that will show the graph(s) options to pick the ML algorithms
# and where the filtering and coordination features.
class graphWindow(QWidget):
    def __init__(self, csv):
        super().__init__()
        # Sets up the data frame that will be passed into
        # each of the ML algorithms
        self.sdf = setup(csv)
        # Window Title
        self.setWindowTitle("Graphing Window")
        self.resize(1400, 700)
        # Overall layout of the window
        self.layout = QGridLayout()
        # Set the dimensions of the grid
        self.layout.setColumnStretch(0, 3)
        self.layout.setColumnStretch(1, 1)
        self.layout.setRowStretch(0, 5)
        self.layout.setRowStretch(1, 4)
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

        # Sets all of the buttons for the ML algorithms
        linear_button = QPushButton("Linear Regression")
        linear_button.clicked.connect(self.linear_win)

        kmeans_button = QPushButton("K-Means")
        kmeans_button.clicked.connect(self.kmeans_window)

        gaussian_button = QPushButton("Gaussian Mixture")
        gaussian_button.clicked.connect(self.gaussian_window)

        pca_button = QPushButton("Principal Component Analysis")
        pca_button.clicked.connect(self.pca_window)

        self.graph_active = None
        self.filter_active = None
        self.active_fil_layout = None
        # self.var_active = None

        # Adds the buttons to the ML window
        self.ml_win.addWidget(linear_button, 0, 0)
        self.ml_win.addWidget(kmeans_button, 0, 1)
        self.ml_win.addWidget(gaussian_button, 1, 0)
        self.ml_win.addWidget(pca_button, 1, 1)

        # Sets the current layout to the one just built
        self.setLayout(self.layout)

    # Calls linear to display the linear regression graph.
    def linear_win(self):
        if self.graph_active == True:
            print("Here")
            clear_graph_win(self.graph_win)
        if self.filter_active == True:
            print("Here2")
            clear_fil_win(self.filter_win, self.active_fil_layout)
        self.graph = linear_reg(self.sdf)
        self.graph_win.addWidget(self.graph, 0, 0)
        self.graph_active = True

    # Calls k_means to display the k-means graph.
    def kmeans_window(self):
        if self.graph_active == True:
            print("Here")
            clear_graph_win(self.graph_win)
        if self.filter_active == True:
            print("Here2")
            clear_fil_win(self.filter_win, self.active_fil_layout)

        self.kmeans_options = QVBoxLayout()
        k_means_label = QLabel("K-Means Options")
        k_means_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        k_means_label.setFont(QFont("Times", 30))
        self.kmeans_options.addWidget(k_means_label)

        self.column_label = QLabel("Variables:")
        k_col = columns()
        self.column1 = None
        self.column2 = None
        self.kmeans_var1 = QComboBox()
        self.kmeans_var2 = QComboBox()
        self.kmeans_var1.addItems(k_col)
        self.kmeans_var2.addItems(k_col)
        self.kmeans_options.addWidget(self.column_label)
        self.kmeans_var1.activated.connect(self.set_kColumn1)
        self.kmeans_var2.activated.connect(self.set_kColumn2)
        self.kmeans_options.addWidget(self.kmeans_var1)
        self.kmeans_options.addWidget(self.kmeans_var2)

        k_num = QLabel("Number of Clusters: ")
        self.k = None
        self.k_textbox = QLineEdit()
        self.k_textbox.setValidator(QIntValidator())
        self.kmeans_options.addWidget(k_num)
        self.kmeans_options.addWidget(self.k_textbox)

        self.enter = QPushButton("Run K-Means", self)
        self.kmeans_options.addWidget(self.enter)
        self.enter.clicked.connect(self.k_check)

        self.filter_win.addLayout(self.kmeans_options)
        self.active_fil_layout = self.kmeans_options
        self.filter_active = True

    # Set the number of clusters for K-Means
    def set_clusters(self):
        self.k = int(self.k_textbox.text())

    # Sets the first column to perform K-Means on
    def set_kColumn1(self):
        self.column1 = self.kmeans_var1.currentText()

    # Sets the second column to perform K-Means on
    def set_kColumn2(self):
        self.column2 = self.kmeans_var2.currentText()

    # Does a check to make sure the number of clusters have been selected
    # also is where the kmeans graph and silhouette graph is added to the
    # graph window.
    def k_check(self):
        self.set_clusters()
        if self.k != None:
            self.kgraph, self.sil_graph = kmeans(
                self.sdf, self.k, self.column1, self.column2
            )
            self.graph_win.addWidget(self.kgraph, 0, 0)
            self.graph_win.addWidget(self.sil_graph, 0, 1)
            self.graph_active = True

    # Calls gaussian distribution window, not implemented yet
    def gaussian_window(self):
        pass

    # Calls pca and creates a visualization.
    def pca_window(self):
        if self.graph_active == True:
            print("Here")
            clear_graph_win(self.graph_win)
        if self.filter_active == True:
            print("Here2")
            clear_fil_win(self.filter_win, self.active_fil_layout)
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
        self.filter_active = True
        self.active_fil_layout = self.pca_options

    # When the runPCA button is pressed, creates the visualization
    def run_pca(self):
        # The number of modified values to output after PCA runs
        # Create the window for PCA
        self.pca_win = QVBoxLayout()
        # Set the number of components from the textbox
        self.num_comp = int(self.pca_textbox.text())
        # If the number of components
        if self.num_comp != None and self.num_comp > 0:
            # Run PCA, get the model and data out
            self.pca_graph = pca(self.sdf, self.num_comp)

            self.pca_win.addWidget(self.pca_graph)

            # # Add label for text output
            # pca_label = QLabel("Principal Component Analysis")
            # self.pca_win.addWidget(pca_label)
            # # Print out the explained variances
            # variances = QLabel("Explained Variances:" + str(model.explainedVariance))
            # self.pca_win.addWidget(variances)
            # # Print out label for datapoints
            # first_str = "First " + str(NUM_OUTPUT_VALUES) + " Data Points:"
            # first_values_label = QLabel(first_str)
            # self.pca_win.addWidget(first_values_label)
            # # Loop through output data and print
            # for out in data:
            #     values = QLabel(str(out.output))
            #     self.pca_win.addWidget(values)

            # Add to the graph window, default to bottom left
            self.graph_win.addLayout(self.pca_win, 1, 0)
            self.graph_active = True
