from PyQt6.QtWidgets import (
    QComboBox,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QApplication,
    QWidget,
    QMainWindow,
)
from PyQt6.QtCore import Qt
from k_means import *
from linear import *
from pca import *
from clear_layout import *
from start_spark import *
from calendar_filter import *
import sys
import os


class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weather app")
        self.resize(1400, 700)

        self.layout = QGridLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.layout.setColumnStretch(0, 3)
        self.layout.setColumnStretch(1, 1)
        self.layout.setRowStretch(0, 5)
        self.layout.setRowStretch(1, 4)

        self.graph_win = QGridLayout()
        self.layout.addLayout(self.graph_win, 0, 0)

        self.options_widget = QWidget()
        self.options_layout = QVBoxLayout()
        self.options_widget.setLayout(self.options_layout)
        self.options_widget.setMaximumWidth(350)  # TODO make this a var

        self.year_win = self.select_year()
        self.options_layout.addWidget(self.year_win)

        self.filter_win = QVBoxLayout()
        self.options_layout.addLayout(self.filter_win)

        self.layout.addWidget(self.options_widget, 0, 1)

        self.var_win = QHBoxLayout()
        self.layout.addLayout(self.var_win, 1, 0)

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
        self.var_active = None
        self.active_var_layout = None
        self.cal_win_active = None

        # Adds the buttons to the ML window
        self.ml_win.addWidget(linear_button, 0, 0)
        self.ml_win.addWidget(kmeans_button, 0, 1)
        self.ml_win.addWidget(gaussian_button, 1, 0)
        self.ml_win.addWidget(pca_button, 1, 1)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.select_year()

        self.csv = "2006.csv"

    # A widget that selects what year the user wants to see.
    def select_year(self):
        years = [
            "2006",
            "2007",
            "2008",
            "2009",
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022",
        ]

        file_path = "Data/processed"
        self.dir_list = os.listdir(file_path)

        self.year_widget = QWidget()
        self.year_layout = QVBoxLayout()
        self.year_widget.setLayout(self.year_layout)

        self.year_label = QLabel("Select Year:")
        self.year_layout.addWidget(self.year_label)

        self.year_select = QComboBox()
        self.year_select.addItems(self.dir_list)
        # self.year_select.activated.connect(self.set_csv)
        self.year_select.currentTextChanged.connect(self.set_csv)
        self.year_layout.addWidget(self.year_select)
        self.filter_active = True

        self.min_year_label = QLabel("Starting Year:")
        self.year_layout.addWidget(self.min_year_label)
        self.min_year_select = QComboBox()
        self.min_year_select.addItems(years)
        self.min_year_select.activated.connect(self.set_min_year)
        self.year_layout.addWidget(self.min_year_select)

        self.max_year_label = QLabel("Ending Year:")
        self.year_layout.addWidget(self.max_year_label)
        self.max_year_select = QComboBox()
        self.max_year_select.addItems(years)
        self.max_year_select.activated.connect(self.set_max_year)
        self.year_layout.addWidget(self.max_year_select)

        self.cal_win = None
        self.calendar_button = QPushButton("Open Calendar")
        self.calendar_button.clicked.connect(self.open_calendar)
        self.year_layout.addWidget(self.calendar_button)

        self.year_widget.setMaximumHeight(200)
        self.year_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        return self.year_widget

    # This will open the calender filter window.
    def open_calendar(self):
        if self.cal_win == None:
            self.cal_win = CalendarFilter()
            self.cal_win.show()
            self.cal_win_active = True
            self.cal_win.done.clicked.connect(self.set_csv)
        else:
            self.cal_win.close()
            self.cal_win_active = False
            self.cal_win = None

    # When a starting year is selected, the variable min_year is updated
    def set_min_year(self):
        self.min_year = self.min_year_select.currentText()

    # When an ending year is selected, the variable max_year is updated
    def set_max_year(self):
        self.max_year = self.max_year_select.currentText()

    # When an option is selected by the user in the combobox
    # it is then passed to self.csv that will be used again in
    # the graph window class.
    def set_csv(self, csv):
        if self.cal_win != None:
            self.csv = self.cal_win.start_date()
            self.cal_win.close()
            self.cal_win == None
        if self.cal_win_active != True:
            self.csv = csv

    # Has the user select the year and calls linear to
    # perform linear regression on daily precipitation with the chosen year
    def linear_win(self):
        self.canceled()
        self.linear_fil = QVBoxLayout()
        self.test_train_layout = QGridLayout()
        self.linear_param = None
        linear_select = columns()
        self.linear_option = QComboBox()
        self.linear_option.addItems(linear_select)
        self.linear_enter = linear_enter()
        self.linear_cancel = linear_cancel()
        self.train_box = QLineEdit()
        self.test_box = QLineEdit()
        self.train_test = train_test_title()
        self.train_title = train_box_label()
        self.test_title = test_box_label()
        self.line_warn = warn_label()
        self.linear_fil.addWidget(self.train_test)
        self.test_train_layout.addWidget(self.train_title, 0, 0)
        self.test_train_layout.addWidget(self.test_title, 0, 1)
        self.test_train_layout.addWidget(self.train_box, 1, 0)
        self.test_train_layout.addWidget(self.test_box, 1, 1)
        self.linear_fil.addLayout(self.test_train_layout)
        self.linear_fil.addWidget(self.line_warn)
        self.linear_fil.addWidget(self.linear_option)
        self.linear_fil.addWidget(self.linear_enter)
        self.linear_fil.addWidget(self.linear_cancel)
        self.filter_win.addLayout(self.linear_fil)
        self.active_fil_layout = self.linear_fil
        self.filter_active = True
        self.linear_enter.clicked.connect(self.linear_run)
        self.linear_cancel.clicked.connect(self.canceled)

    # Takes the year and runs spark and displays the linear regression graph.
    def linear_run(self):
        if self.var_active == True:
            clear_var_win(self.var_win, self.active_var_layout)
        if self.csv != None:
            self.linear_param = self.linear_option.currentText()
            sdf = setup(self.csv)
            (
                self.linear_graph,
                self.coe_label,
                self.inter_label,
                self.r2_label,
            ) = linear_reg(
                sdf, self.linear_param, self.test_box.text(), self.train_box.text()
            )
            self.linear_var_win = QVBoxLayout()
            self.graph_win.addWidget(self.linear_graph, 0, 0)
            self.linear_var_win.addWidget(self.coe_label)
            self.linear_var_win.addWidget(self.inter_label)
            self.linear_var_win.addWidget(self.r2_label)
            self.var_win.addLayout(self.linear_var_win)
            self.graph_active = True
            self.active_var_layout = self.linear_var_win
            self.var_active = True

    #  Calls k_means to display the k-means graph.
    def kmeans_window(self):
        if self.graph_active == True:
            clear_graph_win(self.graph_win)
        if self.filter_active == True:
            clear_fil_win(self.filter_win, self.active_fil_layout)
        if self.var_active == True:
            clear_var_win(self.var_win, self.active_var_layout)

        self.kmeans_options = QVBoxLayout()
        k_means_label = QLabel("K-Means Options")
        k_means_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        k_means_label.setFont(QFont("Times", 30))
        self.kmeans_options.addWidget(k_means_label)

        self.column_label = QLabel("Variables:")
        k_col = columns()
        self.column1 = None
        self.column2 = "None"
        self.column3 = "None"
        self.kmeans_var1 = QComboBox()
        self.kmeans_var2 = QComboBox()
        self.kmeans_var3 = QComboBox()
        self.kmeans_var1.addItems(k_col)
        self.kmeans_var2.addItems(k_col)
        self.kmeans_var3.addItems(k_col)
        self.kmeans_options.addWidget(self.column_label)
        self.kmeans_var1.activated.connect(self.set_kColumn1)
        self.kmeans_var2.activated.connect(self.set_kColumn2)
        self.kmeans_var3.activated.connect(self.set_kColumn3)
        self.kmeans_options.addWidget(self.kmeans_var1)
        self.kmeans_options.addWidget(self.kmeans_var2)
        self.kmeans_options.addWidget(self.kmeans_var3)

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
        print(self.column2)

    # Sets the third column to perform K-Means on
    def set_kColumn3(self):
        self.column3 = self.kmeans_var3.currentText()

    # Does a check to make sure the number of clusters have been selected
    # also is where the kmeans graph and silhouette graph is added to the
    # graph window.
    def k_check(self):
        if self.graph_active == True:
            clear_graph_win(self.graph_win)
        if self.var_active == True:
            clear_var_win(self.var_win, self.active_var_layout)
        self.set_clusters()
        if self.k != None and self.csv != None:
            self.sdf = setup(self.csv)
            self.k_output, self.sil_graph, self.centers = kmeans(
                self.sdf, self.k, self.column1, self.column2, self.column3
            )
            self.select_layout = QVBoxLayout()
            self.k_layout = QHBoxLayout()
            self.x_label = QLabel("Select the X value for the graph:")
            self.select_x = user_select_x(self.column1, self.column2, self.column3)
            self.select_layout.addWidget(self.x_label)
            self.select_layout.addWidget(self.select_x)
            self.x_kmeans = self.column1
            self.select_x.activated.connect(self.select_x_kmeans)
            self.y_label = QLabel("Select the Y value for the graph:")
            self.select_y = user_select_y(self.column1, self.column2, self.column3)
            self.select_layout.addWidget(self.y_label)
            self.select_layout.addWidget(self.select_y)
            self.y_kmeans = self.column2
            self.select_y.activated.connect(self.select_y_kmeans)
            self.enter_selection = QPushButton("Display Graph")
            self.enter_selection.clicked.connect(self.display_k_graph)
            self.select_layout.addWidget(self.enter_selection)
            self.k_layout.addLayout(self.select_layout)
            self.k_layout.addLayout(self.centers)
            self.var_win.addLayout(self.k_layout)
            self.active_var_layout = self.k_layout
            self.graph_active = True
            self.var_active = True

            self.hover_option_layout = QVBoxLayout()
            self.hover_option_label = QLabel("Select the data to show on hover:")
            self.hover_option = QComboBox()
            self.hover_option.addItems(columns()[1:])
            self.hover = Hover(columns()[1])
            self.hover_option.activated.connect(self.set_hover_var)
            self.hover_option_layout.addWidget(self.hover_option_label)
            self.hover_option_layout.addWidget(self.hover_option)
            self.k_layout.addLayout(self.hover_option_layout)

    def set_hover_var(self):
        self.hover.set_hover_var(self.hover_option.currentText())

    # When the user selects which variable to use for the X axis, it
    # is saved here to be used for creating the graph.
    def select_x_kmeans(self):
        self.x_kmeans = self.select_x.currentText()

    # When the user selects which variable to use for the Y axis, it
    # is saved here to be used for creating the graph.
    def select_y_kmeans(self):
        self.y_kmeans = self.select_y.currentText()

    # Takes all results from running K-Means and displays the results through a scatter plot
    # also makes use of the silhouette score and makes a line graph.
    def display_k_graph(self):
        sdf = setup(self.csv)
        self.k_graph = setup_graph_k(
            self.k_output, self.k, self.x_kmeans, self.y_kmeans, self.hover, sdf
        )
        self.graph_win.addWidget(self.k_graph, 0, 0)
        self.graph_win.addWidget(self.sil_graph, 0, 1)
        self.graph_active = True

    def gaussian_window(self):
        pass

    # Calls pca and creates a visualization.
    def pca_window(self):
        if self.graph_active == True:
            clear_graph_win(self.graph_win)
        if self.filter_active == True:
            clear_fil_win(self.filter_win, self.active_fil_layout)
        if self.var_active == True:
            clear_var_win(self.var_win, self.active_var_layout)
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
        NUM_OUTPUT_VALUES = 5  # temporary until PCA visualization gets built
        # Create the window for PCA
        self.pca_win = QVBoxLayout()
        # Set the number of components from the textbox
        self.num_comp = int(self.pca_textbox.text())

        if self.csv != None:
            sdf = setup(self.csv)

        # If the number of components
        if self.num_comp != None and self.num_comp > 0:
            # Run PCA, get the model and data out
            model, data = pca(sdf, self.num_comp, NUM_OUTPUT_VALUES)
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
            self.graph_active = True

    # If a cancel button is present in the filter window,
    # it should clear the window.
    def canceled(self):
        if self.graph_active == True:
            clear_graph_win(self.graph_win)
        if self.filter_active == True:
            clear_fil_win(self.filter_win, self.active_fil_layout)
        if self.var_active == True:
            clear_var_win(self.var_win, self.active_var_layout)


app = QApplication(sys.argv)
window = mainWindow()
window.show()
sys.exit(app.exec())
