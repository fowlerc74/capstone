from PyQt6.QtWidgets import (
    QComboBox, QWidget, QApplication, QMainWindow, QVBoxLayout, QPushButton, QHBoxLayout, QLabel)
from new_menu import *
import sys
import os

# Window that will show the graph(s) options to pick the ML algorithms
# and also implement the hovering, filtering, and coordination features.
class graphWindow(QWidget):
    def __init__(self):
        super().__init__()
        # Window Title
        self.setWindowTitle("Graphing Window")
        # Place holder until ready to show stuff
        layout = QHBoxLayout()
        self.label = QLabel("Options")
        layout.addWidget(self.label)
        self.setLayout(layout)

# Main window, this will display the csv files to choose from and take the
# user choice and open the graph window. 
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Window Title 
        self.setWindowTitle("PyQt Test")

        # File path to the csv files
        file_path = "Data/processed"
        # Place all in directory into array
        dir_list = os.listdir(file_path)
        
        # Make combo box that selects csv file
        # TODO make option for choosing all
        self.combobox = QComboBox()
        self.combobox.addItems(dir_list)
        # Enter and Cancel buttons
        self.enter = QPushButton("Enter", self)
        self.cancel = QPushButton("Cancel", self)

        # No other window is open if None.
        self.w = None
        # Holds csv value picked from user.
        self.csv = None

        # HBox layout for the buttons on the Main window.
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.enter)
        h_layout.addWidget(self.cancel)
        # VBox layout for the overall layout of the window.
        layout = QVBoxLayout()
        layout.addWidget(self.combobox)
        layout.addLayout(h_layout)
        # container holds the entire layout and places it in the center.
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Once option is selected, the option is saved to self.csv.
        self.combobox.activated.connect(self.next_step)
        # If option is picked in combo box, open new window.
        self.enter.clicked.connect(self.open_window)
        # Closes the Main Window.
        self.cancel.clicked.connect(self.canceled)

    # When an option is selected by the user in the combobox
    # it is then passed to self.csv that will be used again in
    # the graph window class.
    def next_step(self):
        self.csv = self.combobox.currentText()

    # Opens the graph window and passes the chosen csv file into the
    # new menu setup.    
    def open_window(self):
        # Checks if a value has been selected from the combobox
        if self.csv != None:
            # If a graph window is not already open
            if self.w is None:
                # Pass csv file into new menu setup()
                setup(self.csv)
                # display graph window
                self.w = graphWindow()
                self.w.show()
            else:
                # Closes graph window if already open and sets to None again.
                self.w.close()
                self.w = None
    
    # If user hits cancel, then close main window.
    def canceled(self):
        sys.exit(app.exit())
    

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())