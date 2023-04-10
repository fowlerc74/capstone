from PyQt6.QtWidgets import (
    QComboBox, QWidget, QApplication, QMainWindow, QVBoxLayout, QPushButton, QHBoxLayout, QLabel)
from PyQt6.QtCore import Qt
import sys
import os

# Window that will show the graph(s) options to pick the ML algorithms
# and also implement the hovering, filtering, and coordination features.
class graphWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphing Window")
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
        file_path = "C:/Users/zacla/OneDrive/Desktop/Capstone/Spark/csv_files"
        # Place all in directory into array
        dir_list = os.listdir(file_path)
        
        # Make combo box that selects csv file
        # TODO make option for choosing all
        self.combobox = QComboBox()
        self.combobox.addItems(dir_list)
        self.combobox.activated.connect(self.next_step)

        # No other window is open if None
        self.w = None

        # Enter and Cancel buttons
        self.enter = QPushButton("Enter", self)
        # If option is picked in combo box, open new window
        self.enter.clicked.connect(self.open_window)
        self.cancel = QPushButton("Cancel", self)
        # Closes the Main Window
        self.cancel.clicked.connect(self.canceled)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.enter)
        h_layout.addWidget(self.cancel)

        layout = QVBoxLayout()
        layout.addWidget(self.combobox)
        layout.addLayout(h_layout)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)
    
    def next_step(self, index):
        # Pass this into original menu and start spark
        selected = self.combobox.itemText(index)
        print (selected)
    
    def open_window(self):
        # Main problem would be that there isn't a default option if nothing is chosen
        # From the combobox 
        if self.w is None:
            self.w = graphWindow()
            self.w.show()

        else:
            self.w.close()  # Close window.
            self.w = None

    def canceled(self):
        sys.exit(app.exit())
    

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
