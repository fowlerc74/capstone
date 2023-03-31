from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QLineEdit
from PyQt6.QtCore import Qt
import sys

# How to change current displayed text in QLabel Widget
# class Window(QWidget):
#     def __init__(self):
#         super().__init__()
#         # Window dimensions
#         self.resize(300, 250)
#         # Name of the window
#         self.setWindowTitle("PyQt Widget Test")

#         # VBox layout
#         layout = QVBoxLayout()
#         self.setLayout(layout)

#         # Widget will start out saying Old Text
#         self.label = QLabel("Old Text")
#         self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.label.adjustSize()
#         layout.addWidget(self.label)

#         # Adds button then goes to update if clicked
#         button = QPushButton("Update Text")
#         button.clicked.connect(self.update)
#         layout.addWidget(button)

#         # Adds another button to print text
#         button = QPushButton("Print Text")
#         button.clicked.connect(self.get)
#         layout.addWidget(button)

#     # Changes the text of the Widget
#     def update(self):
#         self.label.setText("New and Updated Text")

#     # Prints current Widget text   
#     def get(self):
#         print(self.label.text())

# Creating and Retrieving QLineEdit values
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(300, 250)
        self.setWindowTitle(" QLineEdit Example")

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.input = QLineEdit()
        self.input.setFixedWidth(150)
        layout.addWidget(self.input, alignment= Qt.AlignmentFlag.AlignCenter)

        button = QPushButton("Get Text")
        button.clicked.connect(self.get)
        layout.addWidget(button)

        button = QPushButton("Clear Text")
        button.clicked.connect(self.input.clear)
        layout.addWidget(button)

    def get(self):
        text = self.input.text()
        print(text)

# Creating QLabel Widget
# class Window(QWidget):
#     def __init__(self):
#         # Inherit from QWidget
#         super().__init__()
#         # Window dimensions
#         self.resize(300, 250)
#         # Name of the window
#         self.setWindowTitle("PyQt Widget Test")

#         # What the widget will say
#         label = QLabel("GUI Application with PyQt6", self)
#         # Location inside the window, default it (0,0)
#         label.move(80, 100)

app = QApplication(sys.argv)
Window = Window()
Window.show()
sys.exit(app.exec())
