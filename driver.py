from Main_Window import mainWindow
from PyQt6.QtWidgets import QApplication
import sys

# Starts the application and opens the main window
def main():
    app = QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()