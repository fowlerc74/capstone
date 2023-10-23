from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import sys


class CalendarFilter(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QHBoxLayout()

        self.setWindowTitle("Python")

        self.setGeometry(100, 100, 600, 400)

        self.get_start = self.calendar_window()
        self.layout.addLayout(self.get_start)

    def calendar_window(self):
        overall_layout = QGridLayout()
        bottom_layout = QHBoxLayout()
        first_v_layout = QVBoxLayout()
        second_v_layout = QVBoxLayout()

        start_label = QLabel("Starting Date", self)
        end_label = QLabel("Ending Date", self)
        self.start_calendar = QCalendarWidget(self)
        self.end_calendar = QCalendarWidget(self)
        self.start_enter = QPushButton("Enter Start Date", self)
        self.end_enter = QPushButton("Enter End Date", self)
        self.done = QPushButton("Done", self)

        self.start_enter.clicked.connect(self.start_date)
        self.end_enter.clicked.connect(self.end_date)

        self.start_calendar.move(20, 40)
        self.end_calendar.move(300, 40)

        start_label.move(115, 20)
        end_label.move(395, 20)

        self.start_enter.move(100, 230)
        self.end_enter.move(382, 230)

        self.done.move(250, 300)

        self.start_calendar.setMaximumDate(QDate(2022, 9, 30))
        self.start_calendar.setMinimumDate(QDate(2006, 1, 1))
        self.end_calendar.setMaximumDate(QDate(2022, 9, 30))
        self.end_calendar.setMinimumDate(QDate(2006, 1, 1))

        first_v_layout.addWidget(start_label)
        first_v_layout.addWidget(self.start_calendar)
        first_v_layout.addWidget(self.start_enter)

        second_v_layout.addWidget(end_label)
        second_v_layout.addWidget(self.end_calendar)
        second_v_layout.addWidget(self.end_enter)

        bottom_layout.addWidget(self.done)

        overall_layout.addLayout(first_v_layout, 0, 0)
        overall_layout.addLayout(second_v_layout, 0, 2)
        overall_layout.addLayout(bottom_layout, 1, 1)

        return overall_layout

    def start_date(self):
        self.current_start = self.start_calendar.selectedDate()
        selected_start_year = str(self.current_start.year())
        selected_start_year += ".csv"
        return selected_start_year

    def end_date(self):
        self.current_end = self.end_calendar.selectedDate()
        print(self.current_end.year())
