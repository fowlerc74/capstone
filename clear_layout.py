from PyQt6.QtWidgets import *
from PyQt6.QtGui import *


# Clears the graph window
def clear_graph_win(graph_win):
    print(graph_win.count())
    while graph_win.count():
        child = graph_win.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
    print("Cleared graph window")


# Clears the variable window
def clear_var_win(var_win):
    print(var_win.count())
    while var_win.count():
        child = var_win.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
    print("Cleared variable window")


# Clear the previous filter window for the next filter window
def clear_fil_win(fil_win, active):
    print(fil_win.count())

    for i in range(fil_win.count()):
        layout_item = fil_win.itemAt(i)
        if layout_item.layout() == active:
            print("HERE 4")
            deleteItemsOfLayout(layout_item.layout())
            fil_win.removeItem(layout_item)
            break

    print("Cleared filter window")


# Alternate way of deleting all items inside of a layout
def deleteItemsOfLayout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                deleteItemsOfLayout(item.layout())