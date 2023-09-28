from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from PyQt6.QtWidgets import QPushButton
import numpy as np
import pyqtgraph as pg


# Makes a Enter button when the user selects which year to use.
def linear_enter():
    enter = QPushButton("Enter")
    return enter


# Makes a cancel button that clears the filter section.
def linear_cancel():
    cancel = QPushButton("Cancel")
    return cancel


# Runs Linear Regression and returns the line graph of the predicted values
def linear_reg(sdf):
    # Drop NULL values
    sdf = sdf.na.drop()
    # Created a feature vector
    feat_assembler = vector_feature_assembler()
    # Created a new DataFrame with features vector in it
    new_df = feat_assembler.transform(sdf)
    # Selects the features column and daily precipitation columns from the new DataFrame
    feat_df = new_df.select("features", "DailyPrecipitation")
    # Split the data into a training and test data. Will be an option later
    train_data, test_data = feat_df.randomSplit([0.7, 0.3])
    # Creating an object of class Linear Regression
    # Object takes features as an input argument
    line_algo = LinearRegression(featuresCol="features", labelCol="DailyPrecipitation")
    # Fit the training data into the Linear Regression object
    line_algo = line_algo.fit(train_data)
    # Create a new DataFrame with prediction column and test data added in
    line_pred = line_algo.transform(test_data)
    # Setup the arrays to used for making the line graph
    pred_arr, date_arr, rain_arr = setup_plot(line_pred)
    # Making the line graph
    graph = setup_line_graph(pred_arr, date_arr, rain_arr)
    return graph


# Creates a vector that holds all columns except for daily precipitation because
# that is what we are trying to predict using Linear Regression
def vector_feature_assembler():
    assembler = VectorAssembler(
        inputCols=[
            "DailyAverageDryBulbTemperature",
            "DailyAverageRelativeHumidity",
            "DailyAverageSeaLevelPressure",
            "DailyAverageStationPressure",
            "DailyAverageWetBulbTemperature",
            "DailyAverageWindSpeed",
            "DailyCoolingDegreeDays",
            "DailyHeatingDegreeDays",
            "DailyMaximumDryBulbTemperature",
            "DailyMinimumDryBulbTemperature",
            "DailyPeakWindDirection",
            "DailyPeakWindSpeed",
            "DailySustainedWindSpeed",
        ],
        outputCol="features",
    )
    return assembler


# This sets up all the array that will be used in making the line graph
def setup_plot(pred):
    i = 0
    # Holds the dates that are currently being used for prediction
    date_arr = []
    # Selects Daily Precipitation and prediction from prediction DataFrame
    hold_daily = pred.select("DailyPrecipitation").collect()
    hold_pred = pred.select("prediction").collect()

    # Turn the selected columns from DataFrame into arrays
    pred_arr = np.array(hold_pred).reshape(-1)
    rain_arr = np.array(hold_daily).reshape(-1)

    # Count the amount of days for the date array
    while len(hold_daily) != i:
        i += 1
        date_arr.append(i)

    return pred_arr, date_arr, rain_arr


# Creates the line graph that will be displayed in the graph window
def setup_line_graph(pred_arr, date_arr, rain_arr):
    # Create a plot and set the background white, hoverable doesn't work
    line_plot = pg.plot(hoverable=True, background="w")
    # Set the far left axis label as Daily Precipitation
    line_plot.setLabel("left", "Daily Precipitation")
    # Set the x axis label as the days that have passed
    line_plot.setLabel("bottom", "Days")
    # Setting the plot window title
    line_plot.setWindowTitle("Linear Regression")
    # Adding a legend to easily tell which line is which
    legend = line_plot.addLegend()
    # Make the first blue line be the actual daily precipitation values
    # for the current chosen year
    actual_values = line_plot.plot(date_arr, rain_arr, pen="b")
    # This is the red line that shows the predicted values for daily
    # precipitation values
    pred_values = line_plot.plot(date_arr, pred_arr, pen="r")
    # Adds the two lines to the legend
    legend.addItem(actual_values, "Actual Rain Values")
    legend.addItem(pred_values, "Predicted Rain values")

    return line_plot
