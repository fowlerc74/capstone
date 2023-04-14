from pyspark.sql import SparkSession
from new_ML_Part import *
import pyspark.pandas as ps

# Starts spark
spark = SparkSession.builder.getOrCreate()

# Once the user chooses a csv file, it is then passed 
# into here and used to create a new data frame.
def setup(file):
    file_path = 'Data/processed/' + file
    print("File Path: " + file_path)
    df = ps.read_csv(file_path)
    sdf = df.to_spark()
    return sdf


# This calls the linear function and takes the values from there
# and places them into arrays to pass back to the graph window.
def get_linear_plot(sdf):
    pred, test = linear(sdf)
    # Prints the prediction and test values.
    pred.show()
    test.show()
    # Holds and passes values for the actual daily precipitation to an array 
    hold_daily = test.select('DailyPrecipitation').toPandas()
    daily_rain = list(hold_daily['DailyPrecipitation'])
    # Holds the prediction for the daily precipitation to an array
    hold_prediction = pred.select('prediction').toPandas()
    daily_prediction = list(hold_prediction['prediction'])
    
    # Creates an array of how many days are covered 
    i = 0
    date_arr = []
    while (len(daily_rain) != i):
        i += 1
        date_arr.append(i)
    return daily_prediction, daily_rain, date_arr
