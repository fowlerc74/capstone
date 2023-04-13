from pyspark.sql import SparkSession
from new_ML_Part import *
import pyspark.pandas as ps


spark = SparkSession.builder.getOrCreate()

def setup(file):
    file_path = 'Data/processed/' + file
    print("File Path: " + file_path)
    df = ps.read_csv(file_path)
    sdf = df.to_spark()
    return sdf



def get_linear_plot(sdf):
    pred, test = linear(sdf)
    pred.show()
    test.show()
    hold_daily = test.select('DailyPrecipitation').toPandas()
    daily_rain = list(hold_daily['DailyPrecipitation'])
    hold_prediction = pred.select('prediction').toPandas()
    daily_prediction = list(hold_prediction['prediction'])
    
    i = 0
    date_arr = []
    while (len(daily_rain) != i):
        i += 1
        date_arr.append(i)
    return daily_prediction, daily_rain, date_arr
