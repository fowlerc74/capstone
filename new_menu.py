from pyspark.sql import SparkSession
from new_ML_Part import *
import pyspark.pandas as ps


spark = SparkSession.builder.getOrCreate()

def setup(file):
    file_path = 'Data/processed/' + file
    print("File Path: " + file_path)
    df = ps.read_csv(file_path)
    sdf = df.to_spark()
    sdf.printSchema()
    return sdf

def get_linear_stuff(sdf):
    linear_pred, linear_rain_res = linear(sdf)
    linear_r2 = linear_rain_res.r2
    return linear_pred, linear_rain_res, linear_r2 
