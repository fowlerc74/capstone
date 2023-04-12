from pyspark.sql import SparkSession
import pyspark.pandas as ps

spark = SparkSession.builder.getOrCreate()

def setup(file):
    file_path = 'Data/processed/' + file
    print("File Path: " + file_path)
    df = ps.read_csv(file_path)
    sdf = df.to_spark()
    sdf.printSchema()
    