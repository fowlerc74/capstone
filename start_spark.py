from pyspark.sql import SparkSession
import pyspark.pandas as ps

# Starts spark
spark = SparkSession.builder.getOrCreate()


# Once the user selects a csv file, it is then passed
# into here and used to create a new data frame.
def setup(file):
    file_path = "Data/processed/" + file
    print("File Path: " + file_path)
    df = ps.read_csv(file_path)
    sdf = df.to_spark()
    return sdf
