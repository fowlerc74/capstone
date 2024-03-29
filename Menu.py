import glob
import os
from pyspark.sql import SparkSession
import pyspark.pandas as ps
from pyspark.sql.functions import col
from ML_Part import *

# Initialize a SparkSession -- entry point for PySpark
spark = SparkSession.builder.getOrCreate()

# Set up the data frame and return the data frame
def setup():
    print("\nHello!\n")
    # Current file path to the csv files
    # Should change if on another machine
    # file_path = "C:/Users/zacla/OneDrive/Desktop/Capstone/Spark/csv_files"
    file_path = "/Users/cade/School/2023_Spring/Capstone/capstone/Data/processed"
    os.chdir(file_path)
    # Get only csv files
    my_files = glob.glob('*.csv')
    count = 1
    # Display current files in csv_files folder
    for i in my_files:
        print(str(count) + ") " + i)
        count += 1
    # Have a option to combine all csv files together
    print(str(count) + ") Combine all")

    # Take the user input 
    choose_csv = input("Please choose an option from the list to begin: ")
    # If the user chooses the combine all option
    if int(choose_csv) > len(my_files):
        # Place all paths into an array
        path = ['csv_files/asheville_airport_2020_daily.csv', 
            'csv_files/asheville_airport_2021_daily.csv', 
            'csv_files/asheville_airport_2022_daily.csv']
        sdf = spark.read.options(header = True).csv(path)
    else:
        # take the chosen csv file and read into data frame
        choice = int(choose_csv) - 1
        hold = my_files[int(choice)]
        full_Path = file_path + "/" + hold
        df = ps.read_csv(full_Path)
        sdf = df.to_spark()

    return sdf

# Define select, for now just print columns
def select(columns, sdf, spark):
    check = False
    if ', ' in columns:
        check = True
    elif ' ' in columns:
        columns = columns.replace(' ', ', ')
        check = True
    else:
        check = True
    if check == True:
        sdf.createOrReplaceTempView("tempTable")
        sql_input = "SELECT " + columns + " from tempTable"
        spark.sql(sql_input).show()
        menu(sdf)
    else:
        print("Something went wrong")

# Create SQL statement
def second_option(ui, sdf, spark):
    sdf.createOrReplaceTempView(ui)
    statement = input("Create your own SQL statement: ")
    spark.sql(statement).show()
    menu(sdf)

# Set up a SQL between option -- Does not work
def between(first, second, column, sdf, spark):
    sdf.createOrReplaceTempView("temp")
    sql_input = "SELECT * FROM temp WHERE " + column + " BETWEEN " + first + " AND " + second
    print(sql_input)
    spark.sql(sql_input).show()
    menu(sdf)

# Displays the current data frame
def display(sdf):
    sdf.show()
    menu(sdf)

# Displays the column headers and there data types
def display_schema(sdf):
    sdf.printSchema()
    menu(sdf)

# Calls machine menu in ML_Part
def machine(sdf):
    machine_menu(sdf)
    menu(sdf)

# Shows the min and max for both precipitation and snowfall
def min_max(sdf):
    options = "1) Max on Daily Precipitation\n2) Min on Daily Precipitation\n"
    options += "3) Max on Daily Snowfall\n4) Min on Daily Snowfall\n"
    print(options)
    pick = input("Choose an option: ")
    sdf.createOrReplaceTempView("temp")
    if pick == "1":
        sql_input = "SELECT MAX(DailyPrecipitation) FROM temp"
        spark.sql(sql_input).show()
        menu(sdf)
    elif pick == "2":
        sql_input = "SELECT MIN(DailyPrecipitation) FROM temp"
        spark.sql(sql_input).show()
        menu(sdf)
    elif pick == "3":
        sql_input = "SELECT MAX(DailySnowfall) FROM temp"
        spark.sql(sql_input).show()
        menu(sdf)
    elif pick == "4":
        sql_input = "SELECT MIN(DailySnowfall) FROM temp"
        spark.sql(sql_input).show()
        menu(sdf)
    else:
        print("Wrong input, try again")

# Main menu displays options to manipulate the current data frame
def menu(sdf):
    header = "\n      Main Menu       "
    line = "---------------------"
    options = "\n1) Simple SELECT statement\n2) Create your own SQL statement\n3) SQL between\n"
    options += "4) Display Table\n5) Print Schema\n6) Min and Max\n"
    options += "7) Machine Learning\n0) exit\n"
    menu = header + "\n" + line + options +line
    print(menu)
    
    # Pick a option
    user_input = input("Pick a option: ")
    work(user_input, sdf)

# Take user input
def work(ui, sdf):
    # Switch case
    if ui == "1": 
        columns = input("Provide column names you would wish to see: ")
        select(columns, sdf, spark)
    elif ui == "2":
        # Create your own SQL statement
        temp_table = input("Provide a temporary table label to hold result: ")
        second_option(temp_table, sdf, spark)
    elif ui == "3":
        column = input("Provide column name to show between:  ")
        first = input("Enter first value: ")
        second = input("Enter second value: ")
        between(first, second, column, sdf, spark)
    elif ui == "4":
        display(sdf)
    elif ui == "5":
        display_schema(sdf)
    elif ui == "6":
        min_max(sdf)
    elif ui == "7":
        machine(sdf)
    elif ui == "0":
        print("Goodbye")
        quit()
    else:
        print("Not an option, PLease try again.")
        menu(sdf)

# Call setup for start passes user selected options to work
def main():
    sdf = setup()
    result = menu(sdf)
    work(result, sdf)

if __name__ == "__main__":
    main()
