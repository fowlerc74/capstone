import glob
import os
from pyspark.sql import SparkSession
import pyspark.pandas as ps

# Initialize a SparkSession -- entry point for PySpark
spark = SparkSession.builder.getOrCreate()

# Set up the data frame and return the data frame
def setup():
    print("\nHello!\n")
    # Current file path to the csv files
    # Should change if on another machine
    file_path = "C:/Users/zacla/OneDrive/Desktop/Capstone/Spark/csv_files"
    os.chdir("C:/Users/zacla/OneDrive/Desktop/Capstone/Spark/csv_files")
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
    sdf.createGlobalTempView("temp")
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

# Main menu displays options to manipulate the current data frame
def menu(sdf):
    header = "\n      Main Menu       "
    line = "---------------------"
    options = "\n1) Simple SELECT statement\n2) Create your own SQL statement\n3) SQL between\n"
    options += "4) Display Table\n5) Print Schema\n0) exit\n"
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
