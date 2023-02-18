from pyspark.sql import SparkSession
import pyspark.pandas as ps 

# Setup
# Read in csv file, xlsx files are complicated to read in, recommend only using csv files
df = ps.read_csv('asheville_airport_2020_daily.csv')
# to spark
sdf = df.to_spark()
# display data frame
sdf.show()
# Initialize a SparkSession -- used for sql statements
spark = SparkSession.builder.getOrCreate()


# Define select, for now just print columns
# Need a lot more checks for headers
def select(columns):
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
        menu()
    else:
        print("Something went wrong")


# Create SQL statement -- also need a lot of checks
def second_option(ui):
    sdf.createOrReplaceTempView(ui)
    statement = input("Create your own SQL statement: ")
    spark.sql(statement).show()
    menu()


# Set up a SQL between option -- Does not work
def between(first, second, column):
    sdf.createGlobalTempView("temp")
    sql_input = "SELECT * FROM temp WHERE " + column + " BETWEEN " + first + " AND " + second
    print(sql_input)
    spark.sql(sql_input).show()
    menu()

# Main Menu
def menu():
    header = "\n      Main Menu       "
    line = "---------------------"
    options = "\n1) Simple SELECT statement\n2) Create your own SQL statement\n3) SQL between\n"
    options += "4) Display Table\n0) exit\n"
    menu = header + "\n" + line + options +line
    print(menu)
    
    # Pick a option
    user_input = input("Pick a option: ")
    work(user_input)

    
# Take user input
def work(ui):
    # Switch case
    if ui == "1": 
        columns = input("Provide column names you would wish to see: ")
        select(columns)
    elif ui == "2":
        # Create your own SQL statement
        temp_table = input("Provide a temporary table label to hold result: ")
        second_option(temp_table)
    elif ui == "3":
        column = input("Provide column name to show between:  ")
        first = input("Enter first value: ")
        second = input("Enter second value: ")
        between(first, second, column)
    elif ui == "4":
        sdf.show()
        menu()
    elif ui == "0":
        print("Goodbye")
        quit()
    else:
        menu()


# Display first menu
menu()