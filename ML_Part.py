from pyspark.ml.feature import VectorAssembler,PCA
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col
import matplotlib.pyplot as plt


####################### Linear Regression #########################
# Learn more about what goes into linear regression model
def linear(sdf, ui):
    # Drops NULL values
    sdf = sdf.na.drop()
    # Creates vectors from dataset 
    assembler = VectorAssembler(inputCols=['DailyAverageDryBulbTemperature',
                                        'DailyAverageRelativeHumidity',
                                        'DailyAverageSeaLevelPressure',
                                        'DailyAverageStationPressure',
                                        'DailyAverageWetBulbTemperature',
                                        'DailyAverageWindSpeed',
                                        'DailyCoolingDegreeDays',
                                        'DailyHeatingDegreeDays',
                                        'DailyMaximumDryBulbTemperature',
                                        'DailyMinimumDryBulbTemperature',
                                        'DailyPeakWindDirection',
                                        'DailyPeakWindSpeed',
                                        'DailySustainedWindSpeed'], outputCol= 'Features')
    output = assembler.transform(sdf)

    # final data consists of features and label which is daily precipitation
    final_data = output.select('Features', 'DailyPrecipitation')
    # Splitting the data into train and test
    train_data, test_data = final_data.randomSplit([0.7, 0.3])
    # Display the train data based on the final_split split
    # train_data.show()

    # Creating an object of class LinearRegression
    # Object takes Features and label as input arguments
    rain_lr = LinearRegression(featuresCol='Features', labelCol='DailyPrecipitation')
    # Pass train_data to train model
    trained_rain_model = rain_lr.fit(train_data)
    # Evaluating model trained for R-squared error
    rain_results = trained_rain_model.evaluate(train_data)
    # Testing Model on unlabeled data
    # Create unlabeled data from test_data
    # Testing model on unlabeled data
    unlabeled_data = test_data.select('Features')

    # Predictions on the results from test data
    predictions = trained_rain_model.transform(unlabeled_data)
    # options that the user selected
    if ui == "1":
        sdf.show()
        machine_menu(sdf)
    elif ui == "2":
        predictions.show(truncate=False)
        machine_menu(sdf)
    elif ui == "3":
        train_data.show(truncate=False)
        predictions.show(truncate=False)
        machine_menu(sdf)
    elif ui == "4":
        linear_plot(test_data, predictions)
        machine_menu(sdf)
    elif ui == "5":
        # R2 value shows accuracy of model
        print('\nR-squared Error: ', rain_results.r2)
        machine_menu(sdf)
    elif ui == "6":
        sdf.show()
        train_data.show(truncate=False)
        predictions.show(truncate=False)
        print('\nR-squared Error: ', rain_results.r2)
        linear_plot(test_data, predictions)
        machine_menu(sdf)
    elif ui == "0":
        machine_menu(sdf)
    else:
        print("Try again.")
        machine_menu(sdf)

# Menu to display options for Linear Regression 
def linear_menu(sdf):
    display = "\n1) Display Full Dataset\n2) Display Prediction table\n3) Display Prediction table + Subset of Dataset for Test\n"
    display += "4) Display Scatter Plot\n5) Display R-Squared Error value\n6) Display All\n0) Exit\n"
    print(display)
    ui = input("Pick a option: ")
    linear(sdf, ui)

# Displays the scatter plot of the Linear Regression model
def linear_plot(lr_model, pred):
    hold_daily = lr_model.select('DailyPrecipitation').toPandas()
    daily_rain = list(hold_daily['DailyPrecipitation'])
    hold_prediction = pred.select('prediction').toPandas()
    daily_prediction = list(hold_prediction['prediction'])

    i = 0
    date_arr = []
    while (len(daily_rain) != i):
        i += 1
        date_arr.append(i)

    plt.scatter(date_arr, daily_rain)
    plt.scatter(date_arr, daily_prediction)
    plt.show()


#################### K-Means ######################
# Learn more on what goes into the k-means algorithm
def kmeans(sdf):
    # Drops NULL values
    sdf = sdf.na.drop()
    # Creates vectors from dataset 
    assembler = VectorAssembler(inputCols=['DailyAverageDryBulbTemperature',
                                            'DailyAverageRelativeHumidity',
                                            'DailyAverageSeaLevelPressure',
                                            'DailyAverageStationPressure',
                                            'DailyAverageWetBulbTemperature',
                                            'DailyAverageWindSpeed',
                                            'DailyPrecipitation',
                                            'DailyCoolingDegreeDays',
                                            'DailyHeatingDegreeDays',
                                            'DailyMaximumDryBulbTemperature',
                                            'DailyMinimumDryBulbTemperature',
                                            'DailyPeakWindDirection',
                                            'DailyPeakWindSpeed',
                                            'DailySustainedWindSpeed'], outputCol= 'features')
    new_df = assembler.transform(sdf)
    # Trains a k-means model
    kmeans = KMeans(k = 2)
    model = kmeans.fit(new_df.select('features'))
    # Makes the prediction
    output = model.transform(new_df)
    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()
    # Silhouette score measures how similar a data point is within-cluster
    # Compared to other clusters
    silhouette = evaluator.evaluate(output)
    print("\nSilhouette with squared euclidean distance = " + str(silhouette))
    # Displays cluster centers
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)
    # Display which features are in which cluster
    output.select(col('features'), col('prediction')).show()
    # Return to ML menu
    machine_menu(sdf)

####### Gaussian Mixture #######
# Clustering Algorithm
def gaussian(sdf):
    # Drops NULL values
    sdf = sdf.na.drop() # checking if we can use the same csv without dropping NULL values
    num_k = 4
    assembler = VectorAssembler(inputCols=['DailyAverageDryBulbTemperature',
                                        'DailyAverageRelativeHumidity',
                                        'DailyAverageSeaLevelPressure',
                                        'DailyAverageStationPressure',
                                        'DailyAverageWetBulbTemperature',
                                        'DailyAverageWindSpeed',
                                        'DailyCoolingDegreeDays',
                                        'DailyDepartureFromNormalAverageTemperature',
                                        'DailyHeatingDegreeDays',
                                        'DailyMaximumDryBulbTemperature',
                                        'DailyMinimumDryBulbTemperature',
                                        'DailyPeakWindDirection',
                                        'DailyPeakWindSpeed',
                                        'DailySustainedWindSpeed',
                                        'DailyWeather'], outputCol= 'features')
    df = assembler.transform(sdf)
    #parsedData = df.map(lambda line: array([float(x) for x in line.strip().split(' ')]))

    # Build the model (cluster the data)
    gm = GaussianMixture(k=num_k, tol=.001)
    gm.setMaxIter(30)
    model = gm.fit(df)

    
    # output parameters of model
    print("\n\nGaussian Mixture Model (GMM)")
    print("============================")
    print("Number of clusters: ", num_k)
    print("Max iterations: ", gm.getMaxIter())
    # print("Number of Features: ", len(model.getFeaturesCol()), "\n")
    for i in range(num_k):
        print("--------------------------------------")
        print("Cluster ", str(i + 1), ":")
        print("# of items: ", model.summary.clusterSizes[i])
        print("Weight: ", model.weights[i])
        print("--------------------------------------")
        
####### Principal Component Analysis #######
# PCA: Reduces dimensionality of large data sets
def pca(sdf):
    # Drops NULL values
    sdf = sdf.na.drop() # checking if we can use the same csv without dropping NULL values
    # Change this so that each one is its own vector maybe ?
    assembler = VectorAssembler(inputCols=['DailyAverageDryBulbTemperature',
                                        'DailyAverageRelativeHumidity',
                                        'DailyAverageSeaLevelPressure',
                                        'DailyAverageStationPressure',
                                        'DailyAverageWetBulbTemperature',
                                        'DailyAverageWindSpeed',
                                        'DailyCoolingDegreeDays',
                                        'DailyDepartureFromNormalAverageTemperature',
                                        'DailyHeatingDegreeDays',
                                        'DailyMaximumDryBulbTemperature',
                                        'DailyMinimumDryBulbTemperature',
                                        'DailyPeakWindDirection',
                                        'DailyPeakWindSpeed',
                                        'DailySustainedWindSpeed',
                                        'DailyWeather'], outputCol= 'features')
    df = assembler.transform(sdf)

    # setup pca
    pca = PCA(k=2, inputCol="features")
    pca.setOutputCol("PCA_Features")
    # run pca
    model = pca.fit(df)
    model.setOutputCol("output")
    
    # output
    print("Principal Component Analysis")
    print("============================")
    for out in model.transform(df).collect():
        print(out.output)
    

# Machine learning display options for which algorithm to use.
def machine_menu(sdf):
    display = "\n1) Linear Regression\n2) K-Means\n"
    display += "3) Gaussian Mixture\n4) Principal Component Analysis (PCA)\n0) Exit"
    print(display)
    # Pick option
    user_input = input("Pick a option: ")
    machine_work(user_input, sdf)

# Takes the user input
def machine_work(ui,sdf):
    if ui == "1":
        linear_menu(sdf)
    elif ui == "2":
        kmeans(sdf)
    elif ui == "3":
        gaussian(sdf)
    elif ui == "4":
        pca(sdf)
    elif ui == "0":
        return
    else:
        print("Not an option, please try again")
