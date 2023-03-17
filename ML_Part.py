from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
import matplotlib.pyplot as plt


from pyspark.mllib.clustering import GaussianMixture
from numpy import array


####################### Linear Regression #########################
def linear(sdf, ui):
    # Display column name and data types
    # sdf.printSchema()
    sdf = sdf.na.drop()
    # Creates vectors from dataset 
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

    # Display the results from test data
    predictions = trained_rain_model.transform(unlabeled_data)

    if ui == "1":
        sdf.show()
        machine_menu(sdf)
    elif ui == "2":
        predictions.show()
        machine_menu(sdf)
    elif ui == "3":
        train_data.show()
        predictions.show()
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
        train_data.show()
        predictions.show()
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
def kmeans(sdf):
    # Display column name and data types
    sdf.printSchema()
    sdf = sdf.na.drop()
    # Creates vectors from dataset 
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
                                            'DailyWeather'], outputCol= 'Features')
    output = assembler.transform(sdf)

    scale = StandardScaler(inputCol='Features', outputCol='standardized')
        
    data_scale = scale.fit(output)
    data_scale_output = data_scale.transform(output)

    silhouette_score = []
    evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
                                        metricName='silhouette',distanceMeasure='squaredEuclidean')
    for i in range(2,10):
        KMeans_algo = KMeans(featuresCol='standardized', k = i)
        KMeans_fit = KMeans_algo.fit(data_scale_output)
        KOutput = KMeans_fit.transform(data_scale_output)
        score = evaluator.evaluate(KOutput)
        silhouette_score.append(score)
        # print("Silhouette Score: ", score)
    fig, ax = plt.subplots(1,1, figsize = (8,6))
    ax.plot(range(2,10), silhouette_score)
    ax.set_xlabel('k')
    ax.set_ylabel('weather')
    fig.savefig('kmeans.png')

########## Naive Bayes ##########
# 
def naive_bayes(sdf):
    # def parseLine(line):
    #     parts = line.split(',')
    #     label = float(parts[0])
    #     features = Vectors.dense([float(x) for x in parts[1].split(' ')])
    #     return LabeledPoint(label, features)
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
                                        'DailyWeather'], outputCol= 'Features')
    transformed = assembler.transform(sdf)
    data = (transformed.select(col("outcome_column").alias("label"), col("features")).rdd \
        .map(lambda row: LabeledPoint(row.label, row.features)))
    print(type(data))

    # Split data aproximately into training (60%) and test (40%)
    training, test = data.randomSplit([0.6, 0.4], seed = 0)

    # Train a naive Bayes model.
    model = NaiveBayes.train(training, 1.0)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda p : (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda x, v: x == v).count() / test.count()
    print(accuracy)

####### Gaussian Mixture #######
#
def gaussian(sdf):
    # parse the data
    parsedData = sdf.rdd.map(lambda line: array([float(x) for x in line.strip().split(' ')]))

    # Build the model (cluster the data)
    gmm = GaussianMixture.train(parsedData, 2)

    # output parameters of model
    for i in range(2):
        print ("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
            "sigma = ", gmm.gaussians[i].sigma.toArray())


# Machine learning display options for which algorithm to use.
def machine_menu(sdf):
    display = "\n1) Linear Regression\n2) K-Means\n3) Naive Bayes\n4) Gaussian Mixture\n0) Exit"
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
        naive_bayes(sdf)
    elif ui == "4":
        gaussian(sdf)
    elif ui == "0":
        return
    else:
        print("Not an option, please try again")
