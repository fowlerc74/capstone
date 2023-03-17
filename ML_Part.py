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
# Learn more about what goes into linear regression model
def linear(sdf):
    
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

    # final data consists of features and label which is daily precipitation
    final_data = output.select('Features', 'DailyPrecipitation')
    # Splitting the data into train and test
    train_data, test_data = final_data.randomSplit([0.7, 0.3])
    # Display the train data based on the final_split split
    train_data.show()

    # Creating an object of class LinearRegression
    # Object takes Features and label as input arguments
    rain_lr = LinearRegression(featuresCol='Features', labelCol='DailyPrecipitation')
    # Pass train_data to train model
    trained_rain_model = rain_lr.fit(train_data)
    # Evaluating model trained for Rsquared error
    rain_results = trained_rain_model.evaluate(train_data)
    print('\nRsquared Error: ', rain_results.r2)
    # R2 value shows accuracy of model
    # Model accuracy is very good and can be used for predictive analysis

    # Testing Model on unlabeled data
    # Create unlabeled data from test_data
    # Testing model on unlabeled data
    unlabeled_data = test_data.select('Features')
    # unlabeled_data.show(5)

    # Display the results from test data
    predictions = trained_rain_model.transform(unlabeled_data)
    predictions.show()

    # TODO put values into array to display on graph
    hold_daily = test_data.select('DailyPrecipitation').toPandas()
    daily_rain = list(hold_daily['DailyPrecipitation'])
    hold_prediction = predictions.select('prediction').toPandas()
    daily_prediction = list(hold_prediction['prediction'])
    # for i in daily_prediction:
    #     print("Predictions: " , i)
    # for j in daily_rain:
    #     print("Actual: ", j)
    
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
        linear(sdf)
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
