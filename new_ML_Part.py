from pyspark.ml.feature import VectorAssembler,PCA, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col
import numpy as np

# Perform Linear Regression on selected columns from the csv file chosen
def linear(sdf):
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

    return predictions, test_data

#################### K-Means ######################
def kmeans(sdf, n_clusters, assembler):
    # Drops NULL values
    sdf = sdf.na.drop()
    # Same DataFrame with features added to it
    new_df = assembler.transform(sdf)
    # Trains the K-Means model and pass in a certain number of clusters
    kmeans_algo = KMeans(initMode = "k-means||")
    kmeans_algo.setK(n_clusters)
    
    # Fits the features column to perform K-Means.
    kmeans_fit = kmeans_algo.fit(new_df)
    # Full output that has the cluster prediction and features added to the data frame.
    output = kmeans_fit.transform(new_df)

    evaluator = ClusteringEvaluator(predictionCol = 'prediction', featuresCol = 'features')
    evaluator.setMetricName('silhouette')
    evaluator.setDistanceMeasure('squaredEuclidean')
    # Silhouette score measures how close each point in one cluster is to points in the
    # neighboring cluster
    silhouette_score = []

    for i in range(2, 10):
        silhouette_Kmeans = KMeans(featuresCol = 'features', k = i)
        silhouette_Kmeans_fit = silhouette_Kmeans.fit(new_df)
        silhouette_output = silhouette_Kmeans_fit.transform(new_df)

        score = evaluator.evaluate(silhouette_output)
        silhouette_score.append(score)
        
    return output, kmeans_algo, silhouette_score

####### Gaussian Mixture #######
# Clustering Algorithm
def gaussian(sdf, num_clusters):
    # Drops NULL values
    sdf = sdf.na.drop()
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
    gm = GaussianMixture(k=num_clusters, tol=.001)
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
def pca(sdf, num_comp, num_results):
    """
    Runs Principal Component Analysis to reduce the dimensionality of the dataset. 

    sdf:         the dataframe to run PCA on
    num_comp:    (k value) the number of components to reduce to
    num_results: the number of reduced datapoints to return 
    """

    # Drops NULL values
    sdf = sdf.na.drop() 
    # Can remove some of these if wanted
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
                                        'DailySustainedWindSpeed'], outputCol= 'features')
    # Transform the dataframe to have the new columns
    df = assembler.transform(sdf)

    # Setup pca
    print(num_comp)
    pca = PCA(k=num_comp, inputCol="features")
    pca.setOutputCol("PCA_Features")
    # Run pca
    model = pca.fit(df)
    model.setOutputCol("output")

    return model, model.transform(df).collect()[:num_results]