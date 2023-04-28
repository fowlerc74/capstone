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

####### Principal Component Analysis #######
# PCA: Reduces dimensionality of large data sets
def pca(sdf):
    # Drops NULL values
    sdf = sdf.na.drop() # checking if we can use the same csv without dropping NULL values
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
    print("Explained Variance: ", model.explainedVariance)
    num = 5
    print("First ", num, " values:")
    for out in model.transform(df).collect()[:num]:
        print(out.output)