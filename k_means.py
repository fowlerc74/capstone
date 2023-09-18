from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
import numpy as np
import pyqtgraph as pg


#  Runs K Means Clustering to cluster data points and returns a scatter plot
def kmeans(sdf, k, column1, column2):
    # Drops NULL values
    sdf = sdf.na.drop()
    # Makes a vector that contains the features that K-Means will
    # use to predict which cluster that they should belong to
    feat_assembler = k_vector_feature_assembler(column1, column2)

    # A new DataFrame that has the features vector as a new column.
    new_df = feat_assembler.transform(sdf)

    # Trains the K-Means model and passes in 3 clusters, Will change in the future.
    kmeans_algo = KMeans(initMode="k-means||")
    kmeans_algo.setK(k)

    # Fits the new DataFrame that has the features column
    kmeans_fit = kmeans_algo.fit(new_df)
    # The output after K-Means has run, should contain
    # a prediction column that will be for the feature column
    output = kmeans_fit.transform(new_df)
    # Get the silhouette score and pass into setup_graph
    sil_output = silhouette_score(new_df)
    # Makes a scatter plot of the features and which cluster they belong to
    graph, sil_graph = setup_graph(
        output, kmeans_algo.getK(), sil_output, column1, column2
    )
    return graph, sil_graph


# Silhouette score measures how close each point in one cluster is to points in the
# neighboring cluster.
def silhouette_score(df):
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="features")
    evaluator.setMetricName("silhouette")
    evaluator.setDistanceMeasure("squaredEuclidean")

    s_score = []
    for i in range(2, 11):
        s_kmeans = KMeans(featuresCol="features", k=i)
        s_kmeans_fit = s_kmeans.fit(df)
        s_output = s_kmeans_fit.transform(df)

        score = evaluator.evaluate(s_output)
        s_score.append(score)

    return s_score


# Creates a vector that contains Daily Precipitation and Daily Average Dry Bulb Temperature
# columns and names the column "features"
def k_vector_feature_assembler(column1, column2):
    vector = VectorAssembler(
        inputCols=[column1, column2],
        outputCol="features",
    )
    return vector


# Returns the columns to choose from for K-Means.
def k_columns():
    columns = [
        "DailyAverageDryBulbTemperature",
        "DailyAverageRelativeHumidity",
        "DailyAverageSeaLevelPressure",
        "DailyAverageStationPressure",
        "DailyAverageWetBulbTemperature",
        "DailyAverageWindSpeed",
        "DailyCoolingDegreeDays",
        "DailyHeatingDegreeDays",
        "DailyMaximumDryBulbTemperature",
        "DailyMinimumDryBulbTemperature",
        "DailyPeakWindDirection",
        "DailyPeakWindSpeed",
        "DailySustainedWindSpeed",
        "DailyPrecipitation",
    ]
    return columns


# Creates the scatter plot from based on the K-Means results and returns the graph.
def setup_graph(dfoutput, n_clusters, sil_output, column1, column2):
    # Selecting the prediction column
    pred = dfoutput.select("prediction").collect()
    # Selecting the first chosen column by the user
    col1 = dfoutput.select(column1).collect()
    # Selecting the second chosen column by the user
    col2 = dfoutput.select(column2).collect()

    # Turns the selected columns into numpy arrays
    col1_arr = np.array(col1).reshape(-1)
    col2_arr = np.array(col2).reshape(-1)
    pred_arr = np.array(pred).reshape(-1)
    sil_score = np.array(sil_output)

    # Makes a plot widget and setting the background white
    plot = pg.PlotWidget(background="w")
    # Creates a scatter plot and sets hovering to true
    scatter = pg.ScatterPlotItem(hoverable=True)

    # Describing what axis' represent in the graph
    plot.setLabel("bottom", str(column1))
    plot.setLabel("left", str(column2))
    plot.setWindowTitle("K-Means")

    # Set up the silhouette graph
    sil_graph = pg.PlotWidget(background="w")
    # Describing what axis' represent in the graph
    sil_graph.setTitle("K-Means Silhouette Score")
    sil_graph.setLabel(
        "left",
        "Distance between each point in one cluster is to points in the neighboring cluster",
    )
    sil_graph.setLabel("bottom", "# of Clusters")
    sil_graph.plot(range(2, 11), sil_score, pen=pg.mkPen("b", width=3))

    # Making a hold array that will hold the position for each point
    # and color of each point and adds the points to the scatter plot
    hold = []
    for i in range(n_clusters):
        print("Cluster: ", i)
        scatter.setBrush(QColor(pg.intColor(i, n_clusters)))
        scatter.setPen(QColor(pg.intColor(i, n_clusters)))
        for j, k in zip(col1_arr[pred_arr == i], col2_arr[pred_arr == i]):
            print("Col1: ", j, ", Col2: ", k)
            hold.append({"pos": (j, k), "brush": QColor(pg.intColor(i, n_clusters))})
            scatter.addPoints(hold)
    # Adds the scatter plot to the plot widget
    plot.addItem(scatter)
    return plot, sil_graph
