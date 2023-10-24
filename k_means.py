from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import Qt
import numpy as np
import pyqtgraph as pg


#  Runs K Means Clustering to cluster data points and returns a scatter plot
def kmeans(sdf, k, column1, column2, column3):
    # Drops NULL values
    sdf = sdf.na.drop()
    # Makes a vector that contains the features that K-Means will
    # use to predict which cluster that they should belong to

    feat_assembler = k_vector_feature_assembler(column1, column2, column3)

    # A new DataFrame that has the features vector as a new column.
    new_df = feat_assembler.transform(sdf)

    # Trains the K-Means model and passes in 3 clusters, Will change in the future.
    kmeans_algo = KMeans(initMode="k-means||")
    kmeans_algo.setK(k)

    # Fits the new DataFrame that has the features column
    kmeans_fit = kmeans_algo.fit(new_df)
    centers = kmeans_fit.clusterCenters()
    legend_box = legend(column1, column2, column3, centers, k)
    # The output after K-Means has run, should contain
    # a prediction column that will be for the feature column
    output = kmeans_fit.transform(new_df)
    # Get the silhouette score and pass into setup_graph
    sil_output = silhouette_score(new_df)
    # Makes a scatter plot of the features and which cluster they belong to
    graph, sil_graph = setup_graph(
        output, kmeans_algo.getK(), sil_output, column1, column2, column3, sdf
    )

    return graph, sil_graph, legend_box


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


# Creates a vector that contains the variables that the user chosen
# and names the vector "features".
def k_vector_feature_assembler(column1, column2, column3):
    print(column1)
    print(column2)
    print(column3)
    if column3 == "None" and column2 == "None":
        print("Here 1: ", column2)
        vector = VectorAssembler(
            inputCols=[column1],
            outputCol="features",
        )
    if column2 != "None":
        print("Here 2: ", column2)
        vector = VectorAssembler(inputCols=[column1, column2], outputCol="features")

    # if column2 == "None" and column3 == "None":
    #     # Display a error message, must have at least 2
    #     pass
    if column1 != "None" and column2 != "None" and column3 != "None":
        vector = VectorAssembler(
            inputCols=[column1, column2, column3],
            outputCol="features",
        )
    return vector


# Returns the columns to choose from for K-Means.
def k_columns():
    columns = [
        "None",
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


# Makes the the variable section hold additional information.
# Currently displays what the symbols are and displays the cluster centers
# and the color of what center is which on the graph.
def legend(column1, column2, column3, centers, n_clusters):
    legend_box = QVBoxLayout()
    cluster_center_box = QVBoxLayout()
    overall_layout = QHBoxLayout()

    legend_title = QLabel("Legend:")
    legend_box.addWidget(legend_title)

    col1 = "X: " + str(column1)
    col2 = "O: " + str(column2)
    col1_label = QLabel(col1)
    legend_box.addWidget(col1_label)
    if column3 != "None":
        col3 = str(column3) + " = *"
        col3_label = QLabel(col3)
        legend_box.addWidget(col3_label)

    if column2 != "None":
        col2 = str(column2) + " = x"
        col2_label = QLabel(col2)
        legend_box.addWidget(col2_label)

    center_title = QLabel("Cluster centers:")
    cluster_center_box.addWidget(center_title)
    i = 0
    for center in centers:
        string_center = str(center)
        center_label = QLabel(string_center)
        center_label.setAutoFillBackground(True)
        color = QColor(pg.intColor(i, n_clusters))
        set_color = "background-color : " + color.name()
        center_label.setStyleSheet(set_color)
        cluster_center_box.addWidget(center_label)
        i += 1

    overall_layout.addLayout(legend_box)
    overall_layout.addLayout(cluster_center_box)

    return overall_layout


# Creates the scatter plot from based on the K-Means results and returns the graph.
def setup_graph(dfoutput, n_clusters, sil_output, column1, column2, column3, sdf):
    # Selecting the prediction column
    pred = dfoutput.select("prediction").collect()
    # Selecting the first chosen column by the user
    col1 = dfoutput.select(column1).collect()

    # Turns the selected columns into numpy arrays
    col1_arr = np.array(col1).reshape(-1)
    pred_arr = np.array(pred).reshape(-1)
    sil_score = np.array(sil_output)

    # Makes a plot widget and setting the background white
    plot = pg.PlotWidget(background="w")
    # Creates a scatter plot and sets hovering to true
    scatter = pg.ScatterPlotItem(hoverable=True, hoverPen='g')

    # When points are hovered over, print
    def on_hover(evt, points):
        if points.size > 0 and points[0].data() == None:
            print(1)
            x = points[0].pos()[0]
            y = points[0].pos()[1]

            for point in points:
                row = sdf.select("DATE").where(column1 + "==" + str(y)).first()
                point_data = "\nDate: " + str(row.DATE)
                point_data += "\n" + column1 + ": " + str(y)
                point.setData(point_data)

    scatter.sigHovered.connect(on_hover)

    # Describing what axis' represent in the graph
    plot.setLabel("bottom", "Days")
    plot.setWindowTitle("K-Means")

    # Set up the silhouette graph
    sil_graph = pg.PlotWidget(background="w")
    # Describing what axis' represent in the graph
    sil_graph.setTitle("K-Means Silhouette Score")
    sil_graph.setLabel("left", "Average distance between clusters")
    sil_graph.setLabel("bottom", "# of Clusters")
    sil_graph.plot(range(2, 11), sil_score, pen=pg.mkPen("b", width=3))

    counter = 0

    date_arr = []
    hold_daily = dfoutput.select("DATE").collect()

    while len(hold_daily) != counter:
        counter += 1
        date_arr.append(counter)

    # Making a hold array that will hold the position for each point
    # and color of each point and adds the points to the scatter plot
    hold = []

    if column3 != "None":
        col3 = dfoutput.select(column3).collect()
        col3_arr = np.array(col3).reshape(-1)
        for i in range(n_clusters):
            scatter.setBrush(QColor(pg.intColor(i, n_clusters)))
            for j, k in zip(date_arr, col3_arr[pred_arr == i]):
                hold.append(
                    {
                        "pos": (date_arr[j], k),
                        "brush": QColor(pg.intColor(i, n_clusters)),
                        "symbol": "star",
                    }
                )
                scatter.addPoints(hold)

    if column2 != "None":
        col2 = dfoutput.select(column2).collect()
        col2_arr = np.array(col2).reshape(-1)
        for i in range(n_clusters):
            scatter.setBrush(QColor(pg.intColor(i, n_clusters)))
            for j, k in zip(date_arr, col2_arr[pred_arr == i]):
                hold.append(
                    {
                        "pos": (date_arr[j], k),
                        "brush": QColor(pg.intColor(i, n_clusters)),
                        "symbol": "x",
                    }
                )
                scatter.addPoints(hold)

    for i in range(n_clusters):
        scatter.setBrush(QColor(pg.intColor(i, n_clusters)))
        for j, k in zip(date_arr, col1_arr[pred_arr == i]):
            hold.append(
                {
                    "pos": (date_arr[j], k),
                    "brush": QColor(pg.intColor(i, n_clusters)),
                    "symbol": "o",
                }
            )
            scatter.addPoints(hold)

    # Adds the scatter plot to the plot widget
    plot.addItem(scatter)
    return plot, sil_graph
