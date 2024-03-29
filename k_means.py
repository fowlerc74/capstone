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
    # The output after K-Means has run, should contain
    # a prediction column that will be for the feature column
    output = kmeans_fit.transform(new_df)
    cluster_centers = clusters(centers, k)
    # Get the silhouette score and pass into setup_graph
    sil_output = silhouette_score(new_df)

    sil_graph = setup_sil_graph(sil_output)

    return output, sil_graph, cluster_centers


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
    if column3 == "None" and column2 == "None":
        print("Here 1: ", column2)
        vector = VectorAssembler(
            inputCols=[column1],
            outputCol="features",
        )
    if column2 != "None":
        print("Here 2: ", column2)
        vector = VectorAssembler(inputCols=[column1, column2], outputCol="features")

    if column1 != "None" and column2 != "None" and column3 != "None":
        vector = VectorAssembler(
            inputCols=[column1, column2, column3],
            outputCol="features",
        )
    return vector


# Returns the list of columns.
def columns(): #TODO make to a constant
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
# displays the cluster centers and the color of what center is which on the graph.
def clusters(centers, n_clusters):
    cluster_center_box = QVBoxLayout()
    center_title = QLabel("Cluster Centers: [column1],[column2],[column3]")
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

    return cluster_center_box


# The columns that can be selected for the x value for the scatter plot.
def user_select_x(column1, column2, column3):
    options = [str(column1), str(column2), str(column3)]
    x_value = QComboBox()
    x_value.addItems(options)
    return x_value


# The columns that can be selected for the y value for the scatter plot.
def user_select_y(column1, column2, column3):
    options = [str(column1), str(column2), str(column3)]
    y_value = QComboBox()
    y_value.addItems(options)
    return y_value


# Creates the silhouette graph that will display the results of what each score
# is and turn those results into a line graph.
def setup_sil_graph(sil_output):
    # Make an array that holds the results from the silhouette score function
    sil_score = np.array(sil_output)

    # Set up the silhouette graph
    sil_graph = pg.PlotWidget(background="w")
    # Describing what axes represents for the graph
    sil_graph.setTitle("K-Means Silhouette Score")
    sil_graph.setLabel("left", "Average distance between clusters")
    sil_graph.setLabel("bottom", "# of Clusters")
    sil_graph.plot(range(2, 11), sil_score, pen=pg.mkPen("b", width=3))

    return sil_graph


# Creates the scatter plot from based on the K-Means results and returns the graph.
def setup_graph_k(dfoutput, n_clusters, column1, column2, hover, sdf):
    # Selecting the prediction column
    pred = dfoutput.select("prediction").collect()
    # Selecting the first chosen column for the X axis
    col1 = dfoutput.select(column1).collect()
    # Selecting the second chosen column for the Y axis
    col2 = dfoutput.select(column2).collect()

    # Turns the selected columns into numpy arrays
    col1_arr = np.array(col1).reshape(-1)
    col2_arr = np.array(col2).reshape(-1)
    pred_arr = np.array(pred).reshape(-1)

    # Makes a plot widget and setting the background white
    plot = pg.PlotWidget(background="w")
    # Creates a scatter plot and sets hovering to true
    scatter = pg.ScatterPlotItem(hoverable=True, hoverPen="g")
    # Describing what the axes represent for the scatter plot
    plot.setLabel("bottom", str(column1))
    plot.setLabel("left", str(column2))
    plot.setWindowTitle("K-Means")

    # Define the function to run on hover
    def on_hover(evt, points):
        if points.size > 0:
            hover.point_query(points, sdf, column1, column2)
    # Connect that function to the hover signal
    scatter.sigHovered.connect(on_hover)

    # Create the scatterplot and color it by cluster
    hold_k_points = []
    for i in range(n_clusters):
        scatter.setBrush(QColor(pg.intColor(i, n_clusters)))
        for j, k in zip(col1_arr[pred_arr == i], col2_arr[pred_arr == i]):
            hold_k_points.append(
                {
                    "pos": (j, k),
                    "brush": QColor(pg.intColor(i, n_clusters)),
                    "symbol": "o",
                }
            )
            scatter.addPoints(hold_k_points)

    plot.addItem(scatter)
    return plot

# The number of points to query
# This is because of the bug where each scatterplot point is repeated many times
NUM_POINTS_QUERY = 3

# A hover class used to store the hover variable 
class Hover:
    # Init hover var
    def __init__(self, hover_var):
        self.var = hover_var

    def set_hover_var(self, hover_var):
        """
        Sets the variable to be shown on hover

        Parameters:
            hover_var: the new hover variable
        """
        self.var = hover_var

    def point_query(self, points, sdf, column1, column2):
        """
        Query the points when hovering and update their data. 

        Parameters:
            self:
            points: the list of points being hovered over
            sdf: the dataframe with the data
            column1: the name of the x axis variable
            column2: the name of the y axis variable
        """
        x = points[0].pos()[0]
        y = points[0].pos()[1]

        for point in points[:NUM_POINTS_QUERY]:
            row = sdf.select(self.var, 'DATE').where(column1 + "==" + str(x) + " and " + column2 + "==" + str(y)).first()
            point_data = "\n" + "Date: " + str(row['DATE'])
            point_data += "\n" + self.var + ": " + str(row[self.var])
            point.setData(point_data)
