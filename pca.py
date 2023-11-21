from pyspark.ml.feature import PCA, VectorAssembler
import pyqtgraph as pg


####### Principal Component Analysis #######
# PCA: Reduces dimensionality of large data sets
def pca(sdf, num_comp):
    """
    Runs Principal Component Analysis to reduce the dimensionality of the dataset.

    Parameters:
        sdf:         the dataframe to run PCA on
        num_comp:    (k value) the number of components to reduce to

    Returns:
        the scatter plot
        the explained variances
    """
    # Drops NULL values
    sdf = sdf.na.drop()

    # Create a new df with the transformed columns
    feat_vect = pca_feat_vector()
    df = feat_vect.transform(sdf)

    # Setup PCA
    print(num_comp)
    pca = PCA(k=num_comp, inputCol="features")
    pca.setOutputCol("PCA_Features")
    # Run PCA
    model = pca.fit(df)
    model.setOutputCol("output")

    data = convert_data(model.transform(df).collect(), num_comp)

    graph = setup_graph(data)
    

    return graph, model.explainedVariance

def convert_data(data, num_comp):
    converted = []
    for i in range(num_comp):
        converted.append([])

    for point in data:
        for i in range(num_comp):
            converted[i].append(point.output[i])

    return converted


def setup_graph(data):
    # Makes a plot widget and setting the background white
    plot = pg.PlotWidget(background="w")
    # Creates a scatter plot and sets hovering to true
    scatter = pg.ScatterPlotItem(hoverable=True)

    # Describing what axis' represent in the graph
    plot.setLabel("bottom", "Component 1")
    plot.setLabel("left", "Component 2")
    plot.setWindowTitle("Principal Component Analysis")

    scatter.addPoints(data[0], data[1])
    
    plot.addItem(scatter)

    return plot


# Combine input comumns into one vector
def pca_feat_vector():
    assembler = VectorAssembler(
        inputCols=[
            "DailyAverageDryBulbTemperature",
            "DailyAverageRelativeHumidity",
            "DailyAverageSeaLevelPressure",
            "DailyAverageStationPressure",
            "DailyAverageWetBulbTemperature",
            "DailyAverageWindSpeed",
            "DailyCoolingDegreeDays",
            "DailyDepartureFromNormalAverageTemperature",
            "DailyHeatingDegreeDays",
            "DailyMaximumDryBulbTemperature",
            "DailyMinimumDryBulbTemperature",
            "DailyPeakWindDirection",
            "DailyPeakWindSpeed",
            "DailySustainedWindSpeed",
        ],
        outputCol="features",
    )
    return assembler
