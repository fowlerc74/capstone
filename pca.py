from pyspark.ml.feature import PCA, VectorAssembler


####### Principal Component Analysis #######
# PCA: Reduces dimensionality of large data sets
def pca(sdf, num_comp, num_results):
    """
    Runs Principal Component Analysis to reduce the dimensionality of the dataset.

    Parameters:
        sdf:         the dataframe to run PCA on
        num_comp:    (k value) the number of components to reduce to
        num_results: the number of reduced datapoints to return

    Returns:
        the model,
        the requested number of modified datapoints
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

    return model, model.transform(df).collect()[:num_results]


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
