from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import lit
import pandas
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()
		
# Load data,remove header and split at commas since its a CSV
lines = spark.read.text("ReducedData.csv").rdd
head = lines.first()
lines = lines.filter(lambda x: x != head)
parts = lines.map(lambda row: row.value.split(","))
df = spark.createDataFrame(parts,['id','neighbourhood','propertyType','roomType','accommodates','bathrooms','bedrooms','beds','bedType','price','minNights','maxNights','availability','numReviews','lastReview','rating'])
dataset = df.select('id','neighbourhood','roomType','accommodates','bathrooms','bedrooms','beds','price')

# One-hot encode the roomType values
roomType = dataset.select('roomType').distinct().rdd.flatMap(lambda x: x).collect()
for val in roomType:
	dataset = dataset.withColumn(val, when(col("roomType") == val, 1). otherwise(0))
	
# One-hot encode the neighborhood values	
neighbourhood = df.select('neighbourhood').distinct().rdd.flatMap(lambda x: x).collect()
for val in neighbourhood:
	dataset = dataset.withColumn(val, when(col("neighbourhood") == val, 1). otherwise(0))

# Remove the original roomType and neighbourhood columns to not duplicate
dataset = dataset.drop('roomType').drop('neighbourhood')

# Change all column types to intergers
names = dataset.schema.names
for col in names:
	dataset=dataset.withColumn(col, dataset[col].cast(IntegerType()))

# Move id column to the end of the table	
dataset = dataset.withColumn('listId', dataset.id)
dataset = dataset.drop('id')

# split dataset in two
(trainingData, testData) = dataset.randomSplit([0.7, 0.3],seed=123)

# add column indicating if test or train
trainingData = trainingData.withColumn('testTrain',lit(1))
testData = testData.withColumn('testTrain',lit(0))

# save original prices and change with avg price
OgPrice = testData.select('listId','price')

# Rename columns to be called easily
testData = testData.withColumnRenamed('Shared room','shared').withColumnRenamed('Entire home/apt','home').withColumnRenamed('Private room','private').withColumnRenamed('Staten Island','StatenIsland')
trainingData = trainingData.withColumnRenamed('Shared room','shared').withColumnRenamed('Entire home/apt','home').withColumnRenamed('Private room','private').withColumnRenamed('Staten Island','StatenIsland')
trainingData = trainingData.select("accommodates","bathrooms","bedrooms","beds","shared","home","private","Queens","Brooklyn","StatenIsland","Manhattan","Bronx","testTrain","listId","price")
testData = testData.select("accommodates","bathrooms","bedrooms","beds","shared","home","private","Queens","Brooklyn","StatenIsland","Manhattan","Bronx","testTrain","listId","price")

# Join train and test
testTrainData = trainingData.union(testData)

# Make dataframe into LIBSVM format
data = testTrainData.rdd.map(lambda x:(Vectors.dense(x[0:-4]), x[-3],x[-2],x[-1])).toDF(["features", "classes","listId","label"])

# Automatically identify categorical features, and index them.
featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(data)

# Split the data into training and test sets (30% held out for testing)
trainingData = data.filter(data.classes.isin(1))
testData = data.filter(data.classes.isin(0))

# Train a GBT model.
gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, gbt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
cross = predictions.select('prediction','listId')

# Join with predicted price
finalDf = testTrainData.join(cross, 'listId', 'left_outer')

# Merge with appropriate price
finalDf = finalDf.withColumn('price', when(finalDf.testTrain == 0, finalDf.prediction).otherwise(finalDf.price))
finalDf = finalDf.drop('prediction')
finalDf = finalDf.select("accommodates","bathrooms","bedrooms","beds","price","shared","home","private","Queens","Brooklyn","StatenIsland","Manhattan","Bronx","testTrain","listId")

# Save into file to reduce future computation
finalDf.toPandas().to_csv('clustered.csv')