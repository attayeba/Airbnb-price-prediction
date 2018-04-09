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
		
#LOAD ORIGINAL DATASET
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

# Move price column to the end of the table	
dataset = dataset.withColumn('listId', dataset.id)
dataset = dataset.drop('id')

#LOAD PREPARED DATASET
# Load data,remove header and split at commas since its a CSV
lines = spark.read.text("clustered.csv").rdd
head = lines.first()
lines = lines.filter(lambda x: x != head)
parts = lines.map(lambda row: row.value.split(","))
df = spark.createDataFrame(parts,["id","accommodates","bathrooms","bedrooms","beds","price","Shared room","Entire home/apt","Private room","Queens","Brooklyn","Staten Island","Manhattan","Bronx","testTrain","listId"])

#Put in the same order
testTrainData = df.select("accommodates","bathrooms","bedrooms","beds","price","Shared room","Entire home/apt","Private room","Queens","Brooklyn","Staten Island","Manhattan","Bronx","testTrain","listId")

# Make dataframe into LIBSVM format
datak = testTrainData.rdd.map(lambda x:(Vectors.dense(x[0:-2]), x[-1])).toDF(["features", "label"])

# Trains a k-means model.
kmeans = KMeans().setK(10).setSeed(123)
model = kmeans.fit(datak)

# Make predictions
predictions = model.transform(datak)

# Clustered dataframe
testTrainData = testTrainData.select('listId','testTrain')
testTrainData = testTrainData.withColumnRenamed('listId','newId')
OuterCluster = dataset.join(predictions, dataset.listId == predictions.label)
OuterCluster = OuterCluster.drop('features').drop('label')
OuterCluster = OuterCluster.join(testTrainData, OuterCluster.listId == testTrainData.newId)
OuterCluster = OuterCluster.drop('newId')
OuterCluster=OuterCluster.withColumn('testTrain', OuterCluster['testTrain'].cast(IntegerType()))

rm = []
for iter in range(0,10):
	# regress cluster iter
	clusters = OuterCluster.rdd.filter(lambda x: x[-2]==iter)
	
	# if cluster is empty, ignore
	if(clusters.count() != 0):
		dataset = spark.createDataFrame(clusters,['accommodates','bathrooms','bedrooms','beds','price','Shared room','Entire home/apt','Private room','Queens','Brooklyn','Staten Island','Manhattan','Bronx'])
		dataset = dataset.withColumn('prices', dataset.price)
		dataset = dataset.drop('price')

		# Make dataframe into LIBSVM format
		data = dataset.rdd.map(lambda x:(Vectors.dense(x[0:-2]), x[-1],x[-2])).toDF(["features", "label","classes"])

		featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(data)

		# Split the data into training and test sets (30% held out for testing)
		trainingData = data.filter(data.classes.isin(1))
		testData = data.filter(data.classes.isin(0))

		# if cluster is all training, ignore
		if(testData.count() != 0):
			# Train a GBT model.
			gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

			# Chain indexer and GBT in a Pipeline
			pipeline = Pipeline(stages=[featureIndexer, gbt])

			# Train model.  This also runs the indexer.
			model = pipeline.fit(trainingData)

			# Make predictions.
			predictions = model.transform(testData)

			# Select (prediction, true label) and compute test error
			evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
			rmse = evaluator.evaluate(predictions)
			rm.append(rmse)

# print the lowest rmse
rm = sorted(rm)
print("Lowest rmse: "+str(rm[0]))

# print the average rmse
sum = 0
for val in rm:
	sum = sum+val

rmse = sum/(len(rm))
print("Global rmse: "+str(rmse))