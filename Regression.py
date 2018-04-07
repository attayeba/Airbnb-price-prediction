from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

# Load data,remove header and split at commas since its a CSV
lines = spark.read.text("ReducedData2.csv").rdd
head = lines.first()
lines = lines.filter(lambda x: x != head)
parts = lines.map(lambda row: row.value.split(","))
df = spark.createDataFrame(parts,['id','neighbourhood','propertyType','roomType','accommodates','bathrooms','bedrooms','beds','bedType','price','minNights','maxNights','availability','numReviews','lastReview','rating'])

dataset = df.select('neighbourhood','roomType','accommodates','bathrooms','bedrooms','beds','price')
# dataset = df.select('neighbourhood','roomType','accommodates','bathrooms','bedrooms','beds','price','availability')
# dataset = df.select('neighbourhood','roomType','price')
# dataset = df.select('neighbourhood','roomType','accommodates','bathrooms','bedrooms','beds','price','availability','rating')
# dataset = df.select('neighbourhood','roomType','accommodates','bathrooms','bedrooms','beds','price','availability','rating')
# dataset = df.select('neighbourhood','roomType','accommodates','bathrooms','bedrooms','beds','price','availability','rating')
# dataset = df.select('neighbourhood','roomType','accommodates','bathrooms','bedrooms','beds','price','availability','rating')
# dataset = df.select('neighbourhood','roomType','accommodates','bathrooms','bedrooms','beds','price','availability','rating')

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
dataset = dataset.withColumn('prices', dataset.price)
dataset = dataset.drop('price')

	
# Make dataframe into LIBSVM format
data = dataset.rdd.map(lambda x:(Vectors.dense(x[0:-2]), x[-1])).toDF(["features", "label"])

# dataset.printSchema()
# dataset.show(15)
# data.show(15)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=124)

# Train a GBT model.
gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, gbt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(20)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

gbtModel = model.stages[1]
print(gbtModel)  # summary only