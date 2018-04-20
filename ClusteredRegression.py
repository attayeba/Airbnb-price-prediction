from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
from  pyspark.sql.functions import abs

from scipy import stats
import scipy
import pandas as pd
import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

# Load data,remove header and split at commas since its a CSV
lines = spark.read.text("format.csv").rdd
head = lines.first()
lines = lines.filter(lambda x: x != head)
parts = lines.map(lambda row: row.value.split(","))
df = spark.createDataFrame(parts,['temp','id','latitude','longitude','accommodates','bathrooms','bedrooms','beds','price','availability','rating','Queens','Brooklyn','Manhattan','Financial District','Midtown','Hells Kitchen','Greenwich Village','Clinton Hill','Washington Heights','Ditmars Steinway','Flushing','Upper East Side','East Harlem','Astoria','Lower East Side','East Village','Carroll Gardens','Fort Greene','Gramercy','Sunset Park','Williamsburg','Sunnyside','Long Island City','Chinatown','Ridgewood','East Flatbush','Upper West Side','West Village','Kips Bay','Bedford-Stuyvesant','Prospect-Lefferts Gardens','Harlem','Inwood','SoHo','Murray Hill','Crown Heights','Chelsea','Flatbush','Bushwick','Nolita','Prospect Heights','South Slope','Park Slope','Morningside Heights','Greenpoint','Apartment','Townhouse','Resort','Guest suite','Timeshare','Hut','Casa particular (Cuba)','Camper/RV','Boutique hotel','Loft','Guesthouse','Hostel','Cave','Villa','Aparthotel','Other','Serviced apartment','Earth house','Treehouse','Hotel','In-law','Cottage','Dorm','Condominium','House','Chalet','Yurt','Tent','Boat','Tiny house','Vacation home','Bungalow','Bed and breakfast','Cabin','Shared room','Entire home/apt','Private room','Airbed','Futon','Pull-out Sofa','Couch','Real Bed'])
dataset = df.drop('temp')

# Change column types to intergers
names = ['id','accommodates','bedrooms','beds','price','availability','rating','Queens','Brooklyn','Manhattan','Financial District','Midtown','Hells Kitchen','Greenwich Village','Clinton Hill','Washington Heights','Ditmars Steinway','Flushing','Upper East Side','East Harlem','Astoria','Lower East Side','East Village','Carroll Gardens','Fort Greene','Gramercy','Sunset Park','Williamsburg','Sunnyside','Long Island City','Chinatown','Ridgewood','East Flatbush','Upper West Side','West Village','Kips Bay','Bedford-Stuyvesant','Prospect-Lefferts Gardens','Harlem','Inwood','SoHo','Murray Hill','Crown Heights','Chelsea','Flatbush','Bushwick','Nolita','Prospect Heights','South Slope','Park Slope','Morningside Heights','Greenpoint','Apartment','Townhouse','Resort','Guest suite','Timeshare','Hut','Casa particular (Cuba)','Camper/RV','Boutique hotel','Loft','Guesthouse','Hostel','Cave','Villa','Aparthotel','Other','Serviced apartment','Earth house','Treehouse','Hotel','In-law','Cottage','Dorm','Condominium','House','Chalet','Yurt','Tent','Boat','Tiny house','Vacation home','Bungalow','Bed and breakfast','Cabin','Shared room','Entire home/apt','Private room','Airbed','Futon','Pull-out Sofa','Couch','Real Bed']
for col in names:
	dataset=dataset.withColumn(col, dataset[col].cast(IntegerType()))
	
# Change column types to double
names = ['latitude','longitude','bathrooms']
for col in names:
	dataset=dataset.withColumn(col, dataset[col].cast(DoubleType()))

# Rearrange columns for libsvm
cData = dataset.withColumn('listId', dataset.id)
cData = cData.drop('price','id')

# Make dataframe into LIBSVM format
data = cData.rdd.map(lambda x:(Vectors.dense(x[0:-2]), x[-1])).toDF(["features", "label"])

# Trains a k-means model.
kmeans = KMeans().setK(7).setSeed(123)
model = kmeans.fit(data)

# Make predictions
predictions = model.transform(data)

# Join dataframe to have clusters
OuterCluster = dataset.join(predictions, dataset.id == predictions.label)
OuterCluster = OuterCluster.drop('features','label','id')

size = []
errorPercentage = []
rm = []
error = []
for iter in range(0,7):
	# Get the values for cluster i
	clusters = OuterCluster.rdd.filter(lambda x: x[-1]==iter)

	# Create a dataframe of the cluster
	dataset = spark.createDataFrame(clusters,['latitude','longitude','accommodates','bathrooms','bedrooms','beds','price','availability','rating','Queens','Brooklyn','Manhattan','Financial District','Midtown','Hells Kitchen','Greenwich Village','Clinton Hill','Washington Heights','Ditmars Steinway','Flushing','Upper East Side','East Harlem','Astoria','Lower East Side','East Village','Carroll Gardens','Fort Greene','Gramercy','Sunset Park','Williamsburg','Sunnyside','Long Island City','Chinatown','Ridgewood','East Flatbush','Upper West Side','West Village','Kips Bay','Bedford-Stuyvesant','Prospect-Lefferts Gardens','Harlem','Inwood','SoHo','Murray Hill','Crown Heights','Chelsea','Flatbush','Bushwick','Nolita','Prospect Heights','South Slope','Park Slope','Morningside Heights','Greenpoint','Apartment','Townhouse','Resort','Guest suite','Timeshare','Hut','Casa particular (Cuba)','Camper/RV','Boutique hotel','Loft','Guesthouse','Hostel','Cave','Villa','Aparthotel','Other','Serviced apartment','Earth house','Treehouse','Hotel','In-law','Cottage','Dorm','Condominium','House','Chalet','Yurt','Tent','Boat','Tiny house','Vacation home','Bungalow','Bed and breakfast','Cabin','Shared room','Entire home/apt','Private room','Airbed','Futon','Pull-out Sofa','Couch','Real Bed','prediction'])
	dataset = dataset.withColumn('prices', dataset.price)
	dataset = dataset.drop('price')
	
	# remove erroneous values
	pan = dataset.toPandas()
	DataPrice = pan[['prices']]
	pan = pan[(np.abs(scipy.stats.zscore(DataPrice)) < 2.5).all(axis=1)]
	dataset = spark.createDataFrame(pan)
	size.append(dataset.count())
	
	# Make dataframe into LIBSVM format
	data = dataset.rdd.map(lambda x:(Vectors.dense(x[0:-2]), x[-1])).toDF(["features", "label"])

	# Set maxCategories so features with > 5 distinct values are treated as continuous.
	featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(data)

	# Split the data into training and test sets (30% held out for testing)
	(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=123)

	# Train a GBT model.
	gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

	# Chain indexer and GBT in a Pipeline
	pipeline = Pipeline(stages=[featureIndexer, gbt])

	# Train model.  This also runs the indexer.
	model = pipeline.fit(trainingData)

	# Make predictions.
	predictions = model.transform(testData)
	predictions = predictions.withColumn('Mean', abs((predictions.label - predictions.prediction)/predictions.prediction))
	value = predictions.select('Mean').rdd.flatMap(lambda x: x).collect()

	# get average MAPE
	sum = 0
	for val in value:
		sum = sum+val

	mape = sum/(len(value))
	# print("MAPE: "+str(mape))
	# errorPercentage.append(mape)

	# Select (prediction, true label) and compute test error
	evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
	rmse = evaluator.evaluate(predictions)

	# print("Iter rmse"+str(iter)+": "+str(rmse))
	# rm.append(rmse)
	error.append([mape,rmse])

error = sorted(error,key=lambda x:x[0])
print("MAPE: " + str(error[0][0]))
print("RMSE: " + str(error[0][1]))
'''
rm = sorted(rm)
errorPercentage = sorted(errorPercentage)

print("best rmse: "+str(rm[0]))

sum = 0
for val in rm:
	sum = sum+val

sse = sum/(len(rm))
print("average rmse: "+str(sse))


print("best mape: "+str(errorPercentage[0]))

sum = 0
for val in errorPercentage:
	sum = sum+val

sse = sum/(len(errorPercentage))
print("average mape: "+str(sse))



sum = 0
for val in size:
	sum = sum+val
print("size: "+str(sum))
'''