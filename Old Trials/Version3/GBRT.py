from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.linalg import Vectors
from  pyspark.sql.functions import abs

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

dataset = dataset.withColumn('prices', dataset.price)
dataset = dataset.drop('price')

# Make dataframe into LIBSVM format
data = dataset.rdd.map(lambda x:(Vectors.dense(x[0:-2]), x[-1])).toDF(["features", "label"])

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=123)

# Train a GBT model.
gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=20)

# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, gbt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)
predictions = predictions.withColumn('Mean', abs((predictions.label - predictions.prediction)/predictions.prediction))
value = predictions.select('Mean').rdd.flatMap(lambda x: x).collect()

# print the average rmse
sum = 0
for val in value:
	sum = sum+val

rmse = sum/(len(value))
print("Global rmse: "+str(rmse))



# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)