from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors

spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

lines = spark.read.text("Dataset.csv").rdd
head = lines.first()
lines = lines.filter(lambda x: x != head)

parts = lines.map(lambda row: row.value.split(","))
dataRdd = parts.map(lambda p: Row(roomId=(p[0]), surveyId=(p[1]),hostId=(p[2]), roomType=(p[3]), country=(p[4]), city=(p[5]), borough=(p[6]), neighborhood=(p[7]), reviews=(p[8]),\
                                     overallSatisfaction=(p[9]), accommodates=(p[10]), bedroom=(p[11]), bathroom=(p[12]), price=(p[13]), minstay=(p[14]), lastModified=(p[15]),\
									 latitude=(p[16]), longitude=(p[17]), location=(p[18])))
									 
df = spark.createDataFrame(dataRdd)
df = df.select('roomId','roomType','neighborhood','reviews','overallSatisfaction','accommodates',\
			   'bedroom','price')

roomType = df.select('roomType').distinct().rdd.flatMap(lambda x: x).collect()
neighborhood = df.select('neighborhood').distinct().rdd.flatMap(lambda x: x).collect()

for val in roomType:
	df = df.withColumn(val, when(col("roomType") == val, 1). otherwise(0))
	
for val in neighborhood:
	df = df.withColumn(val, when(col("neighborhood") == val, 1). otherwise(0))

dataset = df.drop('roomType').drop('neighborhood')

names = dataset.schema.names
for col in names:
	dataset=dataset.withColumn(col, dataset[col].cast(IntegerType()))

dataset=dataset.withColumn('prices', dataset.price)
dataset = dataset.drop('price')

dataset=dataset.rdd.map(lambda x:(Vectors.dense(x[0:-1]), x[-1])).toDF(["features", "label"])
	
# Trains a k-means model.
kmeans = KMeans().setK(5).setSeed(1)
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset).select('prediction').take(15)
print(predictions)