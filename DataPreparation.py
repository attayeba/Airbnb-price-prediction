from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType, DoubleType

from scipy import stats
import scipy
import pandas as pd
import numpy as np

'''
# Dataset is too large to be included
# Read original csv files - 48852 x 96
listings = pd.read_csv('NYListings.csv')

# Extract columns that seem to have insight - 48852 x 15
ReducedData = listings[['neighbourhood_group_cleansed','neighbourhood_cleansed','latitude','longitude','property_type','room_ype','accommodates','bathrooms','bedrooms','beds','bed_type','price','availability_365','review_scores_rating']]

ReducedData.to_csv('ReducedNy.csv')
'''
spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

# Load data,remove header and split at commas since its a CSV
lines = spark.read.text("ReducedNy.csv").rdd
head = lines.first()
lines = lines.filter(lambda x: x != head)
parts = lines.map(lambda row: row.value.split(","))
df = spark.createDataFrame(parts,['id','neighbourhoodGroup','neighbourhood','latitude','longitude','propertyType','roomType','accommodates','bathrooms','bedrooms','beds','bedType','price','availability','rating'])
dataset = df

# Get distinct neighbourhood and their count
neighbourhood = df.select('neighbourhood').rdd.map(lambda x: (x,1))
neighbourhood = neighbourhood.reduceByKey(lambda x,y: x+y).collect()

# insert count for each row
dataset = dataset.withColumn('count', dataset.id)
for val in neighbourhood:
	city = val[0][0]
	number = val[1]
	dataset = dataset.withColumn('count', when(col("neighbourhood") == city, number).otherwise(dataset["count"]))

# Filter out rows with less than 0.5% representation
dataset=dataset.withColumn('count', dataset['count'].cast(IntegerType()))
limit = (dataset.count())/200
CutOff = dataset.select('id','count').rdd.filter(lambda x: x[1] > limit).map(lambda x: x[0]).map(lambda x: x.split(","))
Remainder = spark.createDataFrame(CutOff,['id'])

# Join new dataset
dataset = dataset.join(Remainder, 'id')

# One-hot encode the neighbourhoodGroup values	
neighbourhoodGroup = dataset.select('neighbourhoodGroup').distinct().rdd.flatMap(lambda x: x).collect()
for val in neighbourhoodGroup:
	dataset = dataset.withColumn(val, when(col("neighbourhoodGroup") == val, 1). otherwise(0))

# One-hot encode the neighbourhood values
neighbourhood = dataset.select('neighbourhood').distinct().rdd.flatMap(lambda x: x).collect()
for val in neighbourhood:
	dataset = dataset.withColumn(val, when(col("neighbourhood") == val, 1). otherwise(0))

# One-hot encode the propertyType values
propertyType = dataset.select('propertyType').distinct().rdd.flatMap(lambda x: x).collect()
for val in propertyType:
	dataset = dataset.withColumn(val, when(col("propertyType") == val, 1). otherwise(0))
	
# One-hot encode the roomType values	
roomType = dataset.select('roomType').distinct().rdd.flatMap(lambda x: x).collect()
for val in roomType:
	dataset = dataset.withColumn(val, when(col("roomType") == val, 1). otherwise(0))
	
# One-hot encode the bedType values	
bedType = dataset.select('bedType').distinct().rdd.flatMap(lambda x: x).collect()
for val in bedType:
	dataset = dataset.withColumn(val, when(col("bedType") == val, 1). otherwise(0))

# Change column types to intergers
names = ['accommodates','bedrooms','beds','price','rating','availability']
for col in names:
	dataset=dataset.withColumn(col, dataset[col].cast(IntegerType()))
	
# Change column types to double
names = ['latitude','longitude','bathrooms']
for col in names:
	dataset=dataset.withColumn(col, dataset[col].cast(DoubleType()))

# Move price column to the end of the table	
dataset = dataset.drop('count','neighbourhoodGroup','neighbourhood','propertyType','roomType','bedType')

# Drop outlier values
pan = dataset.toPandas()
DataPrice = pan[['price']]
pan = pan[(np.abs(scipy.stats.zscore(DataPrice)) < 2.5).all(axis=1)]

# Save dataframe to csv to avoid repeating these steps
dataset = spark.createDataFrame(pan)
dataset.toPandas().to_csv('format.csv')