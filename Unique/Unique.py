import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('ReducedData.csv')

# Extract columns that seem to have insight - 48852 x 15
ReducedData = data[['neighbourhood_group_cleansed','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','price']]

Neigh = ReducedData['neighbourhood_group_cleansed'] == 'Brooklyn'
room = ReducedData['room_type'] == 'Private room'
acco = ReducedData['accommodates'] == 1
bath = ReducedData['bathrooms'] == 1
beds = ReducedData['bedrooms'] == 1

RD = ReducedData[Neigh & room & acco & bath & beds]

RD.to_csv('unique1.csv')