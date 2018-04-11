from scipy import stats
import scipy
import pandas as pd
import numpy as np

data = pd.read_csv('formated.csv')
DataPrice = data[['price']]
OutlierData = data[(np.abs(scipy.stats.zscore(DataPrice)) < 3).all(axis=1)]

# Extract columns that seem to have insight - 48852 x 15

Neigh = OutlierData['neighbourhoodGroup'] == 'Queens'
room = OutlierData['roomType'] == 'Private room'
acco = OutlierData['accommodates'] == 1
bath = OutlierData['bathrooms'] == 1
beds = OutlierData['bedrooms'] == 1
city = OutlierData['neighbourhood'] == 'Astoria'
type = OutlierData['propertyType'] == 'Apartment'
bed = OutlierData['bedType'] == 'Real Bed'
ava = OutlierData['availability'] > 0


RD = OutlierData[Neigh & city & room & acco & bath & beds & type & bed & ava]

RD.to_csv('Queens.csv')