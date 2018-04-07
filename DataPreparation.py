from scipy import stats
import scipy
import pandas as pd
import numpy as np

# Read original csv files - 48852 x 96
listings = pd.read_csv('NYListings.csv')

# Extract columns that seem to have insight - 48852 x 15
ReducedData = listings[['neighbourhood_group_cleansed','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','price','minimum_nights','maximum_nights','availability_365','number_of_reviews','last_review','review_scores_rating']]

# Remove price outliers - 48406 x 15
DataPrice = listings[['price']]
OutlierData = listings[(np.abs(scipy.stats.zscore(DataPrice)) < 3).all(axis=1)]

OutlierData.to_csv('ReducedData.csv')