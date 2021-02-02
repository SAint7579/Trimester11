import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.linalg import eigh 

## Importing the data
dataset = pd.read_csv("train.csv")
print(dataset.head())
print()
print(dataset.info())

## Array Transformation
arr = dataset.iloc[:,[2,5,6,7,9,1]]

## Handeling NaN
impu = SimpleImputer(missing_values=np.nan, strategy = "mean")
arr = arr.values
arr[:,[0,1,2,3,4]] = impu.fit_transform(arr[:,[0,1,2,3,4]])

##X-Y Split
X = arr[:,:-1]
Y = arr[:,-1]

#Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Getting Covariance matrix
cov = np.matmul(X.T, X)
print("\nShape of variance matrix = ", cov.shape)


#Finding Eigen Values
val,eigen = eigh(cov,eigvals=[3,4])
print(val)
print(eigen)

#Taking transpose
eigen = eigen.T
print("\nShape of eigen vector ", eigen.shape)

#Reducing Dimensions
new_coordinates = np.matmul(eigen, X.T)
new_coordinates = new_coordinates.T

print("\nReduced dimensions shape ",new_coordinates.shape)
print("\nReduced data ", new_coordinates)


#Ploting a scatter plot
plt.scatter(new_coordinates[:,0],new_coordinates[:,1])
plt.show()