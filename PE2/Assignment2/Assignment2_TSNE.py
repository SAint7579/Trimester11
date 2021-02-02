import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
print("Data Shape ",X.shape)

#Creating t-SNE object
tsne = TSNE(n_components = 2, random_state = 0)
X = tsne.fit_transform(X)
print("\nReduced Data Shape ",X.shape)

print("\nReduced Data ",X)

#Plotting
plt.scatter(X[:,0],X[:,1])
plt.show()
