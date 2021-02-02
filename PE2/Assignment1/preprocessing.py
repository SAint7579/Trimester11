import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## Importing the data
dataset = pd.read_csv("train.csv")
print(dataset.head())
print()
print(dataset.info())

## Array Transformation
arr = dataset.iloc[:,[2,5,6,7,9,11,1]]

## Handeling NaN
print("\n\nMissing values before imputing:\n",arr.isna().sum())
impu = SimpleImputer(missing_values=np.nan, strategy = "mean")
arr = arr.values
arr[:,[0,1,2,3,4]] = impu.fit_transform(arr[:,[0,1,2,3,4]])
print("\n\nMissing values after imputing:\n",pd.DataFrame(arr).isna().sum())

#Remvoing the entries from categorical columns

arr = pd.DataFrame(arr)
arr = arr.dropna()
print("\n\nMissing values after removal:\n",pd.DataFrame(arr).isna().sum())
arr = arr.values

#OneHot and LableEncoding

lb = LabelEncoder()
arr[:,-1] = lb.fit_transform(arr[:,-1])
print("\n\nAfter Label Encoding:\n",arr)

transformer = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(),[5])],
    remainder='passthrough'
	)
arr = transformer.fit_transform(arr.tolist())
print("\n\nAfter One Hot Encoding:\n",arr)

##X-Y Split
X = arr[:,:-1]
Y = arr[:,-1]

#Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("\n\nAfter Standard Scaling:\n",X)


#Train Test Split
arr_train,arr_test,Y_train,Y_test = train_test_split(arr,Y,test_size=0.2)

#Plotting Graphs
plt.pie([len(Y==1),len(Y==0)],labels=["Survived","Did not survive"],autopct='%1.1f%%')
plt.title("Class distribution")
plt.show()

dataset.groupby('Sex').Age.plot(kind='kde')
plt.title("Gender Wise Age distribution")
plt.legend()
plt.show()

plt.boxplot(arr[:,4])
plt.title("Ship Fair BoxPlot")
plt.legend()
plt.show()