import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

## Importing the data
dataset = pd.read_csv("train.csv")

## Array Transformation
arr = dataset.iloc[:,[2,5,6,7,9,11,1]]

## Handeling NaN
impu = SimpleImputer(missing_values=np.nan, strategy = "mean")
arr = arr.values
arr[:,[0,1,2,3,4]] = impu.fit_transform(arr[:,[0,1,2,3,4]])

#Remvoing the entries from categorical columns

arr = pd.DataFrame(arr)
arr = arr.dropna()
arr = arr.values

#OneHot and LableEncoding

lb = LabelEncoder()
arr[:,-1] = lb.fit_transform(arr[:,-1])
transformer = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(),[5])],
    remainder='passthrough'
	)
arr = transformer.fit_transform(arr.tolist())

##X-Y Split
X = arr[:,:-1]
Y = arr[:,-1].astype('int')

#Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
#Reducing dimensions to 2

pca = PCA(n_components=2)
X = pca.fit_transform(X)

#Train Test Split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=10)

##Importing the libraries for creating the models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier # For support vector machines

##Random Forest
print("Random Forest\n")
model = RandomForestClassifier(n_estimators=10,criterion='entropy')
model.fit(X_train,Y_train)

print("Accuracy: ", model.score(X_test,Y_test))
print("Matrix: \n", confusion_matrix(Y_test, model.predict(X_test)))

print("\n\n")

##Gradient Boosting
print("Gradient Boosting\n")
model = GradientBoostingClassifier(n_estimators=10)
model.fit(X_train,Y_train)

print("Accuracy: ", model.score(X_test,Y_test))
print("Matrix: \n", confusion_matrix(Y_test, model.predict(X_test)))

print("\n\n")

##Ada Boosting
print("Adaptive Boosting\n")
model = AdaBoostClassifier(n_estimators=10)
model.fit(X_train,Y_train)

print("Accuracy: ", model.score(X_test,Y_test))
print("Matrix: \n", confusion_matrix(Y_test, model.predict(X_test)))

print("\n\n")


##Stacking Ensemble
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svc', make_pipeline(StandardScaler(),LinearSVC(random_state=42)))
]

model = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)


print("Stacking Classifier\n")
model.fit(X_train,Y_train)

print("Accuracy: ", model.score(X_test,Y_test))
print("Matrix: \n", confusion_matrix(Y_test, model.predict(X_test)))

print("\n\n")

'''
Output:

Random Forest

Accuracy:  0.702247191011236
Matrix:
 [[86 29]
 [24 39]]



Gradient Boosting

Accuracy:  0.7584269662921348
Matrix:
 [[105  10]
 [ 33  30]]



Adaptive Boosting

Accuracy:  0.7191011235955056
Matrix:
 [[92 23]
 [27 36]]

'''