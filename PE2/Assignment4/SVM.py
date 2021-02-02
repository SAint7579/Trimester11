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
from sklearn.svm import SVC # For support vector machines

#SVM with linear kernel
model= SVC(kernel = 'linear', probability=True)

print("Training Support Vector Machine with linear kernel")
model.fit(X_train,Y_train)

#Accuracy
print("Accuracy: ", model.score(X_test,Y_test))

#Confusion matrix
print("Confusion Matrix: \n", confusion_matrix(Y_test,model.predict(X_test)))

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM Linear')
plt.xlabel('Reduced Feature 1')
plt.ylabel('Reduced Feature 2')
plt.legend()
plt.show()


#Breaking
print()

#SVM with rbf kernel
model= SVC(kernel = 'rbf', probability=True)

print("Training Support Vector Machine with RBF Kernel")
model.fit(X_train,Y_train)

#Accuracy
print("Accuracy: ", model.score(X_test,Y_test))

#Confusion matrix
print("Confusion Matrix: \n", confusion_matrix(Y_test,model.predict(X_test)))

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM RBF')
plt.xlabel('Reduced Feature 1')
plt.ylabel('Reduced Feature 2')
plt.legend()
plt.show()


#Breaking
print()

#SVM with polynomial kernel
model= SVC(kernel = 'poly', probability=True)

print("Training Support Vector Machine with Polynomial Kernel")
model.fit(X_train,Y_train)

#Accuracy
print("Accuracy: ", model.score(X_test,Y_test))

#Confusion matrix
print("Confusion Matrix: \n", confusion_matrix(Y_test,model.predict(X_test)))

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM POLY')
plt.xlabel('Reduced Feature 1')
plt.ylabel('Reduced Feature 2')
plt.legend()
plt.show()


#Breaking
print()











