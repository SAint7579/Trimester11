import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

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


#Train Test Split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=10)

##Importing the libraries for creating the models
from sklearn.svm import SVC # For support vector machines
from sklearn.ensemble import RandomForestClassifier #For random forest
from sklearn.naive_bayes import GaussianNB # For Naive Bayes
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


#No Skill Line
ns_probs = [0 for _ in range(len(Y_test))]
ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)

model_svm = SVC(probability=True)
model_rf = RandomForestClassifier(n_estimators=5)
model_nv = GaussianNB()
model_lr = LogisticRegression()
model_knn = KNeighborsClassifier()




print("Training Support Vector Machine")
model_svm.fit(X_train,Y_train)
print("Accuracy: ", model_svm.score(X_test,Y_test))

#Printing Roc-Auc Score
print("Roc-Auc Score: ", roc_auc_score(Y_test, model_svm.predict_proba(X_test)[:,1]))

#Making the Roc-Auc Plot
fpr, tpr, _ = roc_curve(Y_test, model_svm.predict_proba(X_test)[:,1])
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='SVM')
plt.legend()
plt.show()

#Breaking
print()

print("Training Random Forest Classifier")
model_rf.fit(X_train,Y_train)
print("Accuracy: ", model_rf.score(X_test,Y_test))

#Printing Roc-Auc Score
print("Roc-Auc Score: ", roc_auc_score(Y_test, model_rf.predict_proba(X_test)[:,1]))

#Making the Roc-Auc Plot
fpr, tpr, _ = roc_curve(Y_test, model_rf.predict_proba(X_test)[:,1])
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Random Forest')
plt.legend()
plt.show()

#Breaking
print()

print("Training Naive Bayes")
model_nv.fit(X_train,Y_train)
print("Accuracy: ", model_nv.score(X_test,Y_test))

#Printing Roc-Auc Score
print("Roc-Auc Score: ", roc_auc_score(Y_test, model_nv.predict_proba(X_test)[:,1]))

#Making the Roc-Auc Plot
fpr, tpr, _ = roc_curve(Y_test, model_nv.predict_proba(X_test)[:,1])
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Naive Bayes')
plt.legend()
plt.show()

#Breaking
print()

print("Training Logistic Regression")
model_lr.fit(X_train,Y_train)
print("Accuracy: ", model_lr.score(X_test,Y_test))

#Printing Roc-Auc Score
print("Roc-Auc Score: ", roc_auc_score(Y_test, model_lr.predict_proba(X_test)[:,1]))

#Making the Roc-Auc Plot
fpr, tpr, _ = roc_curve(Y_test, model_lr.predict_proba(X_test)[:,1])
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Logistic Regression')
plt.legend()
plt.show()

#Breaking
print()

print("Training KNN")
model_knn.fit(X_train,Y_train)
print("Accuracy: ", model_knn.score(X_test,Y_test))

#Printing Roc-Auc Score
print("Roc-Auc Score: ", roc_auc_score(Y_test, model_knn.predict_proba(X_test)[:,1]))

#Making the Roc-Auc Plot
fpr, tpr, _ = roc_curve(Y_test, model_knn.predict_proba(X_test)[:,1])
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='KNN Classifier')
plt.legend()
plt.show()

#Breaking
print()




