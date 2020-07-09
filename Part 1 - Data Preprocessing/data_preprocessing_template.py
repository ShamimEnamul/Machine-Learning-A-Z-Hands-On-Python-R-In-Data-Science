#Data preprocessing..

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv') 
x = dataset.iloc[:, : -1].values
y = dataset.iloc[:, 3].values

#taking care of missing data
#After the version scikit-learn SimpleImputer/Imputer has been changed its location from sklearn.preprocessing to package sklearn.impute.
#So, kindly use the below line of code to import.
#from sklearn.impute import SimpleImputer
#Thank you!
#previous one: from sklearn.preprocessing import Imputer
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:,1:3])

#Encoding Catagorical Data  [not done yet]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_X = LabelEncoder();
x[:,0] = labelEncoder_X.fit_transform(x[:,0])
#onhotencoder = OneHotEncoder(categorical_features = [0])
onhotencoder = ColumnTransformer([x[:,0],OneHotEncoder(),[0]], remainder="passthrough")
x = onhotencoder.fit_transform(x).toarray()
