import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv("data.csv")
# only numerical data (mean, count, min, max etc.)
# print(data.describe())
# print(data.info())

x = data.iloc[:, 0:3].values
y = data.iloc[:, -1].values

# taking care of missing data
# first method
# nan = not a number, substute with mean stategy
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# fit data
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)

# Encoding categorical data
# Encoding the Independent Variable

Ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(Ct.fit_transform(x))
print(x)

# Label Encoding

Le = LabelEncoder()
y = Le.fit_transform(y)
print(y)
country = data["Country"]
data["Country"] = Le.fit_transform(country)
print(data)

# Feature Scaling
# standardizes the data by removing the mean and scaling it to unit variance (z-score normalization)

Sc = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
x_train[0:,3:] = Sc.fit_transform(x_train[0:,3:])
x_test[0:,3:] = Sc.transform(x_test[0:,3:])
print(x_train)

