# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""#STEPS
## 0. Preprocess + EDA + Feature Selection
## 1. Extract input and output cols
## 2. Scale the values.
## 3. Train the split
## 4. Train the model.
## 5. Evaluate the model/ slection
## 6. Deploy the Model/
"""

df = pd.read_csv('/placement.csv')
df.head()

df.shape

df = df.iloc[:,1:]
df.head()

plt.scatter(df['cgpa'],df['iq'],c=df['placement'])

X = df.iloc[:,0:2]
y = df.iloc[:,-1]

y.shape

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

X_train

y_train

X_test

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_train

X_test = scaler.transform(X_test)
X_test

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

#model training
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

y_test

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

!pip install mlxtend

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train,y_train.values, clf=clf, legend=2)

import pickle

pickle.dump(clf,open('model.pkl','wb'))
