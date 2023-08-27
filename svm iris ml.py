# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 18:59:37 2023

@author: CEO
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#print(dir(iris))
df['specie'] = iris.target
df['specie_name']  = df.specie.apply(lambda x: iris.target_names[x])
#print(df.head())
df0 = df[df['specie']==0]
df1 = df[df['specie']==1]
df2 = df[df['specie']==2]
'''
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'])
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'])
plt.xlabel('sepal length (cm)');plt.ylabel('sepal width (cm)')
plt.legend()'''

X = df.drop(['specie','specie_name'], axis='columns')
y = df.specie

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

from sklearn.svm import SVC
model = SVC()#C=100, gamma=1, kernel='linear')
model.fit(X_train, y_train)
score = model.score(X_train, y_train)
pred = model.predict(X_train)






























