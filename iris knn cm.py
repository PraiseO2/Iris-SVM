# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:35:10 2023

@author: CEO
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)

df['specie'] = iris.target
df['specie_name']  = df.specie.apply(lambda x: iris.target_names[x])
df0 = df[df['specie']==0]
df1 = df[df['specie']==1]
df2 = df[df['specie']==2]
'''
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'])
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'])
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'])
plt.xlabel('Sepal Width');plt.ylabel('Sepal Length')
plt.legend(df.specie_name)'''

X = df.drop(['specie', 'specie_name'], axis = 1)
y = df['specie']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
print(cm)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted');plt.ylabel('Actual')

from sklearn.metrics import classification_report
cr = classification_report(y_test, pred)








































