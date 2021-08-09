#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
#Importing the data
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:, :-1]
y = dataset['Class']
#Splitting the data (fraud or normal)
norm = dataset.loc[dataset['Class']==0]
fraud = dataset.loc[dataset['Class']==1]

#visualising the data
sns.relplot(x='Amount', y='Time', hue='Class', data=(dataset))

#The train test split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size= 0.35)

#The logistic regression model
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#The model report
cm = confusion_matrix(y_test, y_pred)
rep = classification_report(y_test, y_pred)
