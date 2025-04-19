import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from sklearn.model_selection import train_test_split
#Loading Dataset

disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], inplace = True, axis = 1)
disease_df.rename(columns = {'male': 'Sex_male'}, inplace = True)

#dropping null values

disease_df.dropna(axis = 0, inplace= True)
disease_df

print(disease_df.TenYearCHD.value_counts())

#train test split

x =  np.array(disease_df[['age','Sex_male','cigsPerDay','totChol','sysBP', 'glucose']])

y = np.asarray(disease_df['TenYearCHD'])

x = preprocessing.StandardScaler().fit(x).transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.3, random_state = 4)
print ('Train set:', x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)

#Step 4: Exploratory Data Analysis of Heart Disease Dataset

plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df, hue='TenYearCHD', palette="BuGn_r", legend=False)
#plt.show()

laste = disease_df['TenYearCHD'].plot()
#plt.show()  # Remove the 'laste' argument

#Step 5: Fitting Logistic Regression Model for Heart Disease Prediction

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

from sklearn.metrics import accuracy_score
print('Accuracy of the model is =',
      accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix, classification_report

print('The details for the consuion matrix is =')
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test,y_pred)
conf_matrix = pd.DataFrame(data = cm,
                           columns = ['Predicted:0', 'Predicted:1'],
                           index = ['Actual:0', 'Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Greens')
plt.show()


