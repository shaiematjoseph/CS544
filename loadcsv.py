# Load CSV using Pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

filename = 'auto-mpg.data.csv'

#read data into pandas dataframe
data = pd.read_csv(filename, names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin', 'Car Name'])

#drop null values
data = data[np.isfinite(data['Horsepower'])]

#print size of array
print(data.shape)

#plot  
sns.pairplot(data, x_vars=['Displacement','Horsepower','Weight','Acceleration'], y_vars='MPG', size=7, aspect=0.7, kind = 'reg')
plt.savefig('AP1.png')
plt.show('AP1.png')

lmD = smf.ols(formula='MPG ~ Displacement', data=data).fit()
print lmD.params
print lmD.summary()

lmH = smf.ols(formula='MPG ~ Horsepower', data=data).fit()
print lmH.params
print lmH.summary()

lmW = smf.ols(formula='MPG ~ Weight', data=data).fit()
print lmW.params
print lmW.summary()

lmA = smf.ols(formula='MPG ~ Acceleration', data=data).fit()
print lmA.params
print lmA.summary()

lm = smf.ols(formula='MPG ~ Displacement + Horsepower + Weight + Acceleration', data=data).fit()
print lm.params
print lm.summary()