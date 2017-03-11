import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

df = pd.read_csv('ALP3Data.csv', header=None, names=['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])

df['Class'].replace(2, 0, inplace=True)
df['Class'].replace(4, 1, inplace=True)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

df['Mitoses']=[0 if w<=5 else 1 for w in df['Mitoses']]

#Split into training and test set
y = df['Mitoses']

X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print (X)
print('--> Split Train Data to Trainign and Test Sets')

# Update Status
print('--------- Now Trying Support Vector Machine Classifier ---------')

#Make Support Vector Classifier Pipeline
pipe_svc = Pipeline([('clf', SVC(random_state=1))])
print('--> Made Pipeline')

#Fit Pipeline to Data
pipe_svc.fit(X_train, y_train)
print('--> Fitted Pipeline to training Data')
scores = cross_val_score(estimator=pipe_svc,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

CorrectCount = 0
WrongCount = 0

for i in range(len(X_test)):
    pred = 0
    pred += pipe_svc.predict(np.array(X_test.values[i].reshape(1,-1), dtype=np.float64))
    guess = 0 if pred <= 1 else 1
    if guess == y_test.values[i]:        
        CorrectCount += 1 
    else:        
        WrongCount += 1

print('Combined Accuracy: ', round((CorrectCount/170)*100,0),'%')
print('Correct Predictions: ', CorrectCount)
print('Incorrect Predictions: ', WrongCount)


