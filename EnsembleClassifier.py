# -*- coding: utf-8 -*-
'''
CT4101 MACHINE LEARNING - ASSIGNMENT 2
Prakhar Gurawa (20231064)
Yashitha Agarwal (20230091)

Code by: Combined effort (specific parts mentioned in comments)
'''

# We are importing all necessary libraries to implement our model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
from SupportVectorMachine import SVM

class EnsembleClassifier:
    
    # SVM and LR classifers are considered to make final model more robust
    # Scratch implementation of SVM : SupportVectorMachine.py
    # Scratch implementation of Logistic Regression : LogisticRegression.py
    
    # Code by: Yashitha Agarwal (20230091)
    def __init__(self,lrAlpha=0.01,svmAlpha=0.01,iterations=1000): # Constructor function to initalize individual hyperparameters for LR and SVM
        self.lrAlpha = lrAlpha
        self.svmAlpha =svmAlpha   
        self.iterations = iterations
        self.lrModel = None
        self.svmModel = None
    
    # Code by: Yashitha Agarwal (20230091)
    def fit(self,X,y):
        self.lrModel = LogisticRegression(self.lrAlpha,self.iterations)
        self.svmModel = SVM(self.svmAlpha,self.iterations)
        self.lrModel.fit(X,y)   # Fitting independent and dependent feature in Logistic Regression
        self.svmModel.fit(X,y)  # Fitting independent and dependent feature in Support Vector Machine Classifier
        
    # Code by: Prakhar Gurawa (20231064)
    def predict(self,X,y): 
        lrScore = self.lrModel.score(X,y)
        svmScore = self.svmModel.score(X,y)
        lrPrediction = self.lrModel.predict(X)      # Prediction of Logistic Regression model
        svmPrediction = self.svmModel.predict(X)    # Prediction of Support Vector Machine model
        # Currently we are considering only two algorithms and considering prediction of that higher score classifier in case of disagreement
        finalPrediction = list() # Storing predicted classes
        for i in range(len(lrPrediction)):
            if lrPrediction[i] == svmPrediction[i]:
                finalPrediction.append(lrPrediction[i]) # Case 1: Both LR ans SVM predict to same class
            else:                                       # Case 2: Disagreement between LR and SVM classifiers
                if lrScore > svmScore:
                    finalPrediction.append(lrPrediction[i])
                else:
                    finalPrediction.append(svmPrediction[i])              
        # Future work : If we have more than two algorithms we will make this as a voting classifier.
        # Mutiple classifer are considered and majority of prediction is taken as final prediction.
        return finalPrediction # Final Predictions using ensemble

    # Code by: Prakhar Gurawa (20231064)
    def score(self,X,y): # Function to calculate number of matches between actual classes and predicted classes by our model
        size = len(y)        
        return sum(self.predict(X,y)==y)/size # Number of matches divided by total inputs

# Code by: Yashitha Agarwal (20230091)
# importing dataset using pandas
filename = 'beer.txt'
header_list = ['caloric_value','nitrogen','turbidity','style','alcohol','sugars','biterness','beer_id','colour','degree_of_fermentation']
data = pd.read_csv(filename,sep='\t', header=None,dtype=str,names=header_list)

# creating dependent and independent features
X = data.drop(['style','beer_id'],axis=1).values
y = data['style'].values

# creating a pandas dataframe for storing results
predictions_final = pd.DataFrame(columns=['Iteration','Predicted Value','Actual Value'])


# Code by: Prakhar Gurawa (20231064)
# data stardaization pre-processing
def feature_scaling(X):
    X = X.astype(np.float)
    mean = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    X_scaled= (X -  mean) / sd
    return X_scaled

X = feature_scaling(X)

scores=list()
print("Ensemble Classifier Learning\n")
for i in range(10):
    X_train,X_test,y_train,y_test  = train_test_split(X, y, train_size = 2/3, shuffle = True) # split data 
    model = EnsembleClassifier() 
    model.fit(X_train, y_train)
    # passing y_test to predict since it is needed to get the score of LR and SVM model
    prediction = model.predict(X_test, y_test) 
    score = model.score(X_test,y_test)
    # storing all the predictions and actual values in the dataframe
    for (p , a) in zip(prediction, y_test):
        predictions_final = predictions_final.append({'Iteration': i, 'Predicted Value': p, 'Actual Value': a}, ignore_index=True)
    print("Accuracy ",i," = ",score)
    scores.append(score)
    
print("\nMean accuracy = ",np.mean(scores))

# output the results to csv file
predictions_final.to_csv('Ensemble_Results.csv', index=False)