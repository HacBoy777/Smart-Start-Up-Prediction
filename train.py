
## This file will train and save models for Logistic Regression, KNN and Random Forest Classifier.

import pandas as pd
import numpy as np
from Dataset.Preprocessing import DataPreprocessing
from Models.Logistic_model import LogisticModel
from Models.Knn_models import KnnModel
from Models.RandomForestClassifier import RandomForestModel
from Models.SVM_model import SVMModel
# df=pd.read_csv("Data/startup_dataset.csv")
# print(df.head())

def main():
    # print("Loading and Preprocessing")
    X_train,X_test,y_train,y_test=DataPreprocessing.load_and_preprocess()
    # print(X_train.shape)##(1600,11)
    # print(X_test.shape)##(400,11)
    # print(y_test.shape)##(400,)
    # print(y_train.shape)##(1600,)
    # print("\n Traing the logistic Reg")
    # model=LogisticModel()
    LR_model= LogisticModel.train_logistic(X_train,X_test,y_train,y_test)
    KNN_model = KnnModel.train_knn(X_train,X_test,y_train,y_test)
    RF_model = RandomForestModel.train_random_forest(X_train,X_test,y_train,y_test)
    SVM_model = SVMModel.train_svm(X_train,X_test,y_train,y_test)
    
if __name__=="__main__":
    main()

