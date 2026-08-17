import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
class DataPreprocessing:
    def  load_and_preprocess(features=[],labels=[],test_size=0.20,scale=True):
        df=pd.read_csv("Data/startup_dataset.csv")
        # print(df.head())
        le = LabelEncoder()
        df["IndustryType"] = le.fit_transform(df["IndustryType"])
        df["FounderEducation"] = le.fit_transform(df["FounderEducation"])
        df["ProductStage"] = le.fit_transform(df["ProductStage"])
        # print(df["IndustryType"][:5])
        # print(df["FounderEducation"][:5])
        # print(df["ProductStage"][:5])                                   
        X=df.drop("StartupSuccess",axis=1)
        y=df["StartupSuccess"]
        # print(X.shape)
        # print(y.shape)
        X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train,X_test,y_train,y_test

