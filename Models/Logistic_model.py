from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import json

class LogisticModel:
    def train_logistic(X_train,X_test,y_train,y_test):
        # print(X_train.shape)##(1600,11)
        # print(X_test.shape)##(400,11)
        # print(y_test.shape)##(400,)
        # print(y_train.shape)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        print("Logistic Regression Model: \n")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_acc=accuracy_score(y_train,y_pred_train)
        test_acc=accuracy_score(y_test,y_pred_test)
        print("ACCURACY OF LOGISTIC_MODEL")
        print("Logistic_Model Training Accuracy :", accuracy_score(y_train, y_pred_train))
        print("Logistic_Model Testing Accuracy :", accuracy_score(y_test, y_pred_test))
        
        # Saving model
        pickle.dump(model, open("Models/logistic_model.pkl", "wb"))
        ## Save metrics to json
        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
            }
        with open("Models/logistic_metrics.json", "w") as f:
            json.dump(metrics, f)
        return model
    