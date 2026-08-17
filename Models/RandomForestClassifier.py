from xml.parsers.expat import model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import json

class RandomForestModel:
    def train_random_forest(X_train,X_test,y_train,y_test):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_acc=accuracy_score(y_train,y_pred_train)
        test_acc=accuracy_score(y_test,y_pred_test)
        print("\nACCURACY OF RANDOM_FOREST")
        print("Random Forest Training Accuracy :", accuracy_score(y_train, y_pred_train))
        print("Random Forest Testing Accuracy :", accuracy_score(y_test, y_pred_test))
        
        # Saving model
        pickle.dump(model, open("Models/random_forest_model.pkl", "wb"))
        # Json File
        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
            }
        with open("Models/random_forest_metrics.json", "w") as f:
            json.dump(metrics, f)
            
        return model