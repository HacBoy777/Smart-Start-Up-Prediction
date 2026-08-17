from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import json

class SVMModel:
    def train_svm(X_train, X_test, y_train, y_test):
        # Pipeline for scaling and SVM
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf', 
                probability=True
            ))
        ])
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_acc=accuracy_score(y_train,y_pred_train)
        test_acc=accuracy_score(y_test,y_pred_test)
        print("\nACCURACY OF SVM")
        print("SVM Training Accuracy :", accuracy_score(y_train, y_pred_train))
        print("SVM Testing Accuracy :", accuracy_score(y_test, y_pred_test))
        
         # Saving model
        pickle.dump(model, open("Models/svm_model.pkl", "wb"))
        ## Save metrics to json
        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
            }
        with open("Models/svm_metrics.json", "w") as f:
            json.dump(metrics, f)
            
        return model