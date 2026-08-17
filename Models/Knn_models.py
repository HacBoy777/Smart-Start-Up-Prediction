from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import json

class KnnModel:
    def train_knn(X_train,X_test,y_train,y_test):
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        print("\nKNN Model:")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print("ACCURACY OF KNN")
        train_acc=accuracy_score(y_train,y_pred_train)
        test_acc=accuracy_score(y_test,y_pred_test)
        print("KNN_Model Training Accuracy :", accuracy_score(y_train, y_pred_train))
        print("KNN_Model Testing Accuracy :", accuracy_score(y_test, y_pred_test))
        
        # Saving model
        pickle.dump(model, open("Models/knn_model.pkl", "wb"))
        # Json File
        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
            }
        with open("Models/knn_metrics.json", "w") as f:
            json.dump(metrics, f)
        return model
        