import numpy as np 
from DataLoader import DatasetLoader
from sklearn.neighbors import KNeighborsClassifier

def run_knn(X_train, y_train, X_test, y_test):

    for k in [1, 3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        print(f"k={k}: {acc:.4f}")
