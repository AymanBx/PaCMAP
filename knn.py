import numpy as np 
from DataLoader import DatasetLoader
from sklearn.neighbors import KNeighborsClassifier

def run_knn(db_name, X_train, y_train, X_test, y_test):
    print("Run KNN on ==> ", db_name)
    

    x_train = np.array(X_train)
    x_test = np.array(X_test)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    print("x_train flattened shape:", x_train_flat.shape)
    print("x_test flattened shape:", x_test_flat.shape)

    for k in [1, 3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train_flat, y_train)
        acc = knn.score(x_test_flat, y_test)
        print(f"k={k}: {acc:.4f}")
