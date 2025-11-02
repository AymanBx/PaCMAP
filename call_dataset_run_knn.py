from DataLoader import DatasetLoader
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 

def run_knn(db_name, loader):
    print("Run KNN on ==> ", db_name)
    (X_train, y_train), (X_test, y_test) = loader.load_data()

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




coil_loader = DatasetLoader(
    'coil20',
    dataset_path='datasets/coil-20'
)
run_knn("coil20", coil_loader)


# Load MNIST dataset
mnist_loader = DatasetLoader(
    'mnist',
    training_images='datasets/MNIST/train-images.idx3-ubyte',
    training_labels='datasets/MNIST/train-labels.idx1-ubyte',
    test_images='datasets/MNIST/t10k-images.idx3-ubyte',
    test_labels='datasets/MNIST/t10k-labels.idx1-ubyte'
)
run_knn("mnist", mnist_loader)