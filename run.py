import sys
import pacmap
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import DatasetLoader
from knn import run_knn
from trustworthiness_eval import pacmap_embedding
from trustworthiness_eval import pca_embedding
from trustworthiness_eval import run_trustworthiness
from mrre import run_mrre
from continuity import run_continuity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


dataset = sys.argv[1] if len(sys.argv) > 1 else input("Which dataset would you like to test with?\n")
reducer_type = sys.argv[2] if len(sys.argv) > 2 else "pacmap"
eval_metric = sys.argv[3] if len(sys.argv) > 3 else None

# if type(dataset) != list
match dataset:
    case 'coil20':
       loader = DatasetLoader('coil20',
                dataset_path='../datasets/coil-20'
                )
    case 'coil20-npy':
        loader = DatasetLoader('npy',
                data_path='../datasets/coil20/coil_20.npy',
                labels_path='../datasets/coil20/coil_20_labels.npy'
                )
    case 'mnist':
        loader = DatasetLoader('mnist',
                training_images='datasets/MNIST/train-images.idx3-ubyte',
                training_labels='datasets/MNIST/train-labels.idx1-ubyte',
                test_images='datasets/MNIST/t10k-images.idx3-ubyte',
                test_labels='datasets/MNIST/t10k-labels.idx1-ubyte'
                )
    case 'olivetti':
        loader = DatasetLoader('npy', 
                data_path='../datasets/olivetti/olivetti_faces.npy', 
                labels_path='../datasets/olivetti/olivetti_faces_target.npy'
                )

match reducer_type:
    case 'pacmap': reducer = pacmap_embedding
    case 'pca': reducer = pca_embedding

# loading preprocessed 
(X_train, y_train), (X_test, y_test) = loader.load_data()

# Flatten if necessary / flatten is used also reducer, so instead it of doing it in each function, do it once
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# Get trustworthiness
run_trustworthiness(dataset, X_train_flat, y_train, X_test_flat, y_test, embedding_func=reducer)


# run knn
print("KNN Before:")
run_knn(dataset, X_train, y_train, X_test, y_test)


# initializing the DR instance
X_train_embedded, X_test_embedded = reducer(X_train_flat, X_test_flat)

print("KNN After:")
run_knn(dataset, X_train_embedded, y_train, X_test_embedded, y_test)

run_mrre(X_train_flat, X_train_embedded)
run_continuity(X_train_flat, X_train_embedded)

# visualize the reducer
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], cmap="Spectral", c=y_train, s=0.6)
plt.savefig('after.png')



# saving the reducer
# pacmap.save(reducer, "./coil_20_reducer")

# loading the reducer
# pacmap.load("./coil_20_reducer")