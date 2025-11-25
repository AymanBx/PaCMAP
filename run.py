import sys
import numpy as np
from knn import run_knn
from mrre import run_mrre
import matplotlib.pyplot as plt
from embedding import pca_embedding
from DataLoader import DatasetLoader
from continuity import run_continuity
from embedding import pacmap_embedding
from sklearn.neighbors import KNeighborsClassifier
from trustworthiness_eval import run_trustworthiness

## Set initial flags
knn = trust = mrre = continuity = plot = False

## Read user arguments 
dataset = sys.argv[1] if len(sys.argv) > 1 else input("Which dataset would you like to test with? Options: coil20 - coil20-npy - mnist - olivetti\n")
reducer_type = sys.argv[2] if len(sys.argv) > 2 else "pacmap"
eval_metric = sys.argv[3] if len(sys.argv) > 3 else "all"
plot = True if len(sys.argv) > 4 and sys.argv[4] == 'plot' else False

# if type(dataset) != list
match dataset:
    case 'coil20':
       loader = DatasetLoader(dataset,
                dataset_path='../datasets/coil-20'
                )
    case 'coil20-npy':
        loader = DatasetLoader('npy',
                data_path='../datasets/coil20/coil_20.npy',
                labels_path='../datasets/coil20/coil_20_labels.npy'
                )
    case 'mnist':
        loader = DatasetLoader(dataset,
                training_images='../datasets/MNIST/train-images.idx3-ubyte',
                training_labels='../datasets/MNIST/train-labels.idx1-ubyte',
                test_images='../datasets/MNIST/t10k-images.idx3-ubyte',
                test_labels='../datasets/MNIST/t10k-labels.idx1-ubyte'
                )
    case 'olivetti':
        loader = DatasetLoader('npy', 
                data_path='../datasets/olivetti/olivetti_faces.npy', 
                labels_path='../datasets/olivetti/olivetti_faces_target.npy'
                )

match reducer_type:
    case 'pca': reducer = pca_embedding
    case 'pacmap': reducer = pacmap_embedding

match eval_metric:
    case 'knn': knn = True
    case 'trustworthiness': trust = True
    case 'mrre': mrre = True
    case 'continuity': continuity = True
    case 'all': knn = trust = mrre = continuity = True

## loading preprocessed 
(X_train, y_train), (X_test, y_test) = loader.load_data()

## Flatten data
X_train = np.array(X_train)
X_test = np.array(X_test)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
print("X_train flattened shape:", X_train_flat.shape)
print("X_test flattened shape:", X_test_flat.shape)
print()

## Apply DR technique
print("Generating embeddings...")
X_train_embedded, X_test_embedded = reducer(X_train_flat, X_test_flat)
print("X_train embedded shape:", X_train_embedded.shape)
print("X_test embedded shape:", X_test_embedded.shape)
print()

## Evaluation Metrics
# run knn
if knn:
    print("Run KNN on ==> ", dataset)
    print("KNN Before:")
    run_knn(X_train_flat, y_train, X_test_flat, y_test)

    print("KNN After:")
    run_knn(X_train_embedded, y_train, X_test_embedded, y_test)
    print()

# Get trustworthiness
if trust:
    if dataset == 'mnist':
        print("Run Trustworthiness on ==> ", dataset)
        X_train_flat = X_train_flat.astype('float32')
        X_train_embedded = X_train_embedded.astype('float32')
        X_test_flat = X_test_flat.astype('float32')
        X_test_embedded = X_test_embedded.astype('float32')
        run_trustworthiness(X_train_flat, X_train_embedded, X_test_flat, X_test_embedded, embedding_func=reducer)
    else:
        print("Run Trustworthiness on ==> ", dataset)
        run_trustworthiness(X_train_flat, X_train_embedded, X_test_flat, X_test_embedded, embedding_func=reducer)

if mrre:
    run_mrre(X_train_flat, X_train_embedded)

if continuity:
    run_continuity(X_train_flat, X_train_embedded)


# visualize the reducer
if plot:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], cmap="Spectral", c=y_train, s=0.6)
    plt.savefig('after.png')