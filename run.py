import os
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
from sklearn.model_selection import train_test_split
from trustworthiness_eval import run_trustworthiness

## Set initial flags
knn = trust = mrre = continuity = plot = False
split = False # Do we want to apply the DR technique before or after splitting

## Read user arguments 
dataset = sys.argv[1] if len(sys.argv) > 1 else input("Which dataset would you like to test with? Options: coil20 - coil20-npy - mnist - olivetti\n")
reducer_type = sys.argv[2] if len(sys.argv) > 2 else "pacmap"
eval_metric = sys.argv[3] if len(sys.argv) > 3 else "all"
plot = True if len(sys.argv) > 4 and sys.argv[4] == 'plot' else False
dimention = int(sys.argv[5] if len(sys.argv) > 5 else 3) # Change to select the new dimention size after DR

## Setup log file
os.chdir(f"results/{reducer_type}/{dimention}-dim")
log = open(f"{dataset}_{reducer_type}-{dimention}.log", 'w')
os.chdir("../../..")

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
    case '20newsgroups':
        loader = DatasetLoader('20newsgroups')

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
if split:
    (X_train, y_train), (X_test, y_test) = loader.load_data(split)
else:
    X, y = loader.load_data(split)

    # We move the splitting to here to control how we apply the DR technique
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y if len(np.unique(y)) > 1 else None, random_state=1)

## Flatten data
X_train = np.array(X_train)
X_test = np.array(X_test)
print("X_train shape:", X_train.shape, file=log)
print("X_test shape:", X_test.shape, file=log)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
print("X_train flattened shape:", X_train_flat.shape, file=log)
print("X_test flattened shape:", X_test_flat.shape, file=log)
print(file=log)

## Apply DR technique
print("Generating embeddings...", file=log)
X_train_embedded, X_test_embedded = reducer(X_train_flat, X_test_flat, dimention)
print("X_train embedded shape:", X_train_embedded.shape, file=log)
print("X_test embedded shape:", X_test_embedded.shape, file=log)
print(file=log)

## Evaluation Metrics
# run knn
if knn:
    print("Run KNN on ==> ", dataset, file=log)
    print("KNN Before:", file=log)
    run_knn(X_train_flat, y_train, X_test_flat, y_test, log)

    print("KNN After:", file=log)
    run_knn(X_train_embedded, y_train, X_test_embedded, y_test, log)
    print(file=log)

# Get trustworthiness
if trust:
    if dataset == 'mnist':
        print("Run Trustworthiness on ==> ", dataset, file=log)
        X_train_flat = X_train_flat.astype('float32')
        X_train_embedded = X_train_embedded.astype('float32')
        X_test_flat = X_test_flat.astype('float32')
        X_test_embedded = X_test_embedded.astype('float32')
        run_trustworthiness(X_train_flat, X_train_embedded, X_test_flat, X_test_embedded, embedding_func=reducer, log_file=log)
    else:
        print("Run Trustworthiness on ==> ", dataset, file=log)
        run_trustworthiness(X_train_flat, X_train_embedded, X_test_flat, X_test_embedded, embedding_func=reducer, log_file=log)

if mrre:
    run_mrre(X_train_flat, X_train_embedded, log)

if continuity:
    run_continuity(X_train_flat, X_train_embedded, log)

log.close()

# visualize the reducer
if plot:
    os.chdir(f"results/{reducer_type}/{dimention}-dim")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_train_embedded[:, 0], X_train_embedded[:, 1], cmap="Spectral", c=y_train, s=0.6)
    plt.savefig(f'{dataset}_{reducer_type}-{dimention}.png')