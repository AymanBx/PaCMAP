import sys
import pacmap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


datapath = sys.argv[1]


# loading preprocessed coil_20 dataset
# you can change it with any dataset that is in the ndarray format, with the shape (N, D)
# where N is the number of samples and D is the dimension of each sample
X = np.load("./datasets/coil_20.npy", allow_pickle=True)
y = np.load("./datasets/coil_20_labels.npy", allow_pickle=True)
print("X: ", X.shape)
print("y: ", y.shape)

X = X.reshape(X.shape[0], -1)
print("X: ", X.shape)


# Can't plot images 
# visualize the data as is..?
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.scatter(X[:, 0], X[:, 1], cmap="Spectral", c=y, s=0.6)
# plt.savefig('before.png')

print("KNN Before:")
# Get KNN before DR applies
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y # stratify ensures proportions are the same before and after split.
)

for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"k={k}: {acc:.4f}")


# initializing the pacmap instance
# n_components: the number of dimension of the output. Default to 2.
# n_neighbors: the number of neighbors considered in the k-Nearest Neighbor graph.
# MN_ratio: the ratio of the number of mid-near pairs to the number of neighbors, n_MN = \lfloor n_neighbors * MN_ratio
# FP_ratio: the ratio of the number of further pairs to the number of neighbors, n_FP = \lfloor n_neighbors * FP_ratio
# Setting n_neighbors to "None" leads to an automatic choice shown below in "parameter" section
reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0) 

X_combined = np.vstack([X_train, X_test])

print("KNN After:")
# fit the data (The index of transformed data corresponds to the index of the original data)
X_transformed = reducer.fit_transform(X_combined, init="pca")
print("X: ", X_transformed.shape)


# Get KNN after the 
X_train = X_transformed[0:len(X_train)]
X_test = X_transformed[len(X_train):]

for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"k={k}: {acc:.4f}")


# visualize the reducer
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y, s=0.6)
plt.savefig('after.png')



# saving the reducer
# pacmap.save(reducer, "./coil_20_reducer")

# loading the reducer
# pacmap.load("./coil_20_reducer")