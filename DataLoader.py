import os
import struct
import numpy as np
from PIL import Image
from array import array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## TODO: DO WE WANT TO SPLIT HERE? 

class DatasetLoader:
    def __init__(self, dataset_type, **kwargs):
        """
        dataset_type: 'coil20', 'mnist', or 'npy'
        kwargs:
            For 'coil20': dataset_path='...'
            For 'mnist': training_images='...', training_labels='...', test_images='...', test_labels='...'
            For 'npy': data_path='...', labels_path='...'  (or a single npz file with both)
        """
        self.dataset_type = dataset_type.lower()
        self.kwargs = kwargs

        if self.dataset_type not in ('coil20', 'mnist', 'npy'):
            raise ValueError("dataset_type must be one of: 'coil20', 'mnist', 'npy'")

    def _load_coil20(self):
        dataset_path = self.kwargs.get('dataset_path')
        if not dataset_path:
            raise ValueError("Missing 'dataset_path' argument for COIL-20")

        images, labels = read_coil20_images_labels(dataset_path)
        X = np.array(images)
        y = np.array(labels)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=1)
        return (X_train, y_train), (X_test, y_test)


    def _load_mnist(self):
        req = ['training_images', 'training_labels', 'test_images', 'test_labels']
        for r in req:
            if r not in self.kwargs:
                raise ValueError(f"Missing argument '{r}' for MNIST dataset")

        x_train, y_train = read_mnist_images_labels(
            self.kwargs['training_images'], self.kwargs['training_labels']
        )
        x_test, y_test = read_mnist_images_labels(
            self.kwargs['test_images'], self.kwargs['test_labels']
        )
        return (x_train, y_train), (x_test, y_test)

    # -------------------------------
    # Generic NumPy Dataset (.npy / .npz)
    # -------------------------------
    def _load_npy(self):
        data_path = self.kwargs.get('data_path')
        labels_path = self.kwargs.get('labels_path')
        if not data_path:
            raise ValueError("Missing 'data_path' argument for npy dataset")

        # Handle .npy or .npz
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            X = data['X'] if 'X' in data else data['data']
            y = data['y'] if 'y' in data else data.get('labels', None)
        else:
            X = np.load(data_path, allow_pickle=True)
            y = np.load(labels_path, allow_pickle=True) if labels_path else None

        # If no labels, generate dummy ones
        if y is None:
            y = np.zeros(len(X), dtype=int)

        # Normalize or reshape if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y if len(np.unique(y)) > 1 else None, random_state=1
        )
        return (X_train, y_train), (X_test, y_test)

    # -------------------------------
    # Call Loader
    # -------------------------------
    def load_data(self):
        if self.dataset_type == 'coil20':
            return self._load_coil20()
        elif self.dataset_type == 'mnist':
            return self._load_mnist()
        elif self.dataset_type == 'npy':
            return self._load_npy()


# -------------------------------
# COIL-20
# -------------------------------
def read_coil20_images_labels(dataset_path):
    images, labels = [], []
    for class_folder in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm')):
                    try:
                        img_path = os.path.join(class_path, filename)
                        img = Image.open(img_path).convert("L")
                        images.append(np.array(img))
                        labels.append(class_folder)
                    except Exception as e:
                        print(f"Skipped {filename}: {e}")
    return images, labels


# -------------------------------
# MNIST
# -------------------------------
def read_mnist_images_labels(images_filepath, labels_filepath):
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Label file magic number mismatch: {magic}")
        labels = array("B", file.read())
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Image file magic number mismatch: {magic}")
        image_data = array("B", file.read())
    images = []
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(rows, cols)
        images.append(img)
    return images, labels