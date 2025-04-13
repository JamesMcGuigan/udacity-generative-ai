### Exercise 1: Classification of handwritten digits using an MLP
# https://learn.udacity.com/nanodegrees/nd608/parts/cd13303/lessons/96be0ec2-7a4b-49e8-b8c5-e71a1a927a07/concepts/43bd30b5-43d1-4e72-a132-c4f65bb2b3d9?lesson_tab=lesson
# https://scikit-learn.org/stable/datasets/loading_other_datasets.html#downloading-datasets-from-the-openml-org-repository
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
# https://numpy.org/doc/stable/reference/generated/numpy.array.html
# https://numpy.org/doc/stable/user/basics.broadcasting.html
# https://www.openml.org/search?type=data&sort=runs&status=active

# Load MNIST using sklearn.datasets.fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

# Load data from https://www.openml.org/d/554
#             == https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, parser="auto")

# Split into train and test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
print(f"Training set size: {len(X_train)}")
print(f"Test set size:     {len(X_test)}")

# Convert to numpy arrays and scale for the model
X_train = np.array(X_train) / 255
X_test  = np.array(X_test)  / 255
y_train = np.array(y_train, dtype=np.int8)
y_test  = np.array(y_test,  dtype=np.int8)

# Show the first 3 images
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(X_train[0:3], y_train[0:3])):
    plt.subplot(1, 3, index + 1)
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.title("Label: %s\n" % label, fontsize=20)


# Train an MLP classifier using sklearn.neural_network.MLPClassifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(25,25,),
    max_iter=100,
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
)
# Train the MLPClassifier
mlp.fit(X_train, y_train)