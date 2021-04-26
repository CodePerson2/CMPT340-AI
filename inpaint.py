# CMPT340 Inpainting using PCA
# Original use: https://www.kaggle.com/drikee/tutorial-pca-intuition-and-image-completion

# imports
import numpy as np
import pandas as pd
# plotting
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import decomposition
from sklearn.linear_model import RidgeCV


import time
import os
import warnings
import pathlib
path = pathlib.Path(__file__).parent.absolute()
warnings.filterwarnings('ignore')

def printImg():
    for each in range(4):
        plt.figure(figsize=(8, 5))

        plt.imshow(x_load[each])
        plt.axis('off')
        title = "Sample " + str(each)
        plt.title(title)
    plt.show()


def MakeData(x_load, image_index_list):
    X = x_load.reshape((len(x_load), -1))
    train = X
    test = X[image_index_list]
    n_pixels = X.shape[1]
    # Upper half of the chest
    X_train = train[:, :(n_pixels + 1) // 2]
    X_test = test[:, :(n_pixels + 1) // 2]
    # Lower half of the chest
    y_train = train[:, n_pixels // 2:]
    y_test = test[:, n_pixels // 2:]

    print("Dimensions of training data" + str(X_train.shape))
    

    return X, X_train, y_train, X_test, y_test

#PCA of RidgeVC regression
def pca(X):
    # dimensions
    (n_samples, n_features) = X.shape
    estimator = decomposition.PCA(
        n_components=4, svd_solver='randomized', whiten=True)
    estimator.fit(X)
    components_ = estimator.components_
    images = components_[:4]
    plt.figure(figsize=(6, 5))
    for i, comp in enumerate(images):
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape((64, 64)),
                interpolation='nearest', vmin=-vmax, vmax=vmax)

        # ploting data
        plt.xticks(())
        plt.yticks(())
    plt.title("Principle Component Analysis")
    plt.savefig('PCA.png')
    plt.show()

# loading the path to numpy files
# change if required
x_load = np.load(str(path) + '/X.npy')

img_size = 64
print('X shape:', np.array(x_load).shape)
image_index_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# print example of data used
printImg()

# get all arrays for the synthesis of data
X, X_train, y_train, X_test, y_test = MakeData(x_load, image_index_list)


# Fit estimators
estimator = RidgeCV()
start = time.time()   # returns in second
estimator.fit(X_train, y_train)
end = time.time()
print("Time to Train " + str(end - start) + " second.")
start = time.time()   # returns in second
y_test_predict = estimator.predict(X_test)
end = time.time()
print("Time to predict " + str(end - start) + " second.")


# Plot the completed xrays
n_faces = 10
n_cols = 1


shape = (64, 64)
plt.figure(figsize=(8, 24))
for i in range(10):
    true_digits = np.hstack((X_test[i], y_test[i]))

    # plotting data, gives title to entry 1
    if i > 0:
        sub = plt.subplot(10, 2, i * 2 + 1)
    else:
        sub = plt.subplot(10, 2, i * 2 + 1, title="True Image")

    sub.imshow(true_digits.reshape(shape), interpolation="nearest")
    sub.axis("off")
    completed_digits = np.hstack((X_test[i], y_test_predict[i]))

    # plotting data, gives title to entry 1
    if i > 0:
        sub = plt.subplot(10, 2, i * 2 + 2)

    else:
        sub = plt.subplot(10, 2, i * 2 + 2, title="Lower Half Inpainted")

    sub.imshow(completed_digits.reshape(shape), interpolation="nearest")
    sub.axis("off")

plt.show()


#show what pca is seeing
pca(X)