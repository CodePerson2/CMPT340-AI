# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import decomposition
from sklearn.linear_model import RidgeCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
import os
#print(os.listdir("C:/Users/matti/Downloads/archive/images-224/images-224/"))
import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.

x_load = np.load('C:/Users/matti/Downloads/archive/images-224/X.npy')
y_load = np.load('C:/Users/matti/Downloads/archive/images-224/X.npy')

img_size = 64
print('X_data shape:', np.array(x_load).shape)
image_index_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# one sample from each digits
# for each in range(4):
#     plt.figure(figsize=(8,5))
    
#     plt.imshow(x_load[each])
#     plt.axis('off')
#     title = "Sign " + str(each) 
#     plt.title(title)
# plt.show()


X = x_load.reshape((len(x_load), -1)) 
print(X.shape)
train = X
test = X[image_index_list]
n_pixels = X.shape[1]
# Upper half of the faces
X_train = train[:, :(n_pixels + 1) // 2]
X_test = test[:, :(n_pixels + 1) // 2]
# Lower half of the faces
y_train = train[:, n_pixels // 2:]
y_test = test[:, n_pixels // 2:]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# Fit estimators
estimator = RidgeCV()
start = time.time()   # returns in second
estimator.fit(X_train, y_train)
end = time.time()
print("Training time is "+ str(end - start) + " second.")
start = time.time()   # returns in second
y_test_predict = estimator.predict(X_test)
end = time.time()
print("Prediction time is "+ str(end - start) + " second.")



# Plot the completed faces
image_shape = (64, 64)
n_faces = 10
n_cols = 1


image_shape = (64, 64)
plt.figure(figsize=(8, 24))
for i in range(10):
    true_digits = np.hstack((X_test[i], y_test[i]))
    if i:
        sub = plt.subplot(10, 2, i * 2 + 1)
    else:
        sub = plt.subplot(10, 2, i * 2 + 1, title="true digits")
    
    sub.imshow(true_digits.reshape(image_shape),interpolation="nearest")
    sub.axis("off")
    completed_digits = np.hstack((X_test[i], y_test_predict[i]))

    if i:
        sub = plt.subplot(10, 2, i * 2 + 2 )

    else:
        sub = plt.subplot(10, 2, i * 2 + 2,title="RidgeCV")

    sub.imshow(completed_digits.reshape(image_shape),interpolation="nearest")
    sub.axis("off")

plt.show()