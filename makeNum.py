import cv2
import glob
import numpy as np

X_data = []
files = glob.glob ("C:/Users/matti/Downloads/archive/images-224/images-224/*.PNG")
count = 0
rgb_weights = [0.2989, 0.5870, 0.1140]

for myFile in files:
    if count > 2061:
        break
    print(myFile)
    image = cv2.imread (myFile)
    image = np.dot(image[...,:3], rgb_weights)
    X_data.append (cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC))
    count += 1

print('X_data shape:', np.array(X_data).shape)

np.save("C:/Users/matti/Downloads/archive/images-224/X.npy", X_data)