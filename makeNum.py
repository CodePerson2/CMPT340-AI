# File makeing algorithm for PCA inpainting
# CMPT340 Project

import cv2
import glob
import numpy as np

# get path to images that will be put into numpy file
import pathlib
path = pathlib.Path(__file__).parent.absolute()

data = []
files = glob.glob(
    str(path) + "/trainingData/images-224/*.PNG")
count = 0

# rgb weights
rgb_weights = [0.2989, 0.5870, 0.1140]

for myFile in files:
    if count > 2000:
        break
    print(myFile)
    im = cv2.imread(myFile)
    im = np.dot(im[..., :3], rgb_weights)

    # push image to npy file
    data.append(cv2.resize(im, (64, 64), interpolation=cv2.INTER_CUBIC))
    count += 1

print('X_data shape:', np.array(data).shape)

# store npy file
np.save(str(path) + "/X.npy", data)
