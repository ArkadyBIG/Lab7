# import tensorflow as tf
# import pathlib
# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# np.set_printoptions(precision=4)

# dataset = tf.data.Dataset.range(8)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle
from PIL import Image

train_images = []
train_lables = []

raw_data_path = r'Raw DataSet\fingers'
data_path = r'DataSet\{}'

CATEGORY = ['train', 'test']

num = 0
for cat in CATEGORY:
    path = os.path.join(raw_data_path,cat)
    for img_name in os.listdir(path):
        num += 1
        if num % 100 == 0:
            print(num)
        img = mpimg.imread(os.path.join(path, img_name))
        plt.imshow(img, cmap='gray')
        plt.show()
        lable = img_name.split('.')[0][-2]
        train_images.append(img)
        train_lables.append(lable)

print('Writing...')
pickle.dump(train_images,open(data_path.format('Images.dataset'), 'wb'))
pickle.dump(train_lables,open(data_path.format('Lables.dataset'), 'wb'))
print('Finished!')



