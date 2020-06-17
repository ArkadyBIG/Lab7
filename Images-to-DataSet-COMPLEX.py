from os.path import join as path_join
from os import listdir
from pickle import dump
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import random
import imutils
import threading

frame_size = (480, 640)

CLASS_NAMES = {
    'arm' : ['right', 'left'],
}

IMAGES_PATH = 'Raw-DataSet'

DUMP_PATH = 'Arrays-DataSet-(480,640)'

print('Loading noise...')
noise = cv2.imread(f'noise-0_4-2000-2000.png', 0).astype(np.float32)
noise /= 255



def noise_cut():
    x = random.randint(0, 2000 - frame_size[0])
    y = random.randint(0, 2000 - frame_size[1])
    return noise[x:x + frame_size[0], y:y + frame_size[1]].copy()


def rand_scale_rotate(img):
    img = imutils.rotate(img, random.randint(-50,50), scale=1.4)
    scale = 1 + random.random() * 2
    new_len = int(128 * scale)
    return cv2.resize(img, (new_len, new_len)), new_len

def prepare_batch():
    images = np.empty((0, *frame_size), np.float32)
    finger_lables = np.empty((0), np.float32)
    itr = 0
    for folder in listdir(IMAGES_PATH)[:1]:
        for arm_name in listdir(path_join(IMAGES_PATH, folder)):
            lable = arm_name.split('.')[0][-2:-1]
            for num in range(1):
                arm_path = path_join(IMAGES_PATH, folder, arm_name)
                arm_array = cv2.imread(arm_path, 0).astype(np.float32)
                arm_array, new_len = rand_scale_rotate(arm_array)
                arm_array /= 255

                img_array = noise_cut()

                x = random.randint(0, frame_size[0] - new_len - 1)
                y = random.randint(0, frame_size[1] - new_len - 1)

                img_array[x: x + new_len, y: y + new_len] = arm_array

                images = np.append(images, [img_array], axis=0)
                finger_lables = np.append(finger_lables, [lable])
            if finger_lables.shape[0] % 100 == 0:
                print(finger_lables.shape[0])
            if finger_lables.shape[0] == 5000:
                with open(path_join('Arrays-DataSet-(480,640)', f'{itr}-5000.images'), 'wb') as f:
                    dump(images, f)
                with open(path_join('Arrays-DataSet-(480,640)', f'{itr}-5000.lables'), 'wb') as f:
                    dump(finger_lables, f)
                itr += 1
                images = np.empty((0, *frame_size), np.float32)
                finger_lables = np.empty((0), np.float32)   
    with open(path_join('Arrays-DataSet-(480,640)', f'rest-{images.shape[0]}.images'), 'wb') as f:
        dump(images, f)
    with open(path_join('Arrays-DataSet-(480,640)', f'rest-{finger_lables.shape[0]}.lables'), 'wb') as f:
        dump(finger_lables, f)


prepare_batch()

